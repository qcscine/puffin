# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from typing import Union, Callable, List, Optional, Tuple, Iterator, Any, TYPE_CHECKING, Type
from typing_extensions import TypeGuard, TypeVar, ParamSpec, Concatenate

import shutil
import tarfile
import os
import sys
import ctypes
import io
import time

from scine_puffin.utilities.imports import module_exists, requires, MissingDependency
from scine_puffin.config import Configuration

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")

T = TypeVar("T", bound='Job')
P = ParamSpec("P")
AnyReturnType = TypeVar("AnyReturnType")


def is_configured(run: Callable[Concatenate[T, P], AnyReturnType]) \
        -> Callable[Concatenate[T, P], AnyReturnType]:
    """
    A decorator to check if the job has been configured before running a method
    """

    @wraps(run)
    def _impl(self: T, *args: P.args, **kwargs: P.kwargs) -> AnyReturnType:
        if self.check_configuration(self):
            return run(self, *args, **kwargs)
        else:
            raise RuntimeError("Job has not been configured properly")

    return _impl


def job_configuration_wrapper(run: Callable):
    """
    A wrapping function for the run method of a Job instance

    - Configures run (setting class members for specific database Calculation)
    - Additionally Try/Catch safety to avoid dying pending jobs without error
    """

    @wraps(run)
    def _impl(self, manager: db.Manager, calculation: db.Calculation, config: Configuration):
        self.configure_run(manager, calculation, config)
        try:
            success = run(self, manager, calculation, config)
        except BaseException as e:
            # this should not happen in a properly written run function, but added here for additional safety
            comment = calculation.get_comment()
            comment += "\n" + str(e)
            calculation.set_comment(comment)
            success = False
        if not success:
            # additional safety that every failed job gets also a failed status
            calculation.set_status(db.Status.FAILED)
        return success

    return _impl


class Job(ABC):
    """
    A common interface for all jobs in/carried out by a Puffin
    """

    work_dir: str
    stdout_path: str = "stdout"
    stderr_path: str = "stderr"
    config: Configuration
    _id: db.ID
    _calculation: db.Calculation
    _manager: db.Manager
    _calculations: db.Collection
    _compounds: db.Collection
    _elementary_steps: db.Collection
    _properties: db.Collection
    _reactions: db.Collection
    _structures: db.Collection
    _flasks: db.Collection

    @job_configuration_wrapper
    @abstractmethod
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        """
        Runs the actual job.
        This function has to be implemented by any job that shall be added to
        Puffins job portfolio.

        Parameters
        ----------
        manager : db.Manager (Scine::Database::Manager)
            The manager/database to work on/with.
        calculation : db.Calculation (Scine::Database::Calculation)
            The calculation that triggered the execution of this job.
        config : scine_puffin.config.Configuration
            The configuration of Puffin.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def required_programs() -> List[str]:
        """
        This function has to be implemented by any job that shall be added to
        Puffins job portfolio.

        Returns
        -------
        requirements : List[str]
            A list of names of programs/packages that are required for the
            execution of the job.
        """
        raise NotImplementedError

    @classmethod
    def optional_settings_doc(cls) -> str:
        """
        Returns a docstring description of the available optional settings of the job.
        To be implemented by child classes if applicable
        """
        return ""

    @classmethod
    def generated_data_docstring(cls) -> str:
        """
        Returns a docstring description of the generated data of the job.
        """
        return "**Generated Data**\n"

    @staticmethod
    def general_calculator_settings_docstring() -> str:
        return """
        Additionally, all settings that are recognized by the SCF program chosen.
        are also available. These settings are not required to be prepended with
        any flag.

        Common examples are:

        max_scf_iterations : int
           The number of allowed SCF cycles until convergence.
        """

    @classmethod
    def required_packages_docstring(cls) -> str:
        required = cls.required_programs()
        docs = "\n**Required Packages**\n"
        if "database" in required:
            docs += "  - SCINE: Database (present by default)\n"
        if "molassembler" in required:
            docs += "  - SCINE: Molassembler (present by default)\n"
        if "readuct" in required:
            docs += "  - SCINE: ReaDuct (present by default)\n"
        if "utils" in required:
            docs += "  - SCINE: Utils (present by default)\n"
        docs += "  - A program implementing the SCINE Calculator interface, e.g. Sparrow\n"
        return docs

    def prepare(self, job_dir: str, id: db.ID) -> None:
        """
        Prepares the actual job.
        This function has to be implemented by any job that shall be added to
        Puffins job portfolio.

        Parameters
        ----------
        job_dir : str
            The path to the directory in which all jobs are executed.
        id : db.ID (Scine::Database::ID)
            The calculation that triggered the execution of this job.
        """
        self._id = id
        self.work_dir = os.path.abspath(os.path.join(job_dir, id.string()))
        if self.work_dir and not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def archive(self, archive: str) -> None:
        """
        Archives all files existent in the job's directory into tarball named
        with the job's ID. The tarball is then moved to the given destination.

        Parameters
        ----------
        archive : str
            The path to move the resulting tarball to.
        """
        if not self.work_dir or not os.path.exists(self.work_dir):
            sys.stderr.write(f"The job directory {self.work_dir} does not exist, cannot archive.\n")
            return
        if not self._id:
            sys.stderr.write(f"The job {self.__class__.__name__} has no ID, cannot archive.\n")
            return
        basedir = os.path.dirname(self.work_dir)
        # Tar the folder
        tar_gen_path = os.path.join(basedir, self._id.string() + ".tar.gz")
        tar_archive_path = os.path.join(archive, self._id.string() + ".tar.gz")
        with tarfile.open(tar_gen_path, mode="w:gz") as tar:
            tar.add(self.work_dir, arcname=self._id.string(), recursive=True)
        # Move the tar ball
        if archive and not os.path.exists(archive):
            os.makedirs(archive)
        shutil.move(tar_gen_path, tar_archive_path)

    def clear(self) -> None:
        """
        Clears the directory in which the job was run.
        """
        if not self.work_dir or not os.path.exists(self.work_dir):
            sys.stderr.write(f"The job directory {self.work_dir} does not exist, cannot remove anything.\n")
            return
        shutil.rmtree(self.work_dir)

    @is_configured
    def verify_connection(self) -> None:
        """
        Verifies the connection to the database.
        Returns only if a connection is established, if it is not, the function
        will attempt to generate a connection every 10 seconds, indefinitely.

        Notes
        -----
        * Requires run configuration
        """
        # Retry connection until killed
        while not self._manager.is_connected():
            time.sleep(10)

    @requires("database")
    def store_property(
        self,
        properties: db.Collection,
        property_name: str,
        property_type: str,
        data: Any,
        model: db.Model,
        calculation: db.Calculation,
        structure: db.Structure,
        replace: bool = True,
    ) -> Optional[db.Property]:
        """
        Adds a single property into the database, connecting it with a given
        structure and calculation (it's results section) and also

        Parameters
        ----------
        properties : db.Collection (Scine::Database::Collection)
            The collection housing all properties.
        property_name : str
            The name (key) of the new property, e.g. ``electronic_energy``.
        property_type : str
            The type of property to be added, e.g. ``NumberProperty``.
        data : Any (According to 'property_type')
            The data to be stored in the property, the type of this object is
            dependent on the type of property requested. A ``NumberProperty``
            will require a ``float``, a ``VectorProperty`` will require a
            ``List[float]``, etc.
        model : db.Model (Scine::Database::Model)
            The model used in the calculation that resulted in this property.
        calculation : db.Calculation (Scine::Database::Calculation)
            The calculation that resulted in this property.
            The calculation has to be linked to its collection.
        structure : db.Structure (Scine::Database::Structure)
            The structure for which the property is to be added. The properties
            field of the structure will receive an additional entry, or have
            an entry replaced, based on the options given to this function.
            The structure has to be linked to its collection.
        replace : bool
            If true, replaces an existing property (identical name and model)
            with the new one. This option is true by default.
            If false, doesnothing in the previous case, and returns ``None``

        Returns
        -------
        property : Derived of db.Property (Scine::Database::Property)
            The property, a derived class of db.Property, linked to the
            properties' collection, or ``None`` if no property was generated due
            to duplication.
        """
        existing = self.check_duplicate_property(structure, properties, property_name, model)
        if existing and replace:
            class_ = getattr(db, property_type)
            db_property = class_(existing)
            db_property.link(properties)
            db_property.set_data(data)
            db_property.set_calculation(calculation.id())
            results = calculation.get_results()
            results.add_property(db_property.id())
            calculation.set_results(results)
        elif existing:
            return None
        else:
            class_ = getattr(db, property_type)
            db_property = class_()
            db_property.link(properties)
            db_property.create(model, property_name, structure.id(), calculation.id(), data)
            structure.add_property(property_name, db_property.id())
            results = calculation.get_results()
            results.add_property(db_property.id())
            calculation.set_results(results)
        return db_property

    @staticmethod
    def check_duplicate_property(structure: db.Structure, properties: db.Collection, property_name: str,
                                 model: db.Model) -> Union[db.ID, bool]:
        """
        Checks for a property that is an exact match for the one queried here.
        Exact match meaning that key and model both are matches.

        Parameters
        ----------
        properties : db.Collection (Scine::Database::Collection)
            The collection housing all properties.
        property_name : str
            The name (key) of the queried property, e.g. ``electronic_energy``.
        model : db.Model (Scine::Database::Model)
            The model used in the calculation that resulted in this property.
        structure : db.Structure (Scine::Database::Structure)
            The structure to be checked in. The structure has to be linked to
            its collection.

        Returns
        -------
        ID : db.ID (Scine::Database::ID)
            Returns ``False`` if there is no existing property like the one
            queried or the ID of the first duplicate.
        """
        hits = structure.query_properties(property_name, model, properties)
        if len(hits) > 0:
            return hits[0]
        return False

    def configure_run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> None:
        """
        Configures a job for a given Calculation to do tasks in the run function

        Parameters
        ----------
        manager : db.Manager (Scine::Database::Manager)
            The manager of the database holding all collections
        calculation : db.Calculation (Scine::Database::Calculation)
            The calculation to be performed
        config : Configuration
            The configuration of the Puffin doing the job
        """
        self._manager = manager
        self._calculations = manager.get_collection("calculations")
        self._compounds = manager.get_collection("compounds")
        self._elementary_steps = manager.get_collection("elementary_steps")
        self._properties = manager.get_collection("properties")
        self._reactions = manager.get_collection("reactions")
        self._structures = manager.get_collection("structures")
        self._flasks = manager.get_collection("flasks")
        self._calculation = calculation
        if not self._calculation.has_link():
            self._calculation.link(self._calculations)
        self.config = config

    @classmethod
    def check_configuration(cls: Type[T], instance: T) -> TypeGuard[T]:
        return hasattr(instance, "work_dir") and \
            hasattr(instance, "stdout_path") and \
            hasattr(instance, "stderr_path") and \
            hasattr(instance, "config") and \
            hasattr(instance, "_id") and \
            hasattr(instance, "_calculation") and \
            hasattr(instance, "_manager") and \
            hasattr(instance, "_calculations") and \
            hasattr(instance, "_compounds") and \
            hasattr(instance, "_elementary_steps") and \
            hasattr(instance, "_properties") and \
            hasattr(instance, "_reactions") and \
            hasattr(instance, "_structures") and \
            hasattr(instance, "_flasks")

    @is_configured
    def capture_raw_output(self) -> Tuple[str, str]:
        """
        Tries to capture the raw output of the calculation context and save it in the raw_output field of the
        configured calculation. This should never throw.

        Notes
        -----
        * Requires run configuration
        """
        if any(
            path is None or not os.path.exists(path)
            for path in [self.stderr_path, self.stdout_path]
        ):
            sys.stderr.write(
                "Job paths were not set directly before capturing the output. Raw output cannot be saved"
            )
            return "", ""
        raw_out = open(self.stdout_path, "r").read()
        raw_err = open(self.stderr_path, "r").read()
        try:
            self._calculation.set_raw_output(raw_out + raw_err)
        except RuntimeError:  # large raw output can crash db and then puffin
            self._calculation.set_raw_output(
                "Too large raw output, check out archive/error directory if puffin has one."
            )
            return "", raw_err
        return raw_out, raw_err

    @requires('database')
    @is_configured
    def complete_job(self) -> None:
        """
        Saves the executing Puffin, changes status to db.Status.COMPLETE.
        """
        self.capture_raw_output()
        self._calculation.set_executor(self.config["daemon"]["uuid"])
        self._calculation.set_status(db.Status.COMPLETE)

    @is_configured
    @requires('database')
    def fail_job(self) -> None:
        """
        Saves the executing Puffin, changes status to db.Status.FAILED.
        """
        self._calculation.set_executor(self.config["daemon"]["uuid"])
        self._calculation.set_status(db.Status.FAILED)
        _, error = self.capture_raw_output()
        self._calculation.set_comment(error)

    def postprocess_calculation_context(self) -> bool:
        """
        Postprocesses a calculation context, pushing all errors and comments.

        Returns
        -------
            `True` if the job succeeded, `False` otherwise.
        """
        if os.path.isfile(self.failed_file()) or not os.path.isfile(self.success_file()):
            self.fail_job()
            return False
        else:
            self.complete_job()
            return True

    def failed_file(self) -> str:
        """
        Returns the path to the file indicating a failed calculation, None if job has not been prepared
        """
        if self.work_dir is None:
            return ""
        return os.path.join(self.work_dir, "failed")

    def success_file(self) -> str:
        """
        Returns the path to the file indicating a successful calculation, empty string if job has not been prepared
        """
        if self.work_dir is None:
            return ""
        return os.path.join(self.work_dir, "success")


@contextmanager
def calculation_context(job, stdout_name: str = "output", stderr_name: str = "errors",
                        debug: Optional[bool] = None) -> Iterator:
    """
    A context manager for a types of calculations that are run externally and
    may fail, dump large amounts of files or do other nasty things.

    The executed code will be run in the working directory of the given job, the first
    exceptions will be caught and appended to the error output, the context will
    then close and leave behind a file called ``failed`` in the scratch
    directory. If no exceptions are thrown, a file called ``success`` will be
    generated in the scratch directory.

    The output redirector part has been adapted from
    `here <https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/>`_
    [access date Jun 25th, 2019]

    Parameters
    ----------
    job : Job
        The job holding the working directory and receiving the output and error paths
    stdout_name : str
        Name of the file that the stdout stream should be redirected to.
        The file will be generated in the given scratch directory.
    stderr_name : str
        Name of the file that the stderr stream should be redirected to.
        The file will be generated in the given scratch directory.
    debug : bool
        If not given, will be taken from Job Configuration (config['daemon']['mode'])
        If true, runs in debug mode, disabling all redirections.

    Returns
    -------
        The context generates three files in the ``job.work_dir`` beyond any other
        ones generated by the executed code.
        The first two are the redirected output streams ``stderr` and ``stdout``
        (the name of these files are set by the context's arguments), the third
        file is either called ``failed`` or ``success`` depending on the
        occurrence of an exception in the executed code or not.
    """
    workdir = job.work_dir
    job.stdout_path = os.path.join(workdir, stdout_name)
    job.stderr_path = os.path.join(workdir, stderr_name)
    if debug is None:
        debug = bool(job.config["daemon"]["mode"].lower() == "debug")
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()
    # The original fd stderr points to. Usually 2 on POSIX systems.
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(
            os.fdopen(original_stdout_fd, "wb"), line_buffering=True
        )

    def _redirect_stderr(to_fd):
        """Redirect stderr to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()
        # Make original_stderr_fd point to the same file as to_fd
        os.dup2(to_fd, original_stderr_fd)
        # Create a new sys.stderr that points to the redirected fd
        sys.stderr = io.TextIOWrapper(
            os.fdopen(original_stderr_fd, "wb"), line_buffering=True
        )

    if debug:
        # Do nothing while in debug mode
        prevdir = os.getcwd()
        os.chdir(os.path.expanduser(workdir))
        try:
            with open(job.stdout_path, "w") as stdout, open(job.stderr_path, "w") as stderr:
                yield
                with open(job.success_file(), "w"):
                    pass
            os.chdir(prevdir)
        except breakable.Break:
            with open(job.success_file(), "w"):
                pass
            os.chdir(prevdir)

    else:
        # Previous directory
        prevdir = os.getcwd()
        # Save a copy of the original stderr fd in saved_stderr_fd
        saved_stderr_fd = os.dup(original_stderr_fd)
        saved_stdout_fd = os.dup(original_stdout_fd)
        # Switch to requested directory
        os.chdir(os.path.expanduser(workdir))
        try:
            # Create a temporary file and redirect stderr to it
            with open(job.stdout_path, "w") as stdout, open(job.stderr_path, "w") as stderr:
                _redirect_stderr(stderr.fileno())
                _redirect_stdout(stdout.fileno())
                # Yield to caller, then redirect stderr back to the saved fd
                yield
                with open(job.success_file(), "w"):
                    pass
        except breakable.Break:
            with open(job.success_file(), "w"):
                pass
        except BaseException as err:
            with open(job.failed_file(), "w"):
                pass
            with open(job.stderr_path, "a") as stderr:
                stderr.write(str(err) + "\n")
        finally:
            os.chdir(prevdir)
            _redirect_stderr(saved_stderr_fd)
            _redirect_stdout(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)


class breakable(object):
    """
    Helper to allow breaking out of the contex manager early

    > with breakable(open(path)) as f:
    >     print('before condition')
    >     if condition:
    >         raise breakable.Break
    >     print('after condition')
    """

    class Break(Exception):
        """Break out of the with statement"""

    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        return self.value.__enter__()

    def __exit__(self, etype, value, traceback):
        error = self.value.__exit__(etype, value, traceback)
        if etype == self.Break:
            return True
        return error
