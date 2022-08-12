# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Callable, List, Optional, Tuple
import shutil
import tarfile
import os
import sys
import ctypes
import io
import time
import subprocess

from scine_puffin.config import Configuration

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")


def job_configuration_wrapper(run: Callable):
    """
    A wrapping function for the run method of a Job instance

    - Configures run (setting class members for specific database Calculation)
    - Additionally Try/Catch safety to avoid dying pending jobs without error
    """

    @wraps(run)
    def _impl(self, manager, calculation, config):
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
            import scine_database as db

            calculation.set_status(db.Status.FAILED)
        return success

    return _impl


class Job:
    """
    A common interface for all jobs in/carried out by a Puffin
    """

    def __init__(self):
        self.work_dir = None
        self.stdout_path = None
        self.stderr_path = None
        self.config = None
        self._id = None
        self._calculation = None
        self._manager = None
        self._calculations = None
        self._compounds = None
        self._elementary_steps = None
        self._properties = None
        self._reactions = None
        self._structures = None
        self._flasks = None

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """
        Runs the actual job.
        This function has to be implemented by any job that shall be added to
        Puffins job portfolio.

        Parameters
        ----------
        manager :: db.Manager (Scine::Database::Manager)
            The manager/database to work on/with.
        calculation :: db.Calculation (Scine::Database::Calculation)
            The calculation that triggered the execution of this job.
        config :: scine_puffin.config.Configuration
            The configuration of Puffin.
        """
        raise NotImplementedError

    @staticmethod
    def required_programs() -> List[str]:
        """
        This function has to be implemented by any job that shall be added to
        Puffins job portfolio.

        Returns
        -------
        requirements :: List[str]
            A list of names of programs/packages that are required for the
            execution of the job.
        """
        raise NotImplementedError

    def prepare(self, job_dir: str, id):
        """
        Prepares the actual job.
        This function has to be implemented by any job that shall be added to
        Puffins job portfolio.

        Parameters
        ----------
        job_dir :: str
            The path to the directory in which all jobs are executed.
        id :: db.ID (Scine::Database::ID)
            The calculation that triggered the execution of this job.
        """
        self._id = id
        self.work_dir = os.path.abspath(os.path.join(job_dir, id.string()))
        if self.work_dir and not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def archive(self, archive: str):
        """
        Archives all files existent in the job's directory into tarball named
        with the job's ID. The tarball is then moved to the given destination.

        Parameters
        ----------
        archive :: str
            The path to move the resulting tarball to.
        """
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

    def clear(self):
        """
        Clears the directory in which the job was run.
        """
        shutil.rmtree(self.work_dir)

    def verify_connection(self):
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

    def store_property(
        self,
        properties,
        property_name: str,
        property_type: str,
        data,
        model,
        calculation,
        structure,
        replace=True,
    ) -> object:
        """
        Adds a single property into the database, connecting it with a given
        structure and calculation (it's results section) and also

        Parameters
        ----------
        properties :: db.Collection (Scine::Database::Collection)
            The collection housing all properties.
        property_name :: str
            The name (key) of the new property, e.g. ``electronic_energy``.
        property_type :: str
            The type of property to be added, e.g. ``NumberProperty``.
        data :: object (According to 'property_type')
            The data to be stored in the property, the type of this object is
            dependent on the type of property requested. A ``NumberProperty``
            will require a ``float``, a ``VectorProperty`` will require a
            ``List[float]``, etc.
        model :: db.Model (Scine::Database::Model)
            The model used in the calculation that resulted in this property.
        calculation :: db.Calculation (Scine::Database::Calculation)
            The calculation that resulted in this property.
            The calculation has to be linked to its collection.
        structure :: db.Structure (Scine::Database::Structure)
            The structure for which the property is to be added. The properties
            field of the structure will receive an additional entry, or have
            an entry replaced, based on the options given to this function.
            The structure has to be linked to its collection.
        replace :: bool
            If true, replaces an existing property (identical name and model)
            with the new one. This option is true by default.
            If false, doesnothing in the previous case, and returns ``None``

        Returns
        -------
        property :: Derived of db.Property (Scine::Database::Property)
            The property, a derived class of db.Property, linked to the
            properties' collection, or ``None`` if no property was generated due
            to duplication.
        """
        existing = self.check_duplicate_property(structure, properties, property_name, model)
        if existing and replace:
            import scine_database as db

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
            import scine_database as db

            class_ = getattr(db, property_type)
            db_property = class_()
            db_property.link(properties)
            db_property.create(model, property_name, structure.id(), calculation.id(), data)
            structure.add_property(property_name, db_property.id())
            results = calculation.get_results()
            results.add_property(db_property.id())
            calculation.set_results(results)
        return db_property

    def check_duplicate_property(self, structure, properties, property_name, model) -> object:
        """
        Checks for a property that is an exact match for the one queried here.
        Exact match meaning that key and model both are matches.

        Parameters
        ----------
        properties :: db.Collection (Scine::Database::Collection)
            The collection housing all properties.
        property_name :: str
            The name (key) of the queried property, e.g. ``electronic_energy``.
        model :: db.Model (Scine::Database::Model)
            The model used in the calculation that resulted in this property.
        structure :: db.Structure (Scine::Database::Structure)
            The structure to be checked in. The structure has to be linked to
            its collection.

        Returns
        -------
        ID :: db.ID (Scine::Database::ID)
            Returns ``False`` if there is no existing property like the one
            queried or the ID of the first duplicate.
        """
        hits = structure.query_properties(property_name, model, properties)
        if len(hits) > 0:
            return hits[0]
        return False

    def configure_run(self, manager, calculation, config: Configuration):
        """
        Configures a job for a given Calculation to do tasks in the run function

        Parameters
        ----------
        manager :: db.Manager (Scine::Database::Manager)
            The manager of the database holding all collections
        calculation :: db.Calculation (Scine::Database::Calculation)
            The calculation to be performed
        config :: Configuration
            The configuration of the Puffin doing the job
        """
        self.get_collections(manager)
        self.set_calculation(calculation)
        self.config = config

    def get_collections(self, manager):
        """
        Saves Scine Database collections as class variables

        Parameters
        ----------
        manager :: db.Manager (Scine::Database::Manager)
            The manager of the database holding all collections
        """
        self._manager = manager
        self._calculations = manager.get_collection("calculations")
        self._compounds = manager.get_collection("compounds")
        self._elementary_steps = manager.get_collection("elementary_steps")
        self._properties = manager.get_collection("properties")
        self._reactions = manager.get_collection("reactions")
        self._structures = manager.get_collection("structures")
        self._flasks = manager.get_collection("flasks")

    def set_calculation(self, calculation):
        """
        Sets the current Calculation for this job and ensures connection

        Parameters
        ----------
        calculation :: db.Calculation (Scine::Database::Calculation)
            The calculation to be carried out
        """
        self._calculation = calculation
        if not self._calculation.has_link():
            self._calculation.link(self._calculations)

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

    def complete_job(self) -> None:
        """
        Saves the executing Puffin, changes status to db.Status.COMPLETE.
        """
        import scine_database as db

        self.capture_raw_output()
        self._calculation.set_executor(self.config["daemon"]["uuid"])
        self._calculation.set_status(db.Status.COMPLETE)

    def fail_job(self) -> None:
        """
        Saves the executing Puffin, changes status to db.Status.FAILED.
        """
        import scine_database as db

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


class TurbomoleJob(Job):
    """
    A common interface for all jobs in Puffin that use Turbomole.
    """

    def __init__(self):
        super().__init__()
        self.input_structure = "system.xyz"

        env = os.environ.copy()

        self.turboexe = ""
        self.turboscripts = ""
        self.smp_turboexe = ""

        if "TURBODIR" in env.keys():
            if env["TURBODIR"]:
                if os.environ.get("PARA_ARCH") is not None:
                    del os.environ["PARA_ARCH"]
                if os.path.exists(os.path.join(env["TURBODIR"], "scripts", "sysname")):
                    self.sysname = (
                        subprocess.check_output(os.path.join(env["TURBODIR"], "scripts", "sysname"))
                        .decode("utf-8", errors='replace')
                        .rstrip()
                    )
                    self.sysname_parallel = self.sysname + "_smp"
                    self.turboexe = os.path.join(env["TURBODIR"], "bin", self.sysname)
                    self.smp_turboexe = os.path.join(env["TURBODIR"], "bin", self.sysname_parallel)
                    self.turboscripts = os.path.join(env["TURBODIR"], "scripts")
                else:
                    raise RuntimeError("TURBODIR not assigned correctly. Check spelling or empty the env variable.")

    def prepare_calculation(self, structure, calculation_settings, model, job):

        import scine_utilities as utils
        from scine_puffin.utilities.turbomole_helper import TurbomoleHelper

        tm_helper = TurbomoleHelper()

        # Write xyz file
        utils.io.write(self.input_structure, structure.get_atoms())
        # Write coord file
        tm_helper.write_coord_file(calculation_settings)
        # Check if settings are available
        tm_helper.check_settings_availability(job, calculation_settings)
        # Generate input file for preprocessing tool 'define'
        tm_helper.prepare_define_session(structure, model, calculation_settings, job)
        # Initialize via define
        tm_helper.initialize(model, calculation_settings)

    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs() -> List[str]:
        """See Job.required_programs()"""
        raise NotImplementedError


@contextmanager
def calculation_context(job: Job, stdout_name="output", stderr_name="errors", debug: Optional[bool] = None):
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
    job :: Job
        The job holding the working directory and receiving the output and error paths
    stdout_name :: str
        Name of the file that the stdout stream should be redirected to.
        The file will be generated in the given scratch directory.
    stderr_name :: str
        Name of the file that the stderr stream should be redirected to.
        The file will be generated in the given scratch directory.
    debug :: bool
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
        with open(job.stdout_path, "w") as stdout, open(job.stderr_path, "w") as stderr:
            yield
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
    >     print 'before condition'
    >     if condition:
    >         raise breakable.Break
    >     print 'after condition'
    """

    class Break(Exception):
        """Break out of the with statement"""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value.__enter__()

    def __exit__(self, etype, value, traceback):
        error = self.value.__exit__(etype, value, traceback)
        if etype == self.Break:
            return True
        return error
