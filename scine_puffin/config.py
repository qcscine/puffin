# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import operator
import os
import uuid
import yaml
from functools import reduce
from copy import deepcopy
from typing import Any, Dict, List, Optional


def dict_generator(indict: Dict[str, Any], pre: Optional[List[str]] = None):
    """
    A small helper function/generator recursively generating all chains of keys
    for a given dictionary.

    Parameters
    ----------
    indict :: dict
        The dictionary to traverse.
    pre :: dict
        The parent dictionary (used for recursion).

    Yields
    ------
    key_chain :: List[str]
        A list of keys from top level to bottom level for each end in the tree
        of possible value fields in the given dictionary.
    """
    pre = deepcopy(pre) if pre is not None else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key]
    else:
        yield indict


class Configuration:
    """
    The Puffin configuration. All values are defaulted. The main sections of the
    configuration are:

    **daemon**
      The settings peratining the execution of Puffin and its daemon process.

    **database**
      All information about the database the Puffin will be working on.

    **resources**
      The information about the hardware the Puffin is running on and is
      allowed to use for calculations and the execution of jobs.

    **programs**
      The settings for all the programs and packages Puffin relies on when
      executing jobs. Each program/packages has its own entry with the
      possibility of program specific settings. See the documentation for each
      individual program (found at ``scine_puffin.programs``) for more details about
      the individual settings.

    Note that the config is sensitive to environment variables when it is
    initialized/loaded. Each setting in the config can be set via a
    corresponding environment variable. The settings are given as
    ``PUFFIN_<key1>_<key2>=<value>`` where the keys are the chain of uppercase
    keys to the final setting. As an example: ``PUFFIN_DAEMON_MODE=debug`` would
    equal ``config['daemon']['mode'] = 'debug'``.

    In detail, the options in the configuration are:

    **daemon**
      mode :: str
        The mode to run the Puffin in, options are ``release`` and ``debug``.
        The ``release`` mode will fork the main process and run in a daemonized
        mode while the ``debug`` mode will run in the current shell, reporting
        any output and errors to ``stdout`` and ``stderr``.
      job_dir :: str
        The path to the directory containing the currently running job.
      software_dir :: str
        The path to the directory containing the software bootstrapped from
        sources. The Puffin will generate and fill this directory upon bootstrapping.
      error_dir :: str
        If existent, the Puffin instance will archive all failed jobs into this
        directory.
      archive_dir :: str
        If existent, the Puffin instance will archive all correctly completed
        jobs into this directory.
      uuid :: str
        A unique name for the Puffin instance. This can be set by the user, if
        not, a unique ID will automatically be generated.
      pid :: str
        The path to the file identifying the PID of the Puffin instance.
        Automatically populated on start-up if left empty.
      pid_dir :: str
        The path to a folder holding the file identifying the PID of the Puffin instance.
      log :: str
        The path to the logfile of the Puffin instance.
      stop :: str
        The path to a file that if existent will prompt the Puffin instance to
        stop taking new jobs and shut down instead. The instance will finish any
        running job though.
      remove_stop_file :: bool
        Upon finding a stop file the daemon will stop, if this option is set to
        ``True`` the found file will be deleted allowing instant restarts.
        In cases where multiple puffins depend on the same stop file it may be
        required to keep the stop file, setting this option to ``False``
      cycle_time_in_s :: float
        The time in between scans of the database for new jobs that can be run.
      timeout_in_h :: float
        The number of hours the Puffin instance should stay alive. Once this
        limit is reached, the Puffin is shut down and its running job will be
        killed and re-flagged as `new`.
      idle_timeout_in_h :: float
        The number of hours the Puffin instance should stay alive. After
        receiving the last job, once the limit is reached, the Puffin is shut
        down. Any accepted job will reset the timer. A negative value disables
        this feature and make the Puffin run until the ``timeout_in_h`` is
        reached independent of it being idle the entire time.
      touch_time_in_s :: float
        The time in seconds in between the attempts of the puffin to touch a
        calculation it is running in the database.
        In practice each Puffin will search for jobs in the database that are
        set as running but are not touched and reset them, as they indicate that
        the executing puffin has crashed. See ``job_reset_time_in_s`` for more
        information.
      job_reset_time_in_s :: float
        The time in seconds that may have passed since the last touch on pending
        jobs before they are considered dead and are reset to be worked by
        another puffin.
        Note: The time in this setting has to be larger than the
        ``touch_time_in_s`` of all Puffins working on the same database to work!
      repeated_failure_stop :: int
        The number of consecutive failed jobs that are allowed before the Puffin
        stops in order to avoid failing all jobs in a DB due to e.g. hardware
        issues. Failed jobs will be reset to new and rerun by other Puffins.
        Should always be greater than 1.
      max_number_of_jobs :: int
        The maximum number of jobs a single Puffin will carry out (complete
        or failed), before gracefully exiting. Any negative number or zero disables this
        setting; by default it is disabled.
      enforce_memory_limit :: bool
        If the given memory limit should be enforced (i.e., a job is killed as soon as it reaches it)
        or not. The puffin still continues to work on other calculations either way.

    **database**
      ip :: str
        The IP at which the database server to connect with is found.
      port :: int
        The port at which the database server to connect with is found.
      name :: str
        The name of the database on the database server to connect with.
        Multiple databases (with multiple names) can be given as comma seperated
        list: ``name_one,name_two,name_three``. The databases will be used in
        descending order of priority. Meaning: at any given time all jobs of the
        first named database will have to be completed before any job of the
        second one will be considered by the Puffin instance.

    **resources**
      cores :: int
        The number of threads the executed jobs are allowed to use. Note that
        only jobs that are below this value in their requirements will be
        accepted by the Puffin instance.
      memory :: float
        The total amount of memory the Puffin and its jobs are allowed to use.
        Given in GB. Note that only jobs that are below this value in their
        requirements will be accepted by the Puffin instance.
      disk :: float
        The total amount of disk space the Puffin and its jobs are allowed to
        use. Given in GB. Note that only jobs that are below this value in their
        requirements will be accepted by the Puffin instance.
      ssh_keys :: List[str]
        Any SSH keys needed by the Puffin in order to connect to the database
        or to bootstrap programs.

    **programs**
      The specific details for each program are given in their respective
      documentation. However, common options are:

      available :: bool
        The switch whether the program shall be available to Puffin.
        Any programs set to be unavailable will not be bootstrapped.
      source :: str
        The link to the source location of the given program, usually a https
        link to a git repository
      root :: str
        The folder at which the program is already installed at.
        This will request a non source based bootstrapping of the program.
      version :: str
        The version of the program to use. Can also be a git tag or commit SHA.

    The default version of a configuration file can be generated using
    ``python3 -m puffin configure`` (if no environment variables are set).
    """

    def __init__(self):
        self._data = {}
        self._data["database"] = {"ip": "127.0.0.1", "port": 27017, "name": "default"}
        self._data["resources"] = {
            "cores": 1,
            "memory": 1.0,
            "disk": 5.0,
            "ssh_keys": [],
        }
        self._data["daemon"] = {
            "mode": "release",
            "job_dir": "/scratch/puffin/jobs",
            "software_dir": "/scratch/puffin/software",
            "error_dir": "",
            "archive_dir": "",
            "uuid": "",
            "pid": "",
            "pid_dir": "/scratch/puffin/",
            "log": "/scratch/puffin/puffin.log",
            "stop": "/scratch/puffin/puffin.stop",
            "remove_stop_file": True,
            "cycle_time_in_s": 10.0,
            "timeout_in_h": 48.0,
            "touch_time_in_s": 1500.0,
            "job_reset_time_in_s": 7200.0,
            "idle_timeout_in_h": -1.0,
            "max_number_of_jobs": -1,
            "repeated_failure_stop": 100,
            "enforce_memory_limit": True,
        }
        self._data["programs"] = {
            "readuct": {
                "available": True,
                "source": "https://github.com/qcscine/readuct.git",
                "root": "",
                "version": "5.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "utils": {
                "available": True,
                "source": "https://github.com/qcscine/utilities.git",
                "root": "",
                "version": "8.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "database": {
                "available": True,
                "source": "https://github.com/qcscine/database.git",
                "root": "",
                "version": "1.2.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "sparrow": {
                "available": True,
                "source": "https://github.com/qcscine/sparrow.git",
                "root": "",
                "version": "4.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "molassembler": {
                "available": True,
                "source": "https://github.com/qcscine/molassembler.git",
                "root": "",
                "version": "2.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "swoose": {
                'available': False,
                'source': 'https://github.com/qcscine/swoose.git',
                'root': '',
                'version': 'master',
                'march': 'native',
                "cmake_flags": "",
                "cxx_compiler_flags": "",
            },
            "turbomole": {
                'available': False,
                'source': '',
                'root': '',
                'version': '7.2.0',
            },
            "orca": {
                "available": False,
                "source": "",
                "root": "",
                "version": "4.1",
            },
            "cp2k": {
                "available": False,
                "source": "",
                "root": "",
                "version": "",
            },
            "serenity": {
                "available": False,
                "source": "https://github.com/qcscine/serenity_wrapper.git",
                "root": "",
                "version": "2.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "gaussian": {
                "available": False,
                "source": "",
                "root": "",
                "version": "g09 Rev. D.01",
            },
            "xtb": {
                "available": False,
                "source": "https://github.com/qcscine/xtb_wrapper.git",
                "root": "",
                "version": "2.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
            "kinetx": {
                "available": False,
                "source": "https://github.com/qcscine/kinetx.git",
                "root": "",
                "version": "2.0.0",
                "march": "native",
                "cxx_compiler_flags": "",
                "cmake_flags": "",
            },
        }

    def __getitem__(self, key: str) -> dict:
        """
        The configuration can be used similarly to a regular dictionary.

        Parameters
        ----------
        key :: str
            One of: ``daemon``, ``database``, ``resources``, ``programs``.

        Returns
        -------
        settings :: dict
            A sub-dict of the total configuration.
        """
        return self._data[key]

    def database(self) -> dict:
        """
        Grants direct access to the ``database`` part of the configuration.

        Returns
        -------
        settings :: dict
            A sub-dict of the total configuration.
        """
        return self._data["database"]

    def resources(self) -> dict:
        """
        Grants direct access to the ``resources`` part of the configuration.

        Returns
        -------
        settings :: dict
            A sub-dict of the total configuration.
        """
        return self._data["resources"]

    def daemon(self) -> dict:
        """
        Grants direct access to the ``daemon`` part of the configuration.

        Returns
        -------
        settings :: dict
            A sub-dict of the total configuration.
        """
        return self._data["daemon"]

    def programs(self) -> dict:
        """
        Grants direct access to the ``programs`` part of the configuration.

        Returns
        -------
        settings :: dict
            A sub-dict of the total configuration.
        """
        return self._data["programs"]

    def dump(self, path: str):
        """
        Dumps the current configuration into a .yaml file.

        Parameters
        ----------
        path :: str
            The file to dump the configuration into.
        """
        # Parse environment
        self.load()
        yaml_dir = os.path.split(path)[0]
        if yaml_dir:
            if yaml_dir and not os.path.exists(yaml_dir):
                os.makedirs(yaml_dir)
        with open(path, "w") as outfile:
            yaml.dump(self._data, outfile, default_flow_style=False)

    def load(self, path: Optional[str] = None):
        """
        Loads the configuration. The configuration is initialized using the
        default values, then all settings given in the file (if there is one)
        are applied. Finally all settings given as environment variables are
        applied.

        Each setting in the config can be set via a corresponding environment
        variable. The settings are given as ``PUFFIN_<key1>_<key2>=<value>``
        where the keys are the chain of uppercase keys to the final setting.
        As an example: ``PUFFIN_DAEMON_MODE=debug`` would equal
        ``config['daemon']['mode'] = 'debug'``.

        The exact load order is (with the latter one overriding the former):
          1. defaults
          2. file path
          3. environment variables

        Parameters
        ----------
        path :: str
            The file to read the configuration from. Default: ``None``
        """
        # Parse file
        if path is not None:
            with open(path, "r") as infile:
                new_data = yaml.safe_load(infile)
            if "database" in new_data:
                self._apply_changes(self._data["database"], new_data["database"])
            if "resources" in new_data:
                self._apply_changes(self._data["resources"], new_data["resources"])
            if "programs" in new_data:
                for key in self._data["programs"]:
                    if key in new_data["programs"]:
                        self._apply_changes(self._data["programs"][key], new_data["programs"][key])
            if "daemon" in new_data:
                self._apply_changes(self._data["daemon"], new_data["daemon"])

        # Parse environment
        env = os.environ.copy()
        for key_chain in dict_generator(self._data):
            key = ("PUFFIN_" + "_".join(key_chain)).upper()
            if key in env:
                try:
                    current_value = reduce(operator.getitem, key_chain, self._data)
                except BaseException as e:
                    raise KeyError("The environment variable '{}' does not translate to a valid option.".format(key)) \
                        from e
                try:
                    if isinstance(current_value, bool):
                        value = env[key].lower() in ["true", "1"]
                    else:
                        value = type(current_value)(env[key])
                except BaseException as e:
                    raise KeyError(
                        "The environment variable '{}' can not be translated "
                        "into the correct variable type.".format(key)
                    ) from e
                reduce(operator.getitem, key_chain[:-1], self._data)[key_chain[-1]] = value

        # Generate uuid of specified type if unset
        if not self._data["daemon"]["uuid"] or self._data["daemon"]["uuid"] == "uuid1":
            self._data["daemon"]["uuid"] = "puffin-" + uuid.uuid1().hex
        elif self._data["daemon"]["uuid"] == "uuid4":
            self._data["daemon"]["uuid"] = "puffin-" + uuid.uuid4().hex

        # Generate pid file path
        if not self._data["daemon"]["pid"]:
            self._data["daemon"]["pid"] = os.path.join(
                self._data["daemon"]["pid_dir"],
                f'{self._data["daemon"]["uuid"]}.pid'
            )

    def _apply_changes(self, to_dict: dict, from_dict: dict):
        """
        A small helper applying changes from one dictionary to another, checking
        the types and making sure only existing keys are mapped.

        Parameters
        ----------
        to_dict :: dict
            The dictionary to apply the changes to.
        from_dict :: dict
            The dictionary to read the changes from.

        Raises
        ------
        TypeError
            If a set of values of the same key do not have matching types.
        KeyError
            If a key in the ``fromDict`` does not exist in the ``toDict``.
        """
        for key in to_dict:
            if key in from_dict:
                # type check
                if isinstance(from_dict[key], type(to_dict[key])):
                    to_dict[key] = from_dict[key]
                else:
                    raise TypeError(
                        "Expected " + str(type(to_dict[key])) + " type variable for "
                        "as value for the '" + key + "' field."
                    )
            else:
                raise KeyError("Unknown key: " + key + ".")
