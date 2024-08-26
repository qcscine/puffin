# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
import os
import sys
import signal
from daemon import pidfile, DaemonContext
import setproctitle
from .config import Configuration
from .jobloop import check_setup, loop, slow_connect


def shutdown(_signum, _frame):
    """
    A small helper function triggering the stop of the process.

    Parameters
    ----------
    _signum : int
        Dummy variable to match the signal dependent function signature.
    _frame
        Dummy variable to match the signal dependent function signature.
    """
    sys.exit(0)


def check_environment(config: Configuration):
    """
    Checks the runtime environment for problematic configurations that may
    interfere with job executions down the line.

    Parameters
    ----------.
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    if "OMP_NUM_THREADS" in os.environ:
        if os.environ["OMP_NUM_THREADS"] != str(config["resources"]["cores"]):
            raise RuntimeError("Environment variable OMP_NUM_THREADS must "
                               "match configured number of cores.")
    else:
        os.environ["OMP_NUM_THREADS"] = str(config["resources"]["cores"])


def stop_daemon(config: Configuration):
    """
    Stops the Puffin gracefully, allowing th current job to finish, then shutting
    down.

    Parameters
    ----------.
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    # Generate stop file in order to stop after the current job
    pid_file = config["daemon"]["pid"]
    if os.path.isfile(pid_file):
        stop_file = config.daemon()["stop"]
        basedir = os.path.dirname(stop_file)
        if not os.path.exists(basedir):
            try:
                os.makedirs(basedir)
            except FileExistsError:
                pass
        with open(stop_file, "w"):
            pass


def start_daemon(config: Configuration, detach: bool = True):
    """
    Starts the Puffin, using the given configuration.

    Parameters
    ----------.
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    detach : bool
        If true, forks the daemon process and detaches it.
    """
    check_environment(config)

    # Ensure existence of the directory for job files
    job_dir = config["daemon"]["job_dir"]
    if job_dir and not os.path.exists(job_dir):
        try:
            os.makedirs(job_dir)
        except FileExistsError:
            pass

    # Ensure existence of the directory for a pid file
    pid = config["daemon"]["pid"]
    pid_dir = os.path.split(config["daemon"]["pid"])[0]
    if pid_dir and not os.path.exists(pid_dir):
        try:
            os.makedirs(pid_dir)
        except FileExistsError:
            pass

    # Ensure existence of the directory for a stop file
    stop_dir = os.path.split(config["daemon"]["stop"])[0]
    if stop_dir and not os.path.exists(stop_dir):
        try:
            os.makedirs(stop_dir)
        except FileExistsError:
            pass

    # Generate log file if not present
    if config["daemon"]["log"]:
        log_dir = os.path.split(config["daemon"]["log"])[0]
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except FileExistsError:
                pass
        if not os.path.exists(config["daemon"]["log"]):
            with open(config["daemon"]["log"], "w"):
                pass

    # Give the daemon a decent name
    setproctitle.setproctitle("puffin")

    # Check the setup of all programs
    available_jobs = check_setup(config)

    # Check if the database is reachable
    import scine_database as db

    manager = db.Manager()
    slow_connect(manager, config)

    context = DaemonContext(
        chroot_directory=None,
        working_directory=job_dir,
        stdout=sys.stdout,
        stderr=sys.stderr,
        pidfile=pidfile.TimeoutPIDLockFile(pid),
        detach_process=detach,
    )

    def exit_gracefully(*args, **kwargs):
        print("Puffin shutting down gracefully")
        stop_daemon(config)

    context.signal_map = {
        signal.SIGINT: exit_gracefully,
        signal.SIGTERM: exit_gracefully,
        signal.SIGTSTP: exit_gracefully
    }

    if config["daemon"]["mode"] == "debug":
        loop(config, available_jobs)
    else:
        with context:
            loop(config, available_jobs)
