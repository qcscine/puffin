# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
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
    _signum :: int
        Dummy variable to match the signal dependent function signature.
    _frame
        Dummy variable to match the signal dependent function signature.
    """
    sys.exit(0)


def stop_daemon(config: Configuration):
    """
    Stops the Puffin gracefully, allowing th current job to finish, then shutting
    down.

    Parameters
    ----------.
    config :: scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    # Generate stop file in order to stop after the current job
    pid_file = config["daemon"]["pid"]
    if os.path.isfile(pid_file):
        stop_file = config.daemon()["stop"]
        basedir = os.path.dirname(stop_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        with open(stop_file, "w"):
            pass


def start_daemon(config: Configuration):
    """
    Starts the Puffin, using the given configuration.

    Parameters
    ----------.
    config :: scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    if "OMP_NUM_THREADS" in os.environ:
        if os.environ["OMP_NUM_THREADS"] != str(config["resources"]["cores"]):
            raise RuntimeError("Environment variable OMP_NUM_THREADS must "
                               "match configured number of cores.")
    else:
        os.environ["OMP_NUM_THREADS"] = str(config["resources"]["cores"])

    # Ensure existence of the directory for job files
    job_dir = config["daemon"]["job_dir"]
    if job_dir and not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Ensure existence of the directory for a pid file
    pid = config["daemon"]["pid"]
    pid_dir = os.path.split(config["daemon"]["pid"])[0]
    if pid_dir and not os.path.exists(pid_dir):
        os.makedirs(pid_dir)

    # Ensure existence of the directory for a stop file
    stop_dir = os.path.split(config["daemon"]["stop"])[0]
    if stop_dir and not os.path.exists(stop_dir):
        os.makedirs(stop_dir)

    # Generate log file if not present
    if config["daemon"]["log"]:
        log_dir = os.path.split(config["daemon"]["log"])[0]
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
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
        detach_process=True,
    )
    context.signal_map = {signal.SIGTERM: shutdown, signal.SIGTSTP: shutdown}

    if config["daemon"]["mode"] == "debug":
        loop(config, available_jobs)
    else:
        with context:
            loop(config, available_jobs)
