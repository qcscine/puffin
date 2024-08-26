# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import psutil
import sys
import time
import ctypes
import multiprocessing
import random
import traceback
from datetime import datetime, timedelta
from importlib import import_module, util
from typing import Any, Dict, List
from json import dumps
from .config import Configuration

# A global variable holding the process actually running jobs.
# This variable is used to be able to interact with (mainly to kill) said
# process even from outside the loop() function.
PROCESS = None


def _log(config: Configuration, message: str):
    """
    Logs given message together with time stamp and line break if the running puffin has a configured log.

    Parameters
    ----------
    config : Configuration
        The configuration of the puffin
    message : str
        The message that is padded
    """
    if config["daemon"]["log"]:
        with open(config["daemon"]["log"], "a") as f:
            f.write(str(datetime.utcnow()) + ": " + config["daemon"]["uuid"] + ": " + message + "\n")


def slow_connect(manager, config: Configuration) -> None:
    """
    Connects the given Manager to the database referenced in the Configuration.
    This version of connecting tries 30 times to connect to the database.
    Each attempt is followed by a wait time of `1.0 + random([0.0, 1.0])` seconds in
    order to stagger connection attempts of multiple Puffin instances.

    Parameters
    ----------
    manager : scine_database.Manager
        The database manager/connection.
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    import scine_database as db

    name = config["database"]["name"]
    if "," in name:
        name = name.split(",")[0]
    credentials = db.Credentials(config["database"]["ip"], config["database"]["port"], name)
    _log(config, "Connecting to: {:s}:{:d} {:s}".format(config["database"]["ip"],
                                                        config["database"]["port"],
                                                        name))
    manager.set_credentials(credentials)
    # Allow 30 to 60 seconds to try and connect to the database
    for _ in range(30):
        try:
            manager.connect()
            break
        except BaseException as e:
            _log(config, "Connection failed with " + str(e) + ". Keep trying to connect.")
            r = random.uniform(0, 1)
            time.sleep(1.0 + r)
    else:
        manager.connect()


def kill_daemon(config: Configuration) -> None:
    """
    Kills the Puffin instantaneously without any possibility of a graceful exit.

    Parameters
    ----------
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    # Remove stop file if present
    if config.daemon()["remove_stop_file"]:
        stop_file = config.daemon()["stop"]
        if os.path.isfile(stop_file):
            try:
                os.remove(stop_file)
            except FileNotFoundError:
                pass

    # Kill the daemon process
    pid_file = config["daemon"]["pid"]

    if PROCESS is not None:
        parent = psutil.Process(PROCESS.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    if os.path.isfile(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.readline().strip())
        os.remove(pid_file)
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()


def loop(config: Configuration, available_jobs: dict) -> None:
    """
    The outer loop function.
    This function controls the forked actual loop function, which is implemented
    in _loop_impl(). The loop has an added timeout and also a 15 min ping is
    added showing that the runner is still alive.

    Parameters
    ----------
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    available_jobs : dict
        The dictionary of available jobs, given the current config
        and runtime environment.
    """

    if config["daemon"]["mode"].lower() == "debug":
        _loop_impl(config=config, available_jobs=available_jobs)
        sys.exit()

    # Connect to database
    import scine_database as db

    manager = db.Manager()
    slow_connect(manager, config)

    # Generate shared variable
    # Shared variables have to be ctypes so this is a bit ugly
    JOB: Any = multiprocessing.Array(ctypes.c_char, 200)
    JOB.value = "".encode("utf-8")
    CURRENT_DB: Any = multiprocessing.Array(ctypes.c_char, 200)
    CURRENT_DB.value = manager.get_database_name().encode("utf-8")

    # Run the loop in a second process
    PROCESS = multiprocessing.Process(  # pylint: disable=redefined-outer-name
        target=_loop_impl,
        args=(),
        kwargs={
            "config": config,
            "available_jobs": available_jobs,
            "JOB": JOB,
            "CURRENT_DB": CURRENT_DB,
        },
    )
    PROCESS.start()

    # Check for timeout
    timeout_setting = config["daemon"]["timeout_in_h"]
    timeout = timedelta(hours=timeout_setting) if timeout_setting > 0 else None
    idle_setting = config["daemon"]["idle_timeout_in_h"]
    idle_timeout = timedelta(hours=idle_setting) if idle_setting > 0 else None
    touch_time = timedelta(seconds=config["daemon"]["touch_time_in_s"])
    reset_delta = timedelta(seconds=config["daemon"]["job_reset_time_in_s"])
    last_time_with_a_job = datetime.now()
    last_touch = datetime.now()
    start = datetime.now()
    while PROCESS.is_alive():
        time.sleep(1.0)
        # Kill the puffin if it was idle for too long
        now = datetime.now()
        if JOB.value.decode("utf-8"):
            last_time_with_a_job = now
        if idle_timeout is not None and (now - last_time_with_a_job) > idle_timeout:
            _log(config, "Puffin reached idle timeout")
            kill_daemon(config)
        # Kill job if it is out of time
        if timeout is not None and (now - start) > timeout:
            # But first reset the calculation
            if JOB.value.decode("utf-8"):
                manager.set_database_name(CURRENT_DB.value.decode("utf-8"))
                calculations = manager.get_collection("calculations")
                job = db.Calculation(db.ID(JOB.value.decode("utf-8")), calculations)
                job.set_status(db.Status.NEW)
                job.set_executor("")
            _log(config, "Puffin reached timeout")
            kill_daemon(config)
        # Touch current calculation every so often
        if (now - last_touch) > touch_time:
            # Touch the own calculation
            last_touch = now
            try:
                manager.set_database_name(CURRENT_DB.value.decode("utf-8"))
                calculations = manager.get_collection("calculations")
                if JOB.value.decode("utf-8"):
                    job = db.Calculation(db.ID(JOB.value.decode("utf-8")), calculations)
                    job.touch()
                    _log(config, "Touching Job: {:s}".format(str(job.id())))
                # TODO maybe move this onto a separate timer/if to reduce DB queries
                _check_touch_of_pending_jobs(manager, calculations, config, reset_delta)
            except BaseException as e:
                # If it isn't possible to work with the database, kill the
                # job/loop and stop.
                _log(config, "Failed to work with database, received error {:s}\n".format(str(e)))
                kill_daemon(config)


def _check_touch_of_pending_jobs(
        manager, calculations, config: Configuration, reset_delta: timedelta
) -> None:
    """
    Checks for calculation of other Puffins that are pending.
    If these jobs have not been touched in due time they are
    reset, as their Puffins are expected to be dead.

    Parameters
    ----------
    manager : scine_database.Manager
        The database connection
    calculations : scine_database.Collection
        The collection holding all calculations
    reset_delta : int
        The time difference after which a job is assumed
        to be dead. Time given in seconds.
    """
    import scine_database as db

    # Check for dead jobs in pending status in the database
    selection = {"status": "pending"}
    server_now = manager.server_time()
    for pending_calculation in calculations.query_calculations(dumps(selection)):
        last_modified = pending_calculation.last_modified()
        if (server_now - last_modified) > reset_delta:
            _log(config, "Resetting Job: {:s}".format(str(pending_calculation.id())))
            pending_calculation.set_status(db.Status.NEW)
            pending_calculation.set_executor("")


def check_setup(config: Configuration) -> Dict[str, str]:
    """
    Checks if all the programs are correctly installed or reachable.

    Parameters
    ----------
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    """
    scine_database = util.find_spec("scine_database")
    if scine_database is None:
        print("Missing SCINE Database backend, please bootstrap Puffin.")
        sys.exit(1)
    scine_database = util.find_spec("scine_utilities")
    if scine_database is None:
        print("Missing SCINE Utilities, please bootstrap Puffin.")
        sys.exit(1)
    scine_readuct = util.find_spec("scine_readuct")
    if scine_readuct is None:
        print("SCINE ReaDuct is not available for Puffin. Note that this will disable nearly all exploration jobs.")

    # Generate the list of available programs
    available_programs = []
    for program_name, settings in config.programs().items():
        if settings["available"]:
            available_programs.append(program_name)

    if scine_readuct is None and "readuct" in available_programs:
        raise RuntimeError("SCINE ReaDuct was not found by Puffin but is set as available in the run configuration.\n"
                           "Please make sure that SCINE ReaDuct is installed properly, bootstrap Puffin, or disable\n"
                           "SCINE ReaDuct in the run configuration.")

    # Initialize all available programs
    for program_name in available_programs:
        class_name = "".join([s.capitalize() for s in program_name.split("_")])
        module = import_module("scine_puffin.programs." + program_name)
        class_ = getattr(module, class_name)
        class_.initialize()

    # Gather list of all jobs
    all_jobs = []
    import scine_puffin.jobs

    for path in scine_puffin.jobs.__path__:
        for _, dirs, files in os.walk(path):
            for name in files:
                if name.endswith(".py") and name != "__init__.py" and "templates" in dirs:
                    all_jobs.append(name[:-3])

    # Generate list of jobs for which the required programs are present
    available_jobs = {}
    for job in all_jobs:
        class_name = "".join([s.capitalize() for s in job.split("_")])
        module = import_module("scine_puffin.jobs." + job)
        class_ = getattr(module, class_name)
        required_programs = class_.required_programs()
        for program in required_programs:
            if program not in available_programs:
                break
        else:
            available_jobs[job] = class_name

    # Output results
    print("")
    print("Available Resources:")
    print("   {:18} {:6d}.000".format("Threads:", config["resources"]["cores"]))
    print("   {:18} {:10.3f} GB".format("Total RAM:", config["resources"]["memory"]))
    print("   {:18} {:10.3f} GB".format("RAM/Thread:", config["resources"]["memory"] / config["resources"]["cores"]))
    print("   {:18} {:10.3f} GB".format("Total Disk Space:", config["resources"]["disk"]))
    print(
        "   {:18} {:10.3f} GB".format(
            "Disk Space/Thread:",
            config["resources"]["disk"] / config["resources"]["cores"],
        )
    )
    print("")
    print("Available Programs:")
    for available_program in available_programs:
        print("    - " + available_program)
    print("")
    print("Accepting Jobs:")
    for job_name, class_name in available_jobs.items():
        print("    - " + job_name)
    print("")
    return available_jobs


def _loop_impl(
        config: Configuration, available_jobs: dict, JOB=None, CURRENT_DB=None
) -> None:
    """
    The actual loop, executing jobs and handling all calculation related
    operations.

    Parameters
    ----------
    config : scine_puffin.config.Configuration
        The current configuration of the Puffin.
    available_jobs : dict
        A dictionary of all jobs that are available to this Puffin.
    JOB : multiprocessing.Array
        Possibly a shared array of chars (string) to share the current jobs ID
        with external code. Default: ``None``
    CURRENT_DB : multiprocessing.Array
        The name of the current database, used to sync the two threads in case
        of multi-database usage of a single Puffin.
    """
    import scine_database as db

    # Connect to database
    manager = db.Manager()
    slow_connect(manager, config)

    # Initialize loop variables
    sleep = timedelta(seconds=config["daemon"]["cycle_time_in_s"])
    last_cycle = datetime.now()
    job_list = list(available_jobs.keys())
    program_list = ["any"]
    version_list = []
    for program_name, settings in config.programs().items():
        if settings["available"]:
            program_list.append(program_name)
            version_list.append(program_name + settings["version"])

    # Initialize cache for failure checks
    previously_failed_job_count = 0
    previously_failed_jobs: List[db.ID] = []
    previous_dbs: List[str] = []
    n_jobs_run = 0

    while True:
        # Stop the loop if a stop file has been written
        stop_file = config["daemon"]["stop"]
        if os.path.isfile(stop_file):
            if config.daemon()["remove_stop_file"]:
                try:
                    os.remove(stop_file)
                except FileNotFoundError:
                    pass

            _log(config, "Detected stop file " + stop_file + " and stopped puffin.")
            break

        # Wait if needed
        loop_time = datetime.now() - last_cycle
        if loop_time < sleep:
            time.sleep(int(round((sleep - loop_time).total_seconds())))
        last_cycle = datetime.now()

        # Verify that the database is still listening
        while not manager.is_connected():
            # Retry connection until killed
            time.sleep(10)
            try:
                manager.disconnect()
                manager.connect()
            except BaseException as e:
                _log(config, "Failed to connect to database with error " + str(e) + ". Keep trying to connect.")

        # ===================
        #   Job procurement
        # ===================

        # use ',' as db separator
        db_names = config["database"]["name"].split(",")
        for db_name in db_names:
            # Switch to requested DB
            manager.set_database_name(db_name)
            if CURRENT_DB is not None:
                CURRENT_DB.value = db_name.encode("utf-8")
            # Get calculations collection from current DB
            if not manager.has_collection("calculations"):
                # This DB is not initialized, skip to next one
                continue
            collection = manager.get_collection("calculations")
            selection = {
                "$and": [
                    {"status": "new"},
                    {"job.cores": {"$lte": int(config["resources"]["cores"])}},
                    {"job.disk": {"$lte": float(config["resources"]["disk"])}},
                    {"job.memory": {"$lte": float(config["resources"]["memory"])}},
                    {"job.order": {"$in": job_list}},
                    {"model.program": {"$in": program_list}}
                    # { '$or' : [
                    #    {'model.version' : { '$eq' : 'any'} },
                    #    {'model.program + model.version' : { '$in' : version_list} }
                    #    ]}
                ]
            }
            this_puffin_id = config["daemon"]["uuid"]
            update = {"$set": {"status": "pending", "executor": this_puffin_id}}
            # sort for first priority and then most expensive calculation with cores, then memory, then disk
            sort = {'priority': 1, 'job.cores': -1, 'job.memory': -1, 'job.disk': -1}
            calculation = collection.get_and_update_one_calculation(dumps(selection), dumps(update), dumps(sort))
            calculation.link(collection)

            if calculation.has_id():  # we found a calculation
                time_waited = 0.0
                executor = calculation.get_executor()
                while executor.strip() == "" and time_waited < 120.0:
                    # if the update step has not been properly processed yet, wait
                    time.sleep(0.1)
                    time_waited += 0.1
                    executor = calculation.get_executor()
                if time_waited >= 120.0:
                    # Job acquisition timed out, continue
                    continue
                if executor != this_puffin_id:
                    message = 'Wanted to do calculation {:s} with puffin {:s}, but puffin {:s} already wrote into ' \
                              'calculation in the mean time, we got a problem with atomic operations'.format(
                                  str(calculation.id()), this_puffin_id, executor)
                    _log(config, message)
                    _fail_calculation(calculation, config, message, datetime.now())
                    return  # kill puffin since it would do a pointless calculation
                # touch and thus update the timestamp
                calculation.touch()
                # Leave db loop if calculation was found
                break

        # Skip job execution if no calculation found
        else:
            continue

        # =================
        #   Job execution
        # =================

        # Log the job id
        if JOB is not None:
            JOB.value = calculation.id().string().encode("utf-8")

        # Load requested job
        job_name = calculation.get_job().order
        try:
            class_name = available_jobs[job_name]
        except BaseException as e:
            raise KeyError("Missing Job in list of possible jobs.\n" +
                           "Dev-Note: This error should not be reachable.") from e
        module = import_module("scine_puffin.jobs." + job_name)
        class_ = getattr(module, class_name)

        SUCCESS: Any = multiprocessing.Value('i', False)  # Create value in shared memory. Use int for bool flag
        # Run the job in a third process
        JOB_PROCESS = multiprocessing.Process(
            target=_job_execution,
            args=(),
            kwargs={
                "config": config,
                "job_class": class_,
                "manager": manager,
                "calculation": calculation,
                "SUCCESS": SUCCESS,
            },
        )
        JOB_PROCESS.start()
        start = datetime.now()  # in case we need to abort the job, we can set a runtime
        # monitor job for memory usage and wait for finishing
        if bool(config['daemon']['enforce_memory_limit']):
            process = psutil.Process(JOB_PROCESS.pid)
            mem_limit = float(config['resources']['memory'])
            while JOB_PROCESS.is_alive():
                time.sleep(1.0)
                try:
                    memory = process.memory_info().rss
                    children = process.children(recursive=True)
                except BaseException:
                    # process is likely already finished, simply check if still alive
                    continue
                for child in children:
                    try:
                        memory += child.memory_info().rss
                    except BaseException:
                        # child finished between gathering it and trying to access its info
                        # hence its memory usage should be 0 anyway, continue with next child
                        pass
                # make comparison in GB
                if memory / (1024 ** 3) > mem_limit:
                    # we have exceeded memory limit, kill job process and fail calculation with error
                    children = process.children(recursive=True)  # ensure we get the latest possible
                    for child in children:
                        child.kill()
                    process.kill()
                    SUCCESS = None
                    _fail_calculation(calculation, config, "ERROR: Calculation exceeded given memory limit", start)
                    _log(config, "Stopping calculation {:s} because of exceeded memory limit of {:f} GB.".format(
                        str(calculation.id()), mem_limit))
        else:
            # do not enforce just wait
            JOB_PROCESS.join()

        if calculation.get_status() == db.Status.PENDING:
            _fail_calculation(calculation, config, "ERROR: Puffin did not end this calculation properly.\n"
                                                   "Most likely a process during the job killed the puffin with "
                                                   "an illegal instruction.", start)
            _log(config, "Calculation {:s} was not properly ended and its status was set to failed."
                 .format(str(calculation.id())))

        # accounting of maximum number of jobs and maximum serial job fails
        n_jobs_run += 1
        max_n_jobs = config["daemon"]["max_number_of_jobs"]
        if 0 < max_n_jobs <= n_jobs_run:
            _log(config, "Stopping Puffin due to maximum number of jobs ({:d}) being reached".format(max_n_jobs))
            break

        success = False if SUCCESS is None else bool(SUCCESS.value)
        if success:
            previously_failed_job_count = 0
            previously_failed_jobs = []
            previous_dbs = []
        elif CURRENT_DB is not None:
            previously_failed_job_count += 1
            previously_failed_jobs.append(calculation.id())
            previous_dbs.append(CURRENT_DB.value.decode("utf-8"))

        if JOB is not None:
            JOB.value = "".encode("utf-8")

            # Check for repeated job failures and take action
            if previously_failed_job_count >= config["daemon"]["repeated_failure_stop"]:
                # Reset previous jobs
                for idx, pdb in zip(previously_failed_jobs, previous_dbs):
                    manager.set_database_name(pdb)
                    calculations = manager.get_collection("calculations")
                    calculation = db.Calculation(idx, calculations)
                    calculation.set_status(db.Status.NEW)
                    calculation.set_raw_output("")
                    comment = calculation.get_comment()
                    calculation.set_comment("Calculation has been reset.\nComment of previous run: " + comment)
                # Log and exit
                _log(config, "Stopping Puffin due to {:d} consecutive failed jobs".format(previously_failed_job_count))
                break


def _job_execution(config: Configuration, job_class: type, manager, calculation, SUCCESS=None) -> None:
    """
    We are running job in a separate process to save us from SegFaults and enforce memory limit
    """
    job = job_class()
    _log(config, "Processing Job: {:s}".format(str(calculation.id())))
    # Prepare job directory and start timer
    start = datetime.now()
    job.prepare(config["daemon"]["job_dir"], calculation.id())
    # Initialize programs that need initialization
    for program_name, settings in config.programs().items():
        if settings["available"]:
            # Initialize all available programs
            class_name = "".join([s.capitalize() for s in program_name.split("_")])
            module = import_module("scine_puffin.programs." + program_name)
            class_ = getattr(module, class_name)
            class_.initialize()
    # Run job
    success = job.run(manager, calculation, config)
    # we already write a runtime in case puffin fails during copying operations
    # this avoids that a completed calculation is missing a runtime
    # the runtime is set again after copying to include this additional time
    prelim_end = datetime.now()
    calculation.set_runtime((prelim_end - start).total_seconds())
    # Stop if maximum number of jobs is reached
    # Archive if requested and job was successful
    archive = config["daemon"]["archive_dir"]
    if success and archive:
        try:
            job.archive(archive)
        except OSError:
            _log(config, "Failed to archive job: {:s} with error {:s}".format(
                str(calculation.id()), traceback.format_exc()))
    # Archive error if requested and job has failed
    error = config["daemon"]["error_dir"]
    if not success and error:
        try:
            job.archive(error)
        except OSError:
            _log(config, "Failed to archive job: {:s} with error {:s}".format(
                str(calculation.id()), traceback.format_exc()))
    # End timer
    end = datetime.now()
    calculation.set_runtime((end - start).total_seconds())
    # Update job info with actual system specs
    # Clean job remains
    _update_job_specs(calculation, config)
    job.clear()
    # only now set shared variable to signal whether job was success or not
    # if this process dies it is assumed to be False
    SUCCESS.value = success


def _fail_calculation(calculation, config: Configuration, comment_to_add: str, start: datetime) -> None:
    import scine_database as db

    calculation.set_status(db.Status.FAILED)
    _update_job_specs(calculation, config)
    comment = calculation.get_comment()
    comment += "\n" + comment_to_add
    calculation.set_comment(comment)
    calculation.set_runtime((datetime.now() - start).total_seconds())


def _update_job_specs(calculation, config: Configuration) -> None:
    db_job = calculation.get_job()
    db_job.cores = int(config["resources"]["cores"])
    db_job.disk = float(config["resources"]["disk"])
    db_job.memory = float(config["resources"]["memory"])
    calculation.set_job(db_job)
