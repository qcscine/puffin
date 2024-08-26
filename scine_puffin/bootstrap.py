# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
import os
import sys
import importlib
from .config import Configuration
from .programs.utils import Utils


def bootstrap(config: Configuration):
    """
    Sets up all required and also all additionally requested programs/packages
    for the use with Puffin.
    Generates a ``puffin.sh`` to be sourced before running the actual puffin.

    Parameters
    ----------.
    config : scine_puffin.config.Configuration
       The current configuration of the Puffin.
    """
    # Prepare directories
    initial_dir = os.getcwd()
    jobs = config.daemon()["job_dir"]
    if jobs and not os.path.exists(jobs):
        try:
            os.makedirs(jobs)
        except FileExistsError:
            pass
    software = config.daemon()["software_dir"]
    if software and not os.path.exists(software):
        try:
            os.makedirs(software)
        except FileExistsError:
            pass
    build_dir = os.path.join(software, "build")
    if build_dir and not os.path.exists(build_dir):
        try:
            os.makedirs(build_dir)
        except FileExistsError:
            pass
    install_dir = os.path.join(software, "install")
    if install_dir and not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir)
        except FileExistsError:
            pass
    archive_dir = config.daemon()["archive_dir"]
    if archive_dir and not os.path.exists(archive_dir) and archive_dir:
        try:
            os.makedirs(archive_dir)
        except FileExistsError:
            pass
    error_dir = config.daemon()["error_dir"]
    if error_dir and not os.path.exists(error_dir) and error_dir:
        try:
            os.makedirs(error_dir)
        except FileExistsError:
            pass

    # Install minimal requirement
    print("")
    print("Building SCINE Core/Utils from sources.")
    print("")
    core_build_dir = os.path.join(build_dir, "core")
    core = Utils(config.programs()["core"])
    core.install(core_build_dir, install_dir, config["resources"]["cores"])
    utils_build_dir = os.path.join(build_dir, "utils")
    utils = Utils(config.programs()["utils"])
    utils.install(utils_build_dir, install_dir, config["resources"]["cores"])

    # setup Python path already now for crosslinking for Python type stubs
    env = {}
    python_version = sys.version_info
    env["PYTHONPATH"] = (
        os.path.join(
            install_dir,
            "lib",
            "python" + str(python_version[0]) + "." + str(python_version[1]),
            "site-packages",
        )
        + ":"
        + os.path.join(
            install_dir,
            "lib64",
            "python" + str(python_version[0]) + "." + str(python_version[1]),
            "site-packages",
        )
        + ":"
        + os.path.join(
            install_dir,
            "local",
            "lib",
            "python" + str(python_version[0]) + "." + str(python_version[1]),
            "dist-packages",
        )
        + ":"
        + os.path.join(
            install_dir,
            "local",
            "lib64",
            "python" + str(python_version[0]) + "." + str(python_version[1]),
            "dist-packages",
        )
    )
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]

    # Install all other programs
    for program_name, settings in config.programs().items():
        if program_name in ['core', 'utils'] or not settings["available"]:
            continue
        print("")
        print("Preparing " + program_name.capitalize() + "...")
        print("")
        module = importlib.import_module("scine_puffin.programs." + program_name)
        class_ = getattr(module, program_name.capitalize())
        program = class_(settings)
        program_build_dir = os.path.join(build_dir, program_name)
        program.install(program_build_dir, install_dir, config["resources"]["cores"])

    # Setup environment
    #  General setup
    executables = {}
    executables["OMP_NUM_THREADS"] = str(config["resources"]["cores"])
    env["PATH"] = os.path.join(install_dir, "bin")
    env["LD_LIBRARY_PATH"] = os.path.join(install_dir, "lib") + ":" + os.path.join(install_dir, "lib64")
    env["SCINE_MODULE_PATH"] = os.path.join(install_dir, "lib") + ":" + os.path.join(install_dir, "lib64")
    #  Program specific environment setup
    for program_name, settings in config.programs().items():
        if not settings["available"]:
            continue
        module = importlib.import_module("scine_puffin.programs." + program_name)
        class_ = getattr(module, program_name.capitalize())
        program = class_(settings)
        program.setup_environment(config, env, executables)

    os.chdir(initial_dir)
    # Windows TODO also generate a bat file
    with open("puffin.sh", "w") as f:
        for key, paths in env.items():
            f.write(f"export {key}={paths}:${key}\n")
        for key, paths in executables.items():
            f.write(f"export {key}={paths}\n")
