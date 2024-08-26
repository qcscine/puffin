#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import sys
import unittest
from functools import wraps
from typing import Callable, Dict, List, Union, TYPE_CHECKING
import os
import shutil

from scine_puffin.config import Configuration
from scine_puffin.utilities.imports import (module_exists, calculator_import_resolve, dependency_addition,
                                            MissingDependency)
from .db_setup import get_clean_db

if module_exists("pytest") or TYPE_CHECKING:
    import pytest
else:
    pytest = MissingDependency("pytest")


def _skip(func: Callable, error: str):
    if module_exists("pytest"):
        return pytest.mark.skip(reason=error)(func)
    else:
        return unittest.skip(error)(func)


def skip_without(*dependencies) -> Callable:
    """
    This function is meant to be used as a decorator for individual unittest functions. It will skip the test if the
    required dependencies are not installed or if the database is not running.

    Example
    -------
    Add this function as a decorator with the required modules as arguments

    >>> @skip_without("readuct", "database")
    >>> def test_function():
    >>>     ...

    Returns
    -------
    Callable
        The wrapped function.
    """
    dependency_list: List[str] = list(dependencies)
    dependency_list = dependency_addition(dependency_list)

    def wrap(f: Callable):

        if not all(module_exists(d) for d in dependency_list):
            return _skip(f, "Test requires {:s}".format([d for d in dependency_list if not module_exists(d)][0]))

        @wraps(f)
        def wrapped_f(*args, **kwargs):
            calculator_import_resolve(dependency_list)
            f(*args, **kwargs)

        if "scine_database" not in dependency_list:
            return wrapped_f
        try:
            get_clean_db()
            return wrapped_f
        except RuntimeError as e:
            if module_exists("pytest"):
                pytest.exit("{:s}\nFirst start database before running unittests.".format(str(e)))
            else:
                print("{:s}\nFirst start database before running unittests.".format(str(e)))
                sys.exit(1)

    return wrap


class JobTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.db_name = "default_puffin_unittest_db"
        self.start_dir = os.getcwd()

    def setUp(self):
        if module_exists("scine_database"):
            self.manager = get_clean_db(self.db_name)

    def tearDown(self):
        if module_exists("scine_database"):
            self.manager.wipe()
        os.chdir(self.start_dir)
        work_dir = os.path.join(os.getcwd(), "puffin_unittest_scratch")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def get_calculation(self, query: Union[Dict[str, str], None] = None):
        calculations = self.manager.get_collection("calculations")
        if query is None:
            query = {}
        update = {
            "$set": {
                "status": "pending",
                "executor": "pytest"
            }
        }
        calculation = calculations.get_and_update_one_calculation(query, update)
        calculation.link(calculations)
        return calculation

    def get_configuration(self):
        config = Configuration()
        config["daemon"]["mode"] = "debug"
        config["daemon"]["job_dir"] = os.path.join(os.getcwd(), "puffin_unittest_scratch")
        config["daemon"]["log"] = os.path.join(os.getcwd(), "puffin_unittest.log")
        config["daemon"]["stop"] = os.path.join(os.getcwd(), "puffin_unittest.stop")
        if os.getenv("TURBODIR") is not None:
            config['programs']['turbomole']['available'] = True
            config['programs']['turbomole']['root'] = os.getenv("TURBODIR")
        if os.getenv("ORCA_BINARY_PATH") is not None:
            config['programs']['orca']['available'] = True
            config['programs']['orca']['root'] = os.path.dirname(os.getenv("ORCA_BINARY_PATH"))
        if os.getenv("GAUSSIAN_BINARY_PATH") is not None:
            config['programs']['gaussian']['available'] = True
            config['programs']['gaussian']['root'] = os.path.dirname(os.path.dirname(os.getenv("GAUSSIAN_BINARY_PATH")))
        if os.getenv("CP2K_BINARY_PATH") is not None:
            config['programs']['cp2k']['available'] = True
            config['programs']['cp2k']['root'] = os.getenv("CP2K_BINARY_PATH")
        if os.getenv("AMSHOME") is not None:
            config['programs']['ams']['available'] = True
            config['programs']['ams']['root'] = os.getenv("AMSHOME")
        if os.getenv("MRCC_BINARY_PATH") is not None:
            config['programs']['mrcc']['available'] = True
            config['programs']['mrcc']['root'] = os.getenv("MRCC_BINARY_PATH")

        return config

    def run_job(self, job, calculation, config):
        try:
            success = job.run(self.manager, calculation, config)
            assert success
            job.clear()
        except BaseException as e:
            print(calculation.get_comment())
            job.clear()
            raise e
