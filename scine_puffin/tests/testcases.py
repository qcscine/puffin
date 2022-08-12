#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import unittest
from functools import wraps
from pkgutil import iter_modules
from typing import Callable, Dict, List, Union
from .db_setup import get_clean_db
from scine_puffin.config import Configuration
import os
import shutil


def module_exists(module_name: str) -> bool:
    if module_name.lower() == "cp2k":
        return os.getenv("CP2K_BINARY_PATH") is not None
    elif module_name.lower() == "gaussian":
        return os.getenv("GAUSSIAN_BINARY_PATH") is not None
    elif module_name.lower() == "orca":
        return os.getenv("ORCA_BINARY_PATH") is not None
    elif module_name.lower() == "turbomole":
        return os.getenv("TURBODIR") is not None
    else:
        return module_name in (name for loader, name, ispkg in iter_modules())


def _skip(func: Callable, error: str):
    if module_exists("pytest"):
        import pytest
        return pytest.mark.skip(reason=error)(func)
    else:
        return unittest.skip(error)(func)


def dependency_addition(dependencies: List[str]) -> List[str]:
    # allow to give scine packages without 'scine_' prefix
    short_terms = ['readuct', 'swoose', 'sparrow', 'molassembler', 'database', 'utilities', 'kinetx', 'xtb_wrapper']
    dependencies = ['scine_' + d if d in short_terms else d for d in dependencies]
    # dependencies of key as value list, only utilities must not be included
    dependency_data = {
        'scine_readuct': ['scine_sparrow'],
    }
    for package, dependency in dependency_data.items():
        if package in dependencies:
            dependencies += dependency
    # add utilities
    if any('scine' in d for d in dependencies):
        dependencies.append('scine_utilities')
    return list(set(dependencies))  # give back unique list


def skip_without(*dependencies) -> Callable:
    dependency_list: List[str] = list(dependencies)
    dependency_list = dependency_addition(dependency_list)

    def wrap(f: Callable):
        if all(module_exists(d) for d in dependency_list):
            @wraps(f)
            def wrapped_f(*args, **kwargs):
                f(*args, **kwargs)
            return wrapped_f
        else:
            return _skip(f, "Test requires {:s}".format([d for d in dependency_list if not module_exists(d)][0]))

    return wrap


class JobTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
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
        shutil.rmtree(os.path.join(os.getcwd(), "puffin_unittest_scratch"))

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

        return config

    def run_job(self, job, calculation, config):
        try:
            success = job.run(self.manager, calculation, config)
            assert success
        except BaseException as e:
            print(calculation.get_comment())
            raise e
