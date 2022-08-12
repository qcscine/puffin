#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from importlib import import_module
import os
import unittest

from scine_puffin.config import Configuration
from .testcases import skip_without


class DatabaseConnection(unittest.TestCase):

    @skip_without('database')
    def test_database_connection(self):
        from .db_setup import get_clean_db

        manager = get_clean_db("puffin_unittests_db_connection")
        manager.wipe()


class ValidJobClasses(unittest.TestCase):
    @skip_without('database')
    def test_job_folder(self):
        config = Configuration()
        config.load()
        programs = config.programs().keys()
        all_jobs = []
        import scine_puffin.jobs

        for path in scine_puffin.jobs.__path__:
            for _, dirs, files in os.walk(path):
                for name in files:
                    if name.endswith(".py") and name != "__init__.py" and "templates" in dirs and "deprecated" in dirs:
                        all_jobs.append(name[:-3])

        for job in all_jobs:
            class_name = "".join([s.capitalize() for s in job.split("_")])
            module = import_module("scine_puffin.jobs." + job)
            class_ = getattr(module, class_name)  # fails if wrong class names
            required_programs = class_.required_programs()
            for program in required_programs:
                if program not in programs:
                    raise RuntimeError("Job gives a program that is not available in the Configuration!")
