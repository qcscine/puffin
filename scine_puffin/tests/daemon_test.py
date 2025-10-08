#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from importlib import import_module
from uuid import uuid1
from multiprocessing import Process
import os
import unittest

from scine_puffin.config import Configuration
from .testcases import skip_without
from .resources import resource_path
from ..daemon import start_daemon
from ..jobloop import _loop_impl, check_setup


class DatabaseConnection(unittest.TestCase):

    @skip_without('database')
    def test_database_connection(self):
        from .db_setup import get_clean_db

        manager = get_clean_db("puffin_unittests_db_connection")
        manager.wipe()


class ValidJobClasses(unittest.TestCase):

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


class UnavailableProgramsTest(unittest.TestCase):

    def setUp(self):
        from .db_setup import get_clean_db
        name = "puffin_unittests_unavailable_programs"
        self.manager = get_clean_db(name)
        self.config = Configuration()
        self.config["daemon"]["cycle_time_in_s"] = 0.1
        self.config["daemon"]["idle_timeout_in_h"] = 0.001
        self.config["daemon"]["max_number_of_jobs"] = 1
        self.config["daemon"]["log"] = ""
        self.config["daemon"]["job_dir"] = str(os.path.curdir)
        self.config["daemon"]["pid_dir"] = str(os.path.curdir)
        self.config["daemon"]["stop"] = str(os.path.join(os.path.curdir, "puffin.stop"))
        id_ = uuid1().hex
        self.config["daemon"]["uuid"] = id_
        self.config["daemon"]["pid"] = str(os.path.join(os.path.curdir, f"{id_}.pid"))

        credentials = self.manager.get_credentials()
        self.config["database"]["ip"] = credentials.hostname
        self.config["database"]["port"] = credentials.port
        self.config["database"]["name"] = credentials.database_name

    def tearDown(self):
        self.manager.wipe()

    @skip_without('database')
    def test_all_available(self):
        import scine_database as db
        from .db_setup import add_structure

        structure_path = os.path.join(resource_path(), "OOCHH.xyz")
        s = add_structure(self.manager, structure_path, db.Label.USER_OPTIMIZED)
        calculation = db.Calculation.make(
            db.Model("dftb3", "dftb3", ""),
            db.Job("sleep"),
            [s.id()],
            self.manager.get_collection("calculations")
        )
        calculation.set_setting("time", 0.1)
        calculation.set_status(db.Status.NEW)
        _loop_impl(self.config, check_setup(self.config))
        assert calculation.get_status() != db.Status.NEW

    @skip_without('database')
    def test_standard_unavailable(self):
        import scine_database as db
        from .db_setup import add_structure

        structure_path = os.path.join(resource_path(), "OOCHH.xyz")
        s = add_structure(self.manager, structure_path, db.Label.USER_OPTIMIZED)
        calculation1 = db.Calculation.make(
            db.Model("dftb3", "dftb3", ""),
            db.Job("sleep"),
            [s.id()],
            self.manager.get_collection("calculations")
        )
        calculation1.set_setting("time", 0.1)
        calculation1.set_status(db.Status.NEW)

        calculation2 = db.Calculation.make(
            db.Model("dftb3", "dftb3", ""),
            db.Job("graph"),
            [s.id()],
            self.manager.get_collection("calculations")
        )
        calculation2.set_status(db.Status.NEW)
        self.config["programs"]["molassembler"]["available"] = False

        def run():
            start_daemon(self.config, detach=False)
            assert calculation1.get_status() != db.Status.NEW
            start_daemon(self.config, detach=False)

        p = Process(target=run)
        p.start()
        p.join()
        assert calculation1.get_status() != db.Status.NEW
        assert calculation2.get_status() == db.Status.NEW

    @skip_without('database')
    def test_settings_unavailable(self):
        import scine_database as db
        from .db_setup import add_structure

        structure_path = os.path.join(resource_path(), "OOCHH.xyz")
        s = add_structure(self.manager, structure_path, db.Label.USER_OPTIMIZED)
        calculation2 = db.Calculation.make(
            db.Model("dftb3", "dftb3", ""),
            db.Job("sleep"),
            [s.id()],
            self.manager.get_collection("calculations")
        )
        calculation2.set_setting("time", 0.1)
        calculation2.set_status(db.Status.NEW)

        calculation1 = db.Calculation.make(
            db.Model("dftb3", "dftb3", ""),
            db.Job("scine_react_complex_nt2_pes_switch"),
            [s.id()],
            self.manager.get_collection("calculations")
        )
        calculation1.set_setting("model_switch_program", "turbomole")
        calculation1.set_status(db.Status.NEW)
        self.config["programs"]["turbomole"]["available"] = False

        def run():
            start_daemon(self.config, detach=False)
            assert calculation2.get_status() != db.Status.NEW
            start_daemon(self.config, detach=False)

        p = Process(target=run)
        p.start()
        p.join()
        assert calculation2.get_status() != db.Status.NEW
        assert calculation1.get_status() == db.Status.NEW
