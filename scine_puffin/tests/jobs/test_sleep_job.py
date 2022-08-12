#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_structure
)

from ..resources import resource_path


class SleepJobTest(JobTestCase):

    @skip_without('database')
    def test_sleep(self):
        # import Job
        from scine_puffin.jobs.sleep import Sleep
        import scine_database as db
        from scine_utilities import ValueCollection

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe', 'def2-SVP')
        job = db.Job('sleep')
        calculation = add_calculation(self.manager, model, job, [structure.id()])
        settings = {"time": 2}
        calculation.set_settings(ValueCollection(settings))

        # Run calculation/job
        config = self.get_configuration()
        job = Sleep()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
