#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
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


class MrccTests(JobTestCase):

    @skip_without('mrcc', 'database', 'readuct')
    def test_mrcc_single_point(self):
        from scine_puffin.jobs.scine_single_point import ScineSinglePoint
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe-d3bj', 'def2-svp')
        model.temperature = ""
        model.pressure = ""
        model.electronic_temperature = ""
        model.program = "mrcc"
        settings = {
            "require_charges": False
        }
        job = db.Job('scine_single_point')
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineSinglePoint()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        assert len(energy_props) == 1
        results = calculation.get_results()
        assert energy_props[0] in results.property_ids
