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


class ScineSinglePointJobTest(JobTestCase):

    @skip_without('database', 'readuct')
    def test_energy(self):
        # import Job
        from scine_puffin.jobs.scine_single_point import ScineSinglePoint
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_single_point')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineSinglePoint()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        assert structure.has_property("atomic_charges")
        energy_props = structure.get_properties("electronic_energy")
        charge_props = structure.get_properties("atomic_charges")
        assert len(energy_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 2
        assert energy_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -4.061143327, delta=1e-3)
        # Charges
        atomic_charges = db.VectorProperty(charge_props[0])
        atomic_charges.link(properties)
        charges = atomic_charges.get_data()
        assert len(charges) == 3
        self.assertAlmostEqual(charges[0], -0.67969209, delta=1e-3)
        self.assertAlmostEqual(charges[1], +0.33984604, delta=1e-3)
        self.assertAlmostEqual(charges[2], +0.33984604, delta=1e-3)
