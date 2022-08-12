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


class TurbomoleSinglePointJobTest(JobTestCase):

    @skip_without('database', 'turbomole')
    def test_energy(self):
        # import Job
        from scine_puffin.jobs.turbomole_single_point import TurbomoleSinglePoint
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe', 'def2-SVP')
        job = db.Job('turbomole_single_point')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = TurbomoleSinglePoint()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        assert len(energy_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 1
        assert energy_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -76.26848572171, delta=1e-1)

    @skip_without('database', 'turbomole')
    def test_charges(self):
        # import Job
        from scine_puffin.jobs.turbomole_single_point import TurbomoleSinglePoint
        import scine_database as db
        import scine_utilities as utils

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe', 'def2-SVP')
        job = db.Job('turbomole_single_point')
        calculation = add_calculation(self.manager, model, job, [structure.id()])
        settings = utils.ValueCollection({
            "calculate_loewdin_charges": True
        })
        calculation.set_settings(settings)

        # Run calculation/job
        config = self.get_configuration()
        job = TurbomoleSinglePoint()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        assert len(energy_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 2
        assert energy_props[0] in results.property_ids

        assert structure.has_property("loewdin_charges")
        charge_props = structure.get_properties("loewdin_charges")
        assert len(charge_props) == 1
        assert charge_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -76.26848572171, delta=1e-1)

        # Charges
        properties = self.manager.get_collection("properties")
        charges = db.VectorProperty(charge_props[0])
        charges.link(properties)
        self.assertAlmostEqual(charges.get_data()[0], -0.35773, delta=1e-1)
        self.assertAlmostEqual(charges.get_data()[1], +0.17886, delta=1e-1)
        self.assertAlmostEqual(charges.get_data()[2], +0.17886, delta=1e-1)
