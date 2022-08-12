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


class ScineBondOrdersJobTest(JobTestCase):

    @skip_without('database', 'gaussian')
    def test_water(self):
        # import Job
        from scine_puffin.jobs.gaussian_charge_model_5 import GaussianChargeModel5
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_OPTIMIZED)
        model = db.Model('dft', 'PBEPBE', '6-31G')
        job = db.Job('gaussian_charge_model_5')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = GaussianChargeModel5()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("cm5_charges")
        charge_props = structure.get_properties("cm5_charges")
        assert len(charge_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 1
        assert charge_props[0] in results.property_ids

        # Check generated properties
        # Charges
        properties = self.manager.get_collection("properties")
        atomic_charges = db.VectorProperty(charge_props[0])
        atomic_charges.link(properties)
        charges = atomic_charges.get_data()
        assert len(charges) == 3
        self.assertAlmostEqual(charges[0], -0.622577, delta=1e-1)
        self.assertAlmostEqual(charges[1], +0.311290, delta=1e-1)
        self.assertAlmostEqual(charges[2], +0.311290, delta=1e-1)
