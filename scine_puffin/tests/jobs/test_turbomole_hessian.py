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


class TurbomoleHessianJobTest(JobTestCase):

    @skip_without('database', 'turbomole')
    def test_hessian(self):
        # import Job
        from scine_puffin.jobs.turbomole_hessian import TurbomoleHessian
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe', 'def2-SVP')
        job = db.Job('turbomole_hessian')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = TurbomoleHessian()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        assert structure.has_property("hessian")
        energy_props = structure.get_properties("electronic_energy")
        hessian_props = structure.get_properties("hessian")
        assert len(energy_props) == 1
        assert len(hessian_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 2
        assert energy_props[0] in results.property_ids
        assert hessian_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -76.26848572171, delta=1e-1)
        # Hessian
        hessian = db.DenseMatrixProperty(hessian_props[0])
        hessian.link(properties)
        h = hessian.get_data()
        self.assertAlmostEqual(h[0][0], +0.4298600028, delta=1e-1)
