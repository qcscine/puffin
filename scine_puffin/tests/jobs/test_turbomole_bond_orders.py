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


class TurbomoleBondOrdersJobTest(JobTestCase):

    @skip_without('database', 'turbomole')
    def test_energy(self):
        # import Job
        from scine_puffin.jobs.turbomole_bond_orders import TurbomoleBondOrders
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe', 'def2-SVP')
        job = db.Job('turbomole_bond_orders')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = TurbomoleBondOrders()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        bo_props = structure.get_properties("bond_orders")
        assert len(energy_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 2
        assert energy_props[0] in results.property_ids
        assert bo_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -76.26848572171)
        # Bond orders
        bond_orders = db.SparseMatrixProperty(bo_props[0])
        bond_orders.link(properties)
        bos = bond_orders.get_data()
        self.assertAlmostEqual(bos[0, 1], +1.004910000, delta=1e-1)
        self.assertAlmostEqual(bos[0, 2], +1.004910000, delta=1e-1)
        self.assertAlmostEqual(bos[1, 0], +1.004910000, delta=1e-1)
        self.assertAlmostEqual(bos[2, 0], +1.004910000, delta=1e-1)
        self.assertAlmostEqual(bos[0, 0], +0.000000000)
        self.assertAlmostEqual(bos[1, 1], +0.000000000)
        self.assertAlmostEqual(bos[2, 2], +0.000000000)
        self.assertAlmostEqual(bos[2, 1], +0.000000000)
        self.assertAlmostEqual(bos[1, 2], +0.000000000)
