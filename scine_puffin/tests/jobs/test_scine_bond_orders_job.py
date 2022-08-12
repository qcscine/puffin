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

    @skip_without('database', 'molassembler', 'readuct')
    def test_water(self):
        # import Job
        from scine_puffin.jobs.scine_bond_orders import ScineBondOrders
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_OPTIMIZED)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_bond_orders')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineBondOrders()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        assert structure.has_property("atomic_charges")
        assert structure.has_property("bond_orders")
        energy_props = structure.get_properties("electronic_energy")
        bo_props = structure.get_properties("bond_orders")
        charge_props = structure.get_properties("atomic_charges")
        assert len(energy_props) == 1
        assert len(bo_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 3
        assert charge_props[0] in results.property_ids
        assert energy_props[0] in results.property_ids
        assert bo_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -4.061143327)
        # Charges
        atomic_charges = db.VectorProperty(charge_props[0])
        atomic_charges.link(properties)
        charges = atomic_charges.get_data()
        assert len(charges) == 3
        self.assertAlmostEqual(charges[0], -0.67969209, delta=1e-1)
        self.assertAlmostEqual(charges[1], +0.33984604, delta=1e-1)
        self.assertAlmostEqual(charges[2], +0.33984604, delta=1e-1)
        # Bond orders
        bond_orders = db.SparseMatrixProperty(bo_props[0])
        bond_orders.link(properties)
        bos = bond_orders.get_data()
        self.assertAlmostEqual(bos[0, 1], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bos[0, 2], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bos[1, 0], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bos[2, 0], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bos[0, 0], +0.000000000)
        self.assertAlmostEqual(bos[1, 1], +0.000000000)
        self.assertAlmostEqual(bos[2, 2], +0.000000000)
        self.assertAlmostEqual(bos[2, 1], +0.003157004, delta=1e-1)
        self.assertAlmostEqual(bos[1, 2], +0.003157004, delta=1e-1)

    @skip_without('database', 'molassembler')
    def test_only_distance_connectivity(self):
        from scine_puffin.jobs.scine_bond_orders import ScineBondOrders
        import scine_database as db
        import scine_utilities as utils

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_OPTIMIZED)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_bond_orders')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        settings = {
            'only_distance_connectivity': True
        }
        calculation.set_settings(utils.ValueCollection(settings))

        # Run calculation/job
        config = self.get_configuration()
        job = ScineBondOrders()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("bond_orders")
        bo_props = structure.get_properties("bond_orders")
        assert len(bo_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 1
        assert bo_props[0] in results.property_ids

        # Check generated property
        properties = self.manager.get_collection("properties")
        bond_orders = db.SparseMatrixProperty(bo_props[0], properties)
        bos = bond_orders.get_data()
        self.assertAlmostEqual(bos[0, 1], +1.0, delta=1e-1)
        self.assertAlmostEqual(bos[0, 2], +1.0, delta=1e-1)
        self.assertAlmostEqual(bos[1, 0], +1.0, delta=1e-1)
        self.assertAlmostEqual(bos[2, 0], +1.0, delta=1e-1)
        self.assertAlmostEqual(bos[0, 0], +0.0)
        self.assertAlmostEqual(bos[1, 1], +0.0)
        self.assertAlmostEqual(bos[2, 2], +0.0)
        self.assertAlmostEqual(bos[2, 1], +0.0)
        self.assertAlmostEqual(bos[1, 2], +0.0)
