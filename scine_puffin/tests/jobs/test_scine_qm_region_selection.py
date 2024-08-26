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


class ScineQmRegionSelectionTest(JobTestCase):

    @skip_without('database', 'swoose', 'readuct', 'xtb_wrapper')
    def test_selection_full_qm_reference(self):
        from scine_puffin.jobs.scine_qm_region_selection import ScineQmRegionSelection
        from scine_puffin.jobs.scine_bond_orders import ScineBondOrders
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "8-gly-chain-opt.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)

        # Calculate bond orders
        bond_order_model = db.Model('dftb3', 'dftb3', '')
        bond_order_job = db.Job('scine_bond_orders')
        bond_order_calculation = add_calculation(self.manager, bond_order_model, bond_order_job, [structure.id()])
        config = self.get_configuration()
        job = ScineBondOrders()
        job.prepare(config["daemon"]["job_dir"], bond_order_calculation.id())
        self.run_job(job, bond_order_calculation, config)

        # Set QM atoms
        properties = self.manager.get_collection("properties")

        # Set up + Run QM/MM
        qmmm_model = db.Model('gfn2/gaff', 'gfn2', '')
        qmmm_model.program = "xtb/swoose"
        job = db.Job('scine_qm_region_selection')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job, [structure.id()])
        settings = qmmm_calculation.get_settings()
        settings["electrostatic_embedding"] = True
        settings["qm_region_center_atoms"] = [0]
        settings["initial_radius"] = 5.1
        settings["cutting_probability"] = 0.7
        settings["tol_percentage_error"] = 20.0
        settings["qm_region_max_size"] = 50
        settings["qm_region_min_size"] = 20
        settings["ref_max_size"] = 100
        settings["tol_percentage_sym_score"] = float("inf")

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineQmRegionSelection()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config)

        assert structure.has_property("qm_atoms")
        qm_atoms = db.VectorProperty(structure.get_property("qm_atoms"), properties).get_data()
        assert len(qm_atoms) == 27
        reference_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26,
                               27, 71]
        assert list(qm_atoms) == reference_selection

    @skip_without('database', 'swoose', 'readuct', 'xtb_wrapper')
    def test_selection_qm_mm_reference(self):
        from scine_puffin.jobs.scine_qm_region_selection import ScineQmRegionSelection
        from scine_puffin.jobs.scine_bond_orders import ScineBondOrders
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "8-gly-chain-opt.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)

        # Calculate bond orders
        bond_order_model = db.Model('dftb3', 'dftb3', '')
        bond_order_job = db.Job('scine_bond_orders')
        bond_order_calculation = add_calculation(self.manager, bond_order_model, bond_order_job, [structure.id()])
        config = self.get_configuration()
        job = ScineBondOrders()
        job.prepare(config["daemon"]["job_dir"], bond_order_calculation.id())
        self.run_job(job, bond_order_calculation, config)

        # Set QM atoms
        properties = self.manager.get_collection("properties")

        # Set up + Run QM/MM
        qmmm_model = db.Model('gfn2/gaff', 'gfn2', '')
        qmmm_model.program = "xtb/swoose"
        job = db.Job('scine_qm_region_selection')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job, [structure.id()])
        settings = qmmm_calculation.get_settings()
        settings["electrostatic_embedding"] = True
        settings["qm_region_center_atoms"] = [0]
        settings["initial_radius"] = 2.0
        settings["cutting_probability"] = 0.7
        settings["tol_percentage_error"] = 20.0
        settings["qm_region_max_size"] = 40
        settings["qm_region_min_size"] = 10
        settings["ref_max_size"] = 70
        settings["tol_percentage_sym_score"] = float("inf")

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineQmRegionSelection()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config)

        assert structure.has_property("qm_atoms")
        qm_atoms = db.VectorProperty(structure.get_property("qm_atoms"), properties).get_data()
        assert len(qm_atoms) == 27
        reference_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26,
                               27, 71]
        assert list(qm_atoms) == reference_selection
