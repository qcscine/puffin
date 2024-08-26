#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import numpy as np

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
        assert len(results.elementary_step_ids) == 0
        assert len(results.structure_ids) == 0
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

    @skip_without('database', 'readuct', 'swoose')
    def test_qmmm_gaff(self):
        # import Job
        from scine_puffin.jobs.scine_single_point import ScineSinglePoint
        from scine_puffin.jobs.scine_bond_orders import ScineBondOrders
        from scine_database.energy_query_functions import get_energy_for_structure
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "proline_acid_propanal_complex.xyz")
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
        structures = self.manager.get_collection("structures")
        n_atoms = len(structure.get_atoms())
        qm_atoms = [i for i in range(n_atoms - 10)]  # the 10 last atoms are propanal. We assign them as MM.
        qm_atom_label = "qm_atoms"
        prop = db.VectorProperty.make(qm_atom_label, bond_order_model, np.asarray(qm_atoms), properties)
        structure.add_property(qm_atom_label, prop.id())
        prop.set_structure(structure.id())

        # Set up + Run QM/MM
        qmmm_model = db.Model('gfn2/gaff', 'gfn2', '')
        qmmm_model.program = "xtb/swoose"
        job = db.Job('scine_single_point')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job, [structure.id()])
        settings = qmmm_calculation.get_settings()
        settings["require_gradients"] = True
        settings["require_charges"] = False
        settings["electrostatic_embedding"] = True
        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineSinglePoint()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config)

        energy_reference = -26.352255863
        gradient_reference = [
            [8.237708e-04, 3.125493e-03, 2.131060e-03],
            [1.293170e-03, -4.198595e-03, -2.841488e-03],
            [-4.210309e-03, 8.769684e-03, 2.051029e-04],
            [5.614330e-03, -2.083648e-03, 8.612634e-03],
            [1.969390e-03, -4.311829e-03, -8.419946e-03],
            [-1.046279e-03, -9.618556e-04, -2.861473e-03],
            [-2.420135e-03, 8.110524e-05, -1.119392e-03],
            [-1.704751e-03, 1.552419e-03, -1.657612e-03],
            [1.221916e-03, 5.193619e-03, -3.087307e-03],
            [-1.997496e-03, -4.045199e-03, 4.388533e-03],
            [1.700178e-03, -3.329661e-03, 1.778612e-04],
            [5.710796e-03, -3.283295e-05, -4.074882e-03],
            [-1.365384e-03, -1.367140e-03, -1.667894e-03],
            [1.574524e-02, -7.115792e-03, -9.660434e-03],
            [-3.329011e-02, 9.832087e-03, 2.034265e-03],
            [1.161146e-02, 1.462071e-04, 1.792546e-02],
            [1.244628e-03, -3.579307e-04, 1.707141e-04],
            [1.478093e-03, 2.555476e-02, 1.863031e-02],
            [2.022248e-03, 1.952497e-02, 9.900692e-05],
            [-4.767288e-04, -9.485054e-03, -8.243565e-04],
            [5.520370e-04, -7.912631e-03, -1.061546e-02],
            [-1.581161e-03, -1.971934e-02, -2.166321e-03],
            [5.098476e-04, -2.937521e-03, -2.085972e-03],
            [-3.207473e-03, -3.957743e-03, -2.584540e-03],
            [3.027584e-03, -9.003679e-04, -1.412400e-03],
            [-3.045696e-03, -3.790679e-04, -1.725070e-03],
            [-1.791648e-04, -6.841384e-04, 2.429609e-03]
        ]
        assert qmmm_calculation.get_status() == db.Status.COMPLETE
        gradient_property_ids = structure.query_properties("gradients", qmmm_model, properties)
        assert len(gradient_property_ids) > 0
        gradient_property = db.DenseMatrixProperty(gradient_property_ids[0], properties)
        assert np.max(np.abs(gradient_property.get_data() - gradient_reference)) < 1e-6

        energy = get_energy_for_structure(structure, "electronic_energy", qmmm_model, structures, properties)
        assert abs(energy - energy_reference) < 1e-6

        mm_model = db.Model('gaff', '', '')
        mm_model.program = "swoose"
        mm_model.spin_mode = ""
        mm_model.temperature = ""
        mm_model.electronic_temperature = ""
        mm_model.pressure = ""
        job = db.Job('scine_single_point')
        mm_calculation = add_calculation(self.manager, mm_model, job, [structure.id()])
        settings = mm_calculation.get_settings()
        settings["require_gradients"] = True
        mm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineSinglePoint()
        job.prepare(config["daemon"]["job_dir"], mm_calculation.id())
        self.run_job(job, mm_calculation, config)
        gradient_property_ids = structure.query_properties("gradients", mm_model, properties)
        assert len(gradient_property_ids) == 1
        energy = get_energy_for_structure(structure, "electronic_energy", mm_model, structures, properties)
        assert abs(energy - 0.012852055) < 1e-6
