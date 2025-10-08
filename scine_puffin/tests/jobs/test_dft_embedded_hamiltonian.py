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


class DFTEmbeddedHamiltonianJobTest(JobTestCase):
    def setup_acetone(self):
        import scine_database as db

        system_path = os.path.join(resource_path(), "acetone_radical.xyz")
        structure = add_structure(self.manager, system_path, db.Label.USER_GUESS, charge=0, multiplicity=2)

        model = db.Model('hf/dft', 'hf/pbe-d3bj', 'def2-SVP')
        model.program = "serenity/serenity"
        job = db.Job('dft_embedded_hamiltonian')
        calculation = add_calculation(self.manager, model, job, [structure.id()])
        qm_atoms = [idx for idx in range(11)]
        qm_core = [0, 1, 2]
        environment = list(set(qm_atoms) - set(qm_core))
        settings = calculation.get_settings()
        # Calculator settings
        settings["system_partitioning"] = "POPULATION_THRESHOLD"
        settings["qmqm_atom_indices"] = [qm_core, environment]
        settings["show_serenity_output"] = True
        settings["static_embedding"] = True
        settings["orbital_threshold"] = 0.6
        settings["cas_systems"] = [0]
        # Tasks settings
        settings["require_charges"] = False
        settings["require_partial_energies"] = False
        calculation.set_settings(settings)
        return calculation, structure

    def setup_qmmm_acetone(self):
        import scine_database as db
        from scine_puffin.jobs.scine_bond_orders import ScineBondOrders

        system_path = os.path.join(resource_path(), "qmmm_acetone_radical.xyz")
        structure = add_structure(self.manager, system_path, db.Label.USER_GUESS, charge=0, multiplicity=2)

        # Calculate bond orders
        bond_order_model = db.Model('dftb3', 'dftb3', '')
        bond_order_job = db.Job('scine_bond_orders')
        bond_order_calculation = add_calculation(self.manager, bond_order_model, bond_order_job, [structure.id()])
        config = self.get_configuration()
        job = ScineBondOrders()
        job.prepare(config["daemon"]["job_dir"], bond_order_calculation.id())
        self.run_job(job, bond_order_calculation, config)

        model = db.Model('hf/dft/gaff', 'hf/pbe-d3bj', 'def2-SVP')
        model.program = "serenity/serenity/swoose"
        job = db.Job('dft_embedded_hamiltonian')
        calculation = add_calculation(self.manager, model, job, [structure.id()])
        qm_atoms = [idx for idx in range(11)]
        qm_core = [0, 1, 2]
        environment = list(set(qm_atoms) - set(qm_core))
        settings = calculation.get_settings()
        # Calculator settings
        settings["system_partitioning"] = "POPULATION_THRESHOLD"
        settings["qmqm_atom_indices"] = [qm_core, environment]
        settings["show_serenity_output"] = True
        settings["static_embedding"] = True
        settings["orbital_threshold"] = 0.6
        settings["cas_systems"] = [0]
        # Tasks settings
        settings["require_charges"] = False
        settings["require_partial_energies"] = False

        # Set QM atoms
        properties = self.manager.get_collection("properties")
        qm_atom_label = "qm_atoms"
        prop = db.VectorProperty.make(qm_atom_label, bond_order_model, np.asarray(qm_atoms), properties)
        structure.add_property(qm_atom_label, prop.id())
        prop.set_structure(structure.id())

        calculation.set_settings(settings)
        return calculation, structure

    def check_output(self, calculation, structure, precision: float = 1e-6):
        import scine_database as db
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("core_coefficients_0")
        assert structure.has_property("h_core_0")
        assert structure.has_property("fragment_charges_0")

        # The reference values are taken from a working implementation, 14.01.2025.
        h_core_reference_values_r0 = [-1.94922417e+01, -6.21423830e+00, -2.93789197e+00]
        h_core_reference_values_r1 = [-6.21423830e+00, -6.07804852e+00, -3.90078256e+00]
        n_mos = 91
        results = calculation.get_results()
        assert len(results.property_ids) == 7

        properties = self.manager.get_collection("properties")
        mo_coefficients = db.DenseMatrixProperty(structure.get_property("core_coefficients_0"), properties).get_data()
        assert mo_coefficients.shape == (n_mos, n_mos)

        h_core = db.DenseMatrixProperty(structure.get_property("h_core_0"), properties).get_data()
        assert h_core.shape == (2 * n_mos, n_mos)
        for i, ref in enumerate(h_core_reference_values_r0):
            assert abs(h_core[0, i] - ref) < precision
        for i, ref in enumerate(h_core_reference_values_r1):
            assert abs(h_core[1, i] - ref) < precision

    def run_embedding_job(self, calculation):
        from scine_puffin.jobs.dft_embedded_hamiltonian import DftEmbeddedHamiltonian
        config = self.get_configuration()
        job = DftEmbeddedHamiltonian()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

    @skip_without('database', 'serenity', 'utilities', 'readuct')
    def test_acetone_radical(self):
        calculation, structure = self.setup_acetone()
        self.run_embedding_job(calculation)
        self.check_output(calculation, structure)

    @skip_without('database', 'serenity', 'utilities', 'readuct', 'turbomole')
    def test_turbomole_supersystem_scf(self):
        # Setup DB for calculation
        calculation, structure = self.setup_acetone()
        settings = calculation.get_settings()
        settings["use_turbomole_mos"] = True
        calculation.set_settings(settings)
        self.run_embedding_job(calculation)
        self.check_output(calculation, structure, 1e-4)

    @skip_without('database', 'serenity', 'utilities', 'readuct')
    def test_qmmm_acetone_radical(self):
        calculation, structure = self.setup_qmmm_acetone()
        self.run_embedding_job(calculation)
        self.check_output(calculation, structure)
