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


class ScineAfirOptimizationJobTest(JobTestCase):

    def run_by_label(self, input_label, expected_label):
        # import Job
        from scine_puffin.jobs.scine_afir import ScineAfir
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        # Setup DB for calculation
        rc = os.path.join(resource_path(), "proline_acid_propanal_complex.xyz")
        structure = add_structure(self.manager, rc, input_label)
        model = db.Model('dftb3', 'dftb3', '')
        if input_label == db.Label.SURFACE_GUESS:
            p = db.VectorProperty.make("surface_atom_indices", model, [], self.manager.get_collection('properties'))
            structure.set_property("surface_atom_indices", p.id())
        job = db.Job('scine_afir')
        settings = {
            "afir_afir_lhs_list": [3, 16],
            "afir_afir_rhs_list": [17, 20],
            "afir_convergence_max_iterations": 800
        }
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineAfir()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.structure_ids) == 1
        structures = self.manager.get_collection("structures")
        product = db.Structure(results.structure_ids[0], structures)
        assert structures.count("{}") == 2
        assert expected_label == product.get_label()
        assert product.has_property("electronic_energy")
        energy_props = product.get_properties("electronic_energy")
        assert len(energy_props) == 1
        assert len(results.property_ids) == 2
        assert energy_props[0] == results.property_ids[0]
        energy = db.NumberProperty(energy_props[0])
        properties = self.manager.get_collection("properties")
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -31.633858301419195, delta=1e-1)
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        bonds = utils.BondOrderCollection()
        bonds.matrix = db.SparseMatrixProperty(product.get_property('bond_orders'), properties).get_data()
        job.connectivity_settings['sub_based_on_distance_connectivity'] = False
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1

    @skip_without('database', 'readuct', 'molassembler')
    def test_user_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.USER_GUESS, db.Label.USER_OPTIMIZED)

    @skip_without('database', 'readuct', 'molassembler')
    def test_minimum_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.MINIMUM_GUESS, db.Label.MINIMUM_OPTIMIZED)

    @skip_without('database', 'molassembler', 'readuct')
    def test_surface_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.SURFACE_GUESS, db.Label.SURFACE_OPTIMIZED)
