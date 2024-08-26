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


class ScineGeometryOptimizationJobTest(JobTestCase):
    def calculate_graph(self, structure_file):
        from scine_puffin.jobs.graph import Graph
        import scine_utilities as utils
        import scine_database as db

        properties = self.manager.get_collection("properties")

        structure_file_path = os.path.join(resource_path(), structure_file)
        structure = add_structure(self.manager, structure_file_path, db.Label.IRRELEVANT)
        model = db.Model('dftb3', 'dftb3', '')

        bos = utils.io.read(structure_file_path)[1]
        bo_prop = db.SparseMatrixProperty.make("bond_orders", model, bos.matrix, properties)
        bo_prop.set_structure(structure.id())
        structure.add_property("bond_orders", bo_prop.id())

        job = db.Job('graph')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Test successful run
        config = self.get_configuration()
        job = Graph()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE
        graph = structure.get_graph("masm_cbor_graph")

        return graph

    def run_by_label(self, input_label, expected_label, structure_file="water.xyz"):
        import scine_database as db
        # import Job
        from scine_puffin.jobs.scine_geometry_optimization import ScineGeometryOptimization

        # Setup DB for calculation
        structure_file_path = os.path.join(resource_path(), structure_file)
        input_structure = add_structure(self.manager, structure_file_path, input_label)
        model = db.Model('dftb3', 'dftb3', '')
        if input_label == db.Label.SURFACE_GUESS:
            prop = db.VectorProperty.make("surface_atom_indices", model, [0, 1],
                                          self.manager.get_collection("properties"))
            input_structure.add_property("surface_atom_indices", prop.id())
        job = db.Job('scine_geometry_optimization')
        calculation = add_calculation(self.manager, model, job, [input_structure.id()],
                                      {'opt_convergence_max_iterations': 500})

        # Run calculation/job
        config = self.get_configuration()
        job = ScineGeometryOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.structure_ids) == 1
        structure = db.Structure(results.structure_ids[0])
        structures = self.manager.get_collection("structures")
        assert structures.count("{}") >= 2
        structure.link(structures)
        assert expected_label == structure.get_label()
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        assert len(energy_props) == 1
        assert len(results.property_ids) == 1
        assert energy_props[0] == results.property_ids[0]
        energy = db.NumberProperty(energy_props[0])
        properties = self.manager.get_collection("properties")
        energy.link(properties)
        if structure_file == "water.xyz":
            self.assertAlmostEqual(energy.get_data(), -4.071575644, delta=1e-1)
        return input_structure, structure

    @skip_without('database', 'readuct')
    def test_user_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.USER_GUESS, db.Label.USER_OPTIMIZED)

    @skip_without('database', 'readuct')
    def test_minimum_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.MINIMUM_GUESS, db.Label.MINIMUM_OPTIMIZED)

    @skip_without('database', 'readuct')
    def test_surface_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.SURFACE_GUESS, db.Label.SURFACE_OPTIMIZED)

    @skip_without('database', 'readuct')
    def test_user_guess_optimized_to_complex(self):
        import scine_database as db
        graph = self.calculate_graph("c5of8.mol")
        assert ';' not in graph
        input_structure, output_structure = self.run_by_label(db.Label.USER_GUESS, db.Label.USER_COMPLEX_OPTIMIZED,
                                                              "c5of8.mol")
        assert output_structure.has_graph("masm_cbor_graph")
        assert ';' in output_structure.get_graph("masm_cbor_graph")
        assert not input_structure.has_graph("masm_cbor_graph")
