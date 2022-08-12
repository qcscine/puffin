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


class GraphJobTest(JobTestCase):

    @skip_without('database', 'molassembler')
    def test_butane(self):
        from scine_puffin.jobs.graph import Graph
        import scine_database as db
        import scine_utilities as utils

        properties = self.manager.get_collection("properties")

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)

        model = db.Model('FAKE', '', '')
        bos = utils.io.read(butane)[1]
        bo_prop = db.SparseMatrixProperty.make("bond_orders", model, bos.matrix, properties)
        bo_prop.set_structure(structure.id())
        structure.add_property("bond_orders", bo_prop.id())

        job = db.Job('graph')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Test successful run
        config = self.get_configuration()
        job = Graph()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE
        # Check whether structure has graph
        assert structure.has_graph("masm_cbor_graph")
        assert len(structure.get_graph("masm_cbor_graph").split(";")) == 1
        assert structure.has_graph("masm_decision_list")
        assert structure.has_graph("masm_idx_map")

    @skip_without('database', 'molassembler')
    def test_missing_bond_orders(self):
        from scine_puffin.jobs.graph import Graph
        import scine_database as db

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)

        model = db.Model('FAKE', '', '')
        job = db.Job('graph')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Test failed run
        config = self.get_configuration()
        job = Graph()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert not job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.FAILED
        assert calculation.has_comment()

    @skip_without('database', 'molassembler')
    def test_add_based_on_distance_connectivity(self):
        from scine_puffin.jobs.graph import Graph
        import scine_database as db
        import scine_utilities as utils

        properties = self.manager.get_collection("properties")

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)

        model = db.Model('FAKE', '', '')
        bos = utils.io.read(butane)[1]
        # Remove one C-C bond
        bos.set_order(1, 0, 0.0)
        bo_prop = db.SparseMatrixProperty.make("bond_orders", model, bos.matrix, properties)
        bo_prop.set_structure(structure.id())
        structure.add_property("bond_orders", bo_prop.id())

        # Test run with bond readded due to distance criterion
        job = db.Job('graph')
        calculation = add_calculation(self.manager, model, job, [structure.id()])
        config = self.get_configuration()
        job = Graph()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE
        # Check whether structure has graph
        assert structure.has_graph("masm_cbor_graph")
        assert len(structure.get_graph("masm_cbor_graph").split(";")) == 1
        assert structure.has_graph("masm_decision_list")
        assert structure.has_graph("masm_idx_map")

        # Test run without bond being readded, i.e. with molecule split in two parts
        # Graph should be replaced if rerun on same structure
        job = db.Job('graph')
        settings = utils.ValueCollection({"add_based_on_distance_connectivity": False})
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)
        config = self.get_configuration()
        job = Graph()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE
        # Check whether structure has graph
        assert structure.has_graph("masm_cbor_graph")
        assert len(structure.get_graph("masm_cbor_graph").split(";")) == 2
        assert structure.has_graph("masm_decision_list")
        assert structure.has_graph("masm_idx_map")
