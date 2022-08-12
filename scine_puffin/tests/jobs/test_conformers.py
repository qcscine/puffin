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


class ConformersJobTest(JobTestCase):

    @skip_without('database', 'molassembler')
    def test_butane(self):
        from scine_puffin.jobs.conformers import Conformers
        import scine_database as db
        import scine_utilities as utils

        properties = self.manager.get_collection("properties")
        structures = self.manager.get_collection("structures")

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", "fakingagraph")
        model = db.Model('FAKE', '', '')
        bos = utils.io.read(butane)[1]
        bo_prop = db.SparseMatrixProperty.make("bond_orders", model, bos.matrix, properties)
        bo_prop.set_structure(structure.id())
        structure.add_property("bond_orders", bo_prop.id())

        job = db.Job('conformers')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Test successful run
        config = self.get_configuration()
        job = Conformers()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE
        # Check results
        results = calculation.get_results()
        conf_struct_ids = results.get_structures()
        # 4 butane conformers - 1 start structure
        assert len(conf_struct_ids) == 3
        for struct_id in conf_struct_ids:
            struct = db.Structure(struct_id)
            struct.link(structures)
            assert struct.has_graph("masm_cbor_graph")
            assert struct.get_graph("masm_cbor_graph") == "fakingagraph"
            assert struct.has_graph("masm_decision_list")
            assert struct.get_label() == db.Label.MINIMUM_GUESS
            assert len(struct.get_graph("masm_decision_list").split()) == 4

    @skip_without('database', 'molassembler')
    def test_missing_bond_orders(self):
        # import Job
        from scine_puffin.jobs.conformers import Conformers
        import scine_database as db

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)
        model = db.Model('FAKE', '', '')
        job = db.Job('conformers')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Test missing bond orders
        config = self.get_configuration()
        job = Conformers()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert not job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.FAILED
        assert calculation.has_comment()

    @skip_without('database', 'molassembler')
    def test_different_bond_orders(self):
        from scine_puffin.jobs.conformers import Conformers
        import scine_database as db
        import scine_utilities as utils

        properties = self.manager.get_collection("properties")

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", "fakingagraph")
        model = db.Model('FAKE', '', '')
        bos = utils.io.read(butane)[1]
        bo_prop = db.SparseMatrixProperty.make("bond_orders", model, bos.matrix, properties)
        bo_prop.set_structure(structure.id())
        structure.add_property("bond_orders", bo_prop.id())

        # Assert fail due to different BO model
        model2 = db.Model('FAKE2', '', '')
        job = db.Job('conformers')
        calculation = add_calculation(self.manager, model2, job, [structure.id()])

        config = self.get_configuration()
        job = Conformers()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert not job.run(self.manager, calculation, config)
        assert calculation.has_comment()
        assert calculation.get_status() == db.Status.FAILED

        # Assert success if different model allowed
        job = db.Job('conformers')
        settings = utils.ValueCollection({"enforce_bond_order_model": False})
        calculation = add_calculation(self.manager, model2, job, [structure.id()], settings)

        config = self.get_configuration()
        job = Conformers()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE

    @skip_without('database', 'molassembler')
    def test_add_based_on_distance_connectivity(self):
        from scine_puffin.jobs.conformers import Conformers
        import scine_database as db
        import scine_utilities as utils

        properties = self.manager.get_collection("properties")

        # Setup DB for calculation
        butane = os.path.join(resource_path(), "butane.mol")
        structure = add_structure(self.manager, butane, db.Label.USER_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", "fakingagraph")
        model = db.Model('FAKE', '', '')
        bos = utils.io.read(butane)[1]
        # Remove one C-C bond
        bos.set_order(1, 0, 0.0)
        bo_prop = db.SparseMatrixProperty.make("bond_orders", model, bos.matrix, properties)
        bo_prop.set_structure(structure.id())
        structure.add_property("bond_orders", bo_prop.id())

        # Assert fail due to disconnected structures
        job = db.Job('conformers')
        settings = utils.ValueCollection({"add_based_on_distance_connectivity": False})
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        config = self.get_configuration()
        job = Conformers()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert not job.run(self.manager, calculation, config)
        assert calculation.has_comment()
        assert calculation.get_status() == db.Status.FAILED

        # Assert success if distance connectivity added
        job = db.Job('conformers')
        settings = utils.ValueCollection({"add_based_on_distance_connectivity": True})
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        config = self.get_configuration()
        job = Conformers()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE
