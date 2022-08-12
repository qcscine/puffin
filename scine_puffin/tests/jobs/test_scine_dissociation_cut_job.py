#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os
import json
from typing import List

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_structure
)

from ..resources import resource_path


class ScineReactDissociationCutJobTest(JobTestCase):

    def _g_help(self, sid) -> str:
        import scine_database as db
        structures = self.manager.get_collection("structures")
        return db.Structure(sid, structures).get_graph('masm_cbor_graph')

    def _e_help(self, sid) -> float:
        import scine_database as db
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        return db.NumberProperty(db.Structure(sid, structures).get_properties('electronic_energy')[-1],
                                 properties).get_data()

    def _setup_and_execute(self, dissociations: List[int], charge_propensity_check: int,
                           charge: int = 0, multiplicity: int = 1):
        from scine_puffin.jobs.scine_dissociation_cut import ScineDissociationCut
        import scine_database as db
        reactant_path = os.path.join(resource_path(), "butane.mol")
        reactant_guess = add_structure(self.manager, reactant_path, db.Label.MINIMUM_OPTIMIZED,
                                       charge=charge, multiplicity=multiplicity)
        graph = json.load(open(os.path.join(resource_path(), "butane.json"), "r"))
        for key, value in graph.items():
            reactant_guess.set_graph(key, value)

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_dissociation_cut')
        settings = {
            "dissociations": dissociations,
            "charge_propensity_check": charge_propensity_check,
            "max_scf_iterations": 1000,
        }

        calculation = add_calculation(self.manager, model, job, [reactant_guess.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineDissociationCut()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        return calculation

    @skip_without('database', 'readuct', 'molassembler')
    def test_single_butane_cut(self):
        import scine_database as db
        import scine_utilities as utils
        import scine_molassembler as masm

        calculation = self._setup_and_execute([1, 2], 1)

        # Check results
        structures = self.manager.get_collection("structures")
        elementary_steps = self.manager.get_collection("elementary_steps")

        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        expected_structures = 6  # -1 0 +1 charge for two structures from single split
        assert len(results.structure_ids) == expected_structures
        # bo + energy for products + reactant + lowest dissociated structures property
        assert len(results.property_ids) == (expected_structures + 1) * 2 + 1
        assert len(results.elementary_step_ids) == 1

        step = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert step.get_type() == db.ElementaryStepType.BARRIERLESS

        reactants = step.get_reactants(db.Side.BOTH)
        assert len(reactants[0]) == 1
        assert len(reactants[1]) == 2
        for product in reactants[1]:
            assert db.Structure(product, structures).get_label() == db.Label.MINIMUM_OPTIMIZED
            assert db.Structure(product, structures).has_property('bond_orders')
            assert db.Structure(product, structures).has_property('electronic_energy')
            assert db.Structure(product, structures).has_graph('masm_cbor_graph')

        assert masm.JsonSerialization.equal_molecules(self._g_help(reactants[1][0]), self._g_help(reactants[1][1]))
        assert masm.JsonSerialization.equal_molecules(
            self._g_help(
                reactants[0][0]), self._g_help(
                calculation.get_structures()[0]))

        reaction_energy = (sum(self._e_help(p)
                           for p in reactants[1]) - self._e_help(reactants[0][0])) * utils.KJPERMOL_PER_HARTREE
        self.assertAlmostEqual(reaction_energy, 426.1682851069426, delta=1)

    @skip_without('database', 'readuct', 'molassembler')
    def test_double_butane_cut(self):
        import scine_database as db
        import scine_utilities as utils
        import scine_molassembler as masm

        calculation = self._setup_and_execute([0, 1, 1, 2], 1)

        # Check results
        structures = self.manager.get_collection("structures")
        elementary_steps = self.manager.get_collection("elementary_steps")

        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        expected_structures = 9  # -1 0 +1 charge for three structures from double split
        assert len(results.structure_ids) == expected_structures
        # bo + energy for products + reactant + lowest dissociated structures property
        assert len(results.property_ids) == (expected_structures + 1) * 2 + 1
        assert len(results.elementary_step_ids) == 1

        step = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert step.get_type() == db.ElementaryStepType.BARRIERLESS

        reactants = step.get_reactants(db.Side.BOTH)
        assert len(reactants[0]) == 1
        assert len(reactants[1]) == 3
        for product in reactants[1]:
            assert db.Structure(product, structures).get_label() == db.Label.MINIMUM_OPTIMIZED
            assert db.Structure(product, structures).has_property('bond_orders')
            assert db.Structure(product, structures).has_property('electronic_energy')
            assert db.Structure(product, structures).has_graph('masm_cbor_graph')

        assert masm.JsonSerialization.equal_molecules(
            self._g_help(
                reactants[0][0]), self._g_help(
                calculation.get_structures()[0]))
        assert not masm.JsonSerialization.equal_molecules(self._g_help(reactants[1][0]), self._g_help(reactants[1][1]))
        assert not masm.JsonSerialization.equal_molecules(self._g_help(reactants[1][0]), self._g_help(reactants[1][2]))
        assert not masm.JsonSerialization.equal_molecules(self._g_help(reactants[1][1]), self._g_help(reactants[1][2]))

        reaction_energy = (sum(self._e_help(p)
                           for p in reactants[1]) - self._e_help(reactants[0][0])) * utils.KJPERMOL_PER_HARTREE
        self.assertAlmostEqual(reaction_energy, 895.4669538980517, delta=1)

    @skip_without('database', 'readuct', 'molassembler')
    def test_single_butane_charged_cut(self):
        import scine_database as db
        import scine_utilities as utils
        import scine_molassembler as masm

        calculation = self._setup_and_execute([1, 2], 1, charge=1, multiplicity=2)

        # Check results
        structures = self.manager.get_collection("structures")
        elementary_steps = self.manager.get_collection("elementary_steps")

        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        expected_structures = 6  # -1 0 +1 charge for two structures from single split
        assert len(results.structure_ids) == expected_structures
        # bo + energy for products + reactant + lowest dissociated structures property
        assert len(results.property_ids) == (expected_structures + 1) * 2 + 1
        assert len(results.elementary_step_ids) == 1

        step = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert step.get_type() == db.ElementaryStepType.BARRIERLESS

        reactants = step.get_reactants(db.Side.BOTH)
        assert len(reactants[0]) == 1
        assert len(reactants[1]) == 2
        for product in reactants[1]:
            assert db.Structure(product, structures).get_label() == db.Label.MINIMUM_OPTIMIZED
            assert db.Structure(product, structures).has_property('bond_orders')
            assert db.Structure(product, structures).has_property('electronic_energy')
            assert db.Structure(product, structures).has_graph('masm_cbor_graph')

        assert masm.JsonSerialization.equal_molecules(self._g_help(reactants[1][0]), self._g_help(reactants[1][1]))
        assert masm.JsonSerialization.equal_molecules(
            self._g_help(
                reactants[0][0]), self._g_help(
                calculation.get_structures()[0]))

        reaction_energy = (sum(self._e_help(p)
                           for p in reactants[1]) - self._e_help(reactants[0][0])) * utils.KJPERMOL_PER_HARTREE
        self.assertAlmostEqual(reaction_energy, 255.10683618838843, delta=1)
