#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import json
from typing import List, Optional

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
                           charge: int = 0, multiplicity: int = 1, with_opt: bool = False,
                           system_name: str = "butane", additional_settings: Optional[dict] = None,
                           model=None):
        from scine_puffin.jobs.scine_dissociation_cut import ScineDissociationCut
        from scine_puffin.jobs.scine_dissociation_cut_with_optimization import ScineDissociationCutWithOptimization
        import scine_database as db

        reactant_path = os.path.join(resource_path(), f"{system_name}.mol")
        if not os.path.exists(reactant_path):
            reactant_path = os.path.join(resource_path(), f"{system_name}.xyz")
            if not os.path.exists(reactant_path):
                raise FileNotFoundError(f"Could not find {system_name}.mol or {system_name}.xyz in {resource_path()}")
        reactant_guess = add_structure(self.manager, reactant_path, db.Label.MINIMUM_OPTIMIZED, model=model,
                                       charge=charge, multiplicity=multiplicity)
        graph = json.load(open(os.path.join(resource_path(), f"{system_name}.json"), "r"))
        for key, value in graph.items():
            reactant_guess.set_graph(key, value)

        if model is None:
            model = db.Model('dftb3', 'dftb3', '') if not with_opt else db.Model('pm6', 'pm6', '')
        db_job = db.Job('scine_dissociation_cut') if not with_opt \
            else db.Job('scine_dissociation_cut_with_optimization')
        settings = {
            "dissociations": dissociations,
            "charge_propensity_check": charge_propensity_check,
            "max_scf_iterations": 1000,
        }
        if additional_settings is not None:
            settings.update(additional_settings)

        calculation = add_calculation(self.manager, model, db_job, [reactant_guess.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineDissociationCut() if not with_opt else ScineDissociationCutWithOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        return calculation

    @skip_without('database', 'readuct', 'molassembler')
    def test_single_butane_cut(self):
        import scine_database as db
        import scine_utilities as utils
        import scine_molassembler as masm

        for with_opt in [False, True]:
            calculation = self._setup_and_execute([1, 2], 1, with_opt=with_opt)

            # Check results
            structures = self.manager.get_collection("structures")
            elementary_steps = self.manager.get_collection("elementary_steps")

            assert calculation.get_status() == db.Status.COMPLETE
            results = calculation.get_results()
            expected_structures = 6 + int(with_opt)  # -1 0 +1 charge for two structures from single split
            assert len(results.structure_ids) == expected_structures
            # bo + energy for products + reactant + lowest dissociated structures property
            assert len(results.property_ids) == (expected_structures - int(with_opt) + 1) * 2 + 1
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
            assert with_opt or masm.JsonSerialization.equal_molecules(
                self._g_help(
                    reactants[0][0]), self._g_help(
                    calculation.get_structures()[0]))

            reaction_energy = (sum(self._e_help(p)
                               for p in reactants[1]) - self._e_help(reactants[0][0])) * utils.KJPERMOL_PER_HARTREE
            ref = 240.8673484190177 if with_opt else 426.1682851069426
            self.assertAlmostEqual(reaction_energy, ref, delta=1)

    @skip_without('database', 'readuct', 'molassembler')
    def test_double_butane_cut(self):
        import scine_database as db
        import scine_utilities as utils
        import scine_molassembler as masm

        for with_opt in [False, True]:
            calculation = self._setup_and_execute([0, 1, 1, 2], 1, with_opt=with_opt)

            # Check results
            structures = self.manager.get_collection("structures")
            elementary_steps = self.manager.get_collection("elementary_steps")

            assert calculation.get_status() == db.Status.COMPLETE
            results = calculation.get_results()
            expected_structures = 9 + int(with_opt)  # -1 0 +1 charge for three structures from double split
            assert len(results.structure_ids) == expected_structures
            # bo + energy for products + reactant + lowest dissociated structures property
            assert len(results.property_ids) == (expected_structures - int(with_opt) + 1) * 2 + 1
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

            # two doublet and one triplet
            assert sum([db.Structure(product, structures).get_multiplicity() == 2 for product in reactants[1]]) == 2
            if with_opt:
                assert sum([db.Structure(product, structures).get_multiplicity() == 3 for product in reactants[1]]) == 1
                assert sum([db.Structure(product, structures).get_multiplicity() == 1 for product in reactants[1]]) == 0
            else:
                assert sum([db.Structure(product, structures).get_multiplicity() == 3 for product in reactants[1]]) == 0
                assert sum([db.Structure(product, structures).get_multiplicity() == 1 for product in reactants[1]]) == 1

            assert with_opt or masm.JsonSerialization.equal_molecules(
                self._g_help(
                    reactants[0][0]), self._g_help(
                    calculation.get_structures()[0]))
            assert not masm.JsonSerialization.equal_molecules(
                self._g_help(reactants[1][0]), self._g_help(reactants[1][1]))
            assert not masm.JsonSerialization.equal_molecules(
                self._g_help(reactants[1][0]), self._g_help(reactants[1][2]))
            assert not masm.JsonSerialization.equal_molecules(
                self._g_help(reactants[1][1]), self._g_help(reactants[1][2]))

            reaction_energy = (sum(self._e_help(p)
                               for p in reactants[1]) - self._e_help(reactants[0][0])) * utils.KJPERMOL_PER_HARTREE
            ref = 602.2781779325359 if with_opt else 895.4669538980517
            self.assertAlmostEqual(reaction_energy, ref, delta=1)

    @skip_without('database', 'readuct', 'molassembler')
    def test_single_butane_charged_cut(self):
        import scine_database as db
        import scine_utilities as utils
        import scine_molassembler as masm

        charge = 1
        calculation = self._setup_and_execute([1, 2], 1, charge=charge, multiplicity=2)

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
        assert db.Structure(reactants[0][0], structures).get_charge() == charge
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

    @skip_without('database', 'readuct', 'molassembler')
    def test_single_co2_cut(self):
        import scine_database as db
        import scine_utilities as utils

        calculation = self._setup_and_execute([1, 2], 1, with_opt=False,
                                              system_name="co2")

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
        assert any(len(db.Structure(product, structures).get_atoms()) == 1 for product in reactants[1])
        assert any(db.Structure(product, structures).get_multiplicity() == 3 for product in reactants[1])

        reaction_energy = (sum(self._e_help(p)
                               for p in reactants[1]) - self._e_help(reactants[0][0])) * utils.KJPERMOL_PER_HARTREE
        ref = 542.5066448245391
        self.assertAlmostEqual(reaction_energy, ref, delta=1)

    @skip_without('database', 'readuct', 'molassembler')
    def test_with_additional_nt_run(self):
        import scine_database as db
        import scine_molassembler as masm

        success_result = json.load(open(os.path.join(resource_path(), "mn_result.json"), "r"))
        success_result_2 = json.load(open(os.path.join(resource_path(), "mn_result_2.json"), "r"))
        model = db.Model("pm6", "pm6", "")
        model.program = "sparrow"
        model.spin_mode = "unrestricted"
        for energy_limit, expected_success in zip([0.0, 5.0, 100.0], [False, False, True]):
            settings = {
                "additional_nt_run_dissociation_energy_limit": energy_limit,
                "imaginary_wavenumber_threshold": -25.0,
                "max_scf_iterations": 1000,
                "nt_convergence_max_iterations": 600,
                "nt_nt_filter_passes": 10,
                "nt_nt_fixed_number_of_micro_cycles": True,
                "nt_nt_number_of_micro_cycles": 10,
                "nt_nt_total_force_norm": 0.1,
                "nt_nt_use_micro_cycles": True,
                "nt_sd_factor": 1.0,
                "tsopt_dimer_trust_radius": 0.2,
                "tsopt_dimer_calculate_hessian_once": True,
                "tsopt_convergence_delta_value": 1.0e-06,
                "tsopt_convergence_gradient_max_coefficient": 0.0002,
                "tsopt_convergence_gradient_rms": 0.0001,
                "tsopt_convergence_max_iterations": 1000,
                "tsopt_convergence_requirement": 3,
                "tsopt_convergence_step_max_coefficient": 0.002,
                "tsopt_convergence_step_rms": 0.001,
                "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
                "tsopt_optimizer": "dimer",
                "irc_convergence_delta_value": 1.0e-06,
                "irc_convergence_gradient_max_coefficient": 0.0002,
                "irc_convergence_gradient_rms": 0.0001,
                "irc_convergence_max_iterations": 100,
                "irc_convergence_step_max_coefficient": 0.002,
                "irc_convergence_step_rms": 0.001,
                "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
                "irc_irc_initial_step_size": 0.3,
                "irc_sd_factor": 2.0,
                "ircopt_bfgs_trust_radius": 0.2,
                "ircopt_bfgs_use_trust_radius": True,
                "ircopt_convergence_delta_value": 1.0e-06,
                "ircopt_convergence_gradient_max_coefficient": 0.0002,
                "ircopt_convergence_gradient_rms": 0.0001,
                "ircopt_convergence_max_iterations": 1000,
                "ircopt_convergence_requirement": 3,
                "ircopt_convergence_step_max_coefficient": 0.002,
                "ircopt_convergence_step_rms": 0.001,
                "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
                "opt_bfgs_trust_radius": 0.2,
                "opt_bfgs_use_trust_radius": True,
                "opt_convergence_delta_value": 1.0e-06,
                "opt_convergence_gradient_max_coefficient": 0.0002,
                "opt_convergence_gradient_rms": 0.0001,
                "opt_convergence_max_iterations": 1000,
                "opt_convergence_requirement": 3,
                "opt_convergence_step_max_coefficient": 0.002,
                "opt_convergence_step_rms": 0.001,
                "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
                "rcopt_bfgs_trust_radius": 0.2,
                "rcopt_bfgs_use_trust_radius": True,
                "rcopt_convergence_delta_value": 1.0e-06,
                "rcopt_convergence_gradient_max_coefficient": 0.0002,
                "rcopt_convergence_gradient_rms": 0.0001,
                "rcopt_convergence_max_iterations": 1000,
                "rcopt_convergence_requirement": 3,
                "rcopt_convergence_step_max_coefficient": 0.002,
                "rcopt_convergence_step_rms": 0.001,
                "rcopt_geoopt_coordinate_system": "cartesianWithoutRotTrans"
            }
            calculation = self._setup_and_execute([1, 9, 9, 23], 1,
                                                  charge=1,
                                                  multiplicity=1,
                                                  with_opt=False,
                                                  system_name="mn_catalyst",
                                                  additional_settings=settings,
                                                  model=model)
            # Check results
            structures = self.manager.get_collection("structures")
            elementary_steps = self.manager.get_collection("elementary_steps")
            assert calculation.get_model() == model
            assert calculation.get_status() == db.Status.COMPLETE
            assert len(calculation.get_results().elementary_step_ids) == int(expected_success)
            if expected_success:
                step = db.ElementaryStep(calculation.get_results().elementary_step_ids[0], elementary_steps)
                assert step.get_type() == db.ElementaryStepType.REGULAR
                reactants, products = step.get_reactants(db.Side.BOTH)

                assert len(reactants) == 1 and len(products) == 2 or len(products) == 1 and len(reactants) == 2
                if len(reactants) == 1:
                    result_structure = db.Structure(reactants[0], structures)
                    # other side contains ethene
                    assert any(len(db.Structure(sid, structures).get_atoms()) == 6 for sid in products)
                else:
                    result_structure = db.Structure(products[0], structures)
                    # other side contains ethene
                    assert any(len(db.Structure(sid, structures).get_atoms()) == 6 for sid in reactants)

                result = {
                    "masm_cbor_graph": result_structure.get_graph("masm_cbor_graph"),
                    "masm_decision_list": result_structure.get_graph("masm_decision_list"),
                    "masm_idx_map": result_structure.get_graph("masm_idx_map"),
                }
                # The unit test is unstable and may converge to slightly different geometries which are resolved to
                # a seesaw or tetrahedron configuration at the Mn center.
                assert any(masm.JsonSerialization.equal_molecules(
                    res["masm_cbor_graph"],
                    result["masm_cbor_graph"]
                ) for res in [success_result, success_result_2])
                assert any(masm.JsonSerialization.equal_decision_lists(
                    res["masm_decision_list"],
                    result["masm_decision_list"]
                ) for res in [success_result, success_result_2])
                assert any(res["masm_idx_map"] == result["masm_idx_map"] for res in [success_result, success_result_2])
