#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os
import json

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_compound_and_structure,
    add_structure
)

from ..resources import resource_path


class ScineStepRefinementJobTest(JobTestCase):

    @skip_without('database', 'readuct', 'molassembler')
    def test_energy_starting_from_separated_reactants(self):
        # import Job
        from scine_puffin.jobs.scine_step_refinement import ScineStepRefinement
        import scine_database as db

        # Amine-addition of the proline derivative to the aldehyde group of propanal.
        # These structures where originally optimized with dftb3
        ts_guess_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
        ts_guess = add_structure(self.manager, ts_guess_path, db.Label.TS_OPTIMIZED)
        compound_one = add_compound_and_structure(self.manager, reactant_one_path)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path)
        reactant_one_guess = db.Structure(compound_one.get_centroid())
        reactant_two_guess = db.Structure(compound_two.get_centroid())
        reactant_one_guess.link(self.manager.get_collection('structures'))
        reactant_two_guess.link(self.manager.get_collection('structures'))
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_step_refinement')
        settings = {
            "nt_convergence_max_iterations": 600,
            "nt_nt_total_force_norm": 0.1,
            "nt_sd_factor": 1.0,
            "nt_nt_use_micro_cycles": True,
            "nt_nt_fixed_number_of_micro_cycles": True,
            "nt_nt_number_of_micro_cycles": 10,
            "nt_nt_filter_passes": 10,
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.2,
            "tsopt_bofill_follow_mode": 0,
            "irc_convergence_max_iterations": 75,
            "irc_sd_factor": 2.0,
            "irc_irc_initial_step_size": 0.3,
            "irc_stop_on_error": False,
            "irc_convergence_step_max_coefficient": 0.002,
            "irc_convergence_step_rms": 0.001,
            "irc_convergence_gradient_max_coefficient": 0.0002,
            "irc_convergence_gradient_rms": 0.0001,
            "irc_convergence_delta_value": 1e-06,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 800,
            "ircopt_convergence_step_max_coefficient": 0.002,
            "ircopt_convergence_step_rms": 0.001,
            "ircopt_convergence_gradient_max_coefficient": 0.0002,
            "ircopt_convergence_gradient_rms": 0.0001,
            "ircopt_convergence_requirement": 3,
            "ircopt_convergence_delta_value": 1e-06,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.2,
            "opt_convergence_max_iterations": 800,
            "opt_convergence_step_max_coefficient": 0.002,
            "opt_convergence_step_rms": 0.001,
            "opt_convergence_gradient_max_coefficient": 0.0002,
            "opt_convergence_gradient_rms": 0.0001,
            "opt_convergence_requirement": 3,
            "opt_convergence_delta_value": 1e-06,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_use_trust_radius": True,
            "opt_bfgs_trust_radius": 0.4,
            "imaginary_wavenumber_threshold": -30,
            "nt_nt_associations": [
                3,
                17,
                16,
                20
            ],
            "nt_nt_dissociations": [],
            "rc_x_alignment_0": [
                -0.405162890256947,
                0.382026105406949,
                0.830601641670804,
                -0.382026105406949,
                0.754648889886101,
                -0.533442693999341,
                -0.830601641670804,
                -0.533442693999341,
                -0.159811780143048
            ],
            "rc_x_alignment_1": [
                0.881555658365378,
                -0.439248859713409,
                -0.172974161204658,
                -0.311954059087669,
                -0.817035288655183,
                0.484910303160151,
                -0.354322291456116,
                -0.373515429845424,
                -0.857287546535394
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 3.48715302740618,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False
        }

        calculation = add_calculation(self.manager, model, job, [reactant_one_guess.id(), reactant_two_guess.id()],
                                      settings)
        auxiliaries = {
            "transition-state-id": ts_guess.id()
        }
        calculation.set_auxiliaries(auxiliaries)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineStepRefinement()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        assert structures.count(json.dumps({})) == 8
        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 1
        complex_structure = structures.get_one_structure(json.dumps(selection))
        complex_structure.link(structures)
        selection = {"label": "ts_optimized"}
        assert structures.count(json.dumps(selection)) == 2
        assert properties.count(json.dumps({})) == 14
        assert elementary_steps.count(json.dumps({})) == 2
        results = calculation.get_results()
        assert len(results.property_ids) == 10
        assert len(results.structure_ids) == 5
        assert len(results.elementary_step_ids) == 2
        # The regular elementary step should be the last one in the list.
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)

        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert new_elementary_step.get_type() == db.ElementaryStepType.BARRIERLESS
        lhs_rhs_structures = new_elementary_step.get_reactants(db.Side.BOTH)
        assert lhs_rhs_structures[1][0] == complex_structure.id()
        assert len(lhs_rhs_structures[0]) == 2

        reactants = new_elementary_step.get_reactants(db.Side.BOTH)
        lhs_energies = [-20.9685007365807, -10.70501645756783]
        rhs_energies = [-31.6786171872375]
        structure_ids = reactants[0] + reactants[1]
        for s_id in structure_ids:
            structure = db.Structure(s_id, structures)
            assert structure.get_model() == model
            assert structure.has_property('electronic_energy')
            assert structure.has_property('bond_orders')
            energy_props = structure.get_properties("electronic_energy")
            energy = db.NumberProperty(energy_props[0], properties).get_data()
            # The ordering of the energies may change.
            if energy > -11.0:
                self.assertAlmostEqual(energy, lhs_energies[1], delta=1e-1)
            elif energy > -22.0:
                self.assertAlmostEqual(energy, lhs_energies[0], delta=1e-1)
            else:
                self.assertAlmostEqual(energy, rhs_energies[0], delta=1e-1)

    @skip_without('database', 'readuct', 'molassembler')
    def test_energy_starting_from_complex(self):
        # import Job
        from scine_puffin.jobs.scine_step_refinement import ScineStepRefinement
        import scine_database as db

        # Amine-addition of the proline derivative to the aldehyde group of propanal.
        # These structures where originally optimized with dftb3
        ts_guess_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        reactant_one_path = os.path.join(resource_path(), "proline_propanal_complex.xyz")
        ts_guess = add_structure(self.manager, ts_guess_path, db.Label.TS_OPTIMIZED)
        reactant_one_guess = add_structure(self.manager, reactant_one_path, db.Label.COMPLEX_OPTIMIZED)
        graph_one = json.load(open(os.path.join(resource_path(), "proline_propanal_complex.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_step_refinement')
        settings = {
            "nt_convergence_max_iterations": 600,
            "nt_nt_total_force_norm": 0.1,
            "nt_sd_factor": 1.0,
            "nt_nt_use_micro_cycles": True,
            "nt_nt_fixed_number_of_micro_cycles": True,
            "nt_nt_number_of_micro_cycles": 10,
            "nt_nt_filter_passes": 10,
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.2,
            "tsopt_bofill_follow_mode": 0,
            "irc_convergence_max_iterations": 75,
            "irc_sd_factor": 2.0,
            "irc_irc_initial_step_size": 0.3,
            "irc_stop_on_error": False,
            "irc_convergence_step_max_coefficient": 0.002,
            "irc_convergence_step_rms": 0.001,
            "irc_convergence_gradient_max_coefficient": 0.0002,
            "irc_convergence_gradient_rms": 0.0001,
            "irc_convergence_delta_value": 1e-06,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 800,
            "ircopt_convergence_step_max_coefficient": 0.002,
            "ircopt_convergence_step_rms": 0.001,
            "ircopt_convergence_gradient_max_coefficient": 0.0002,
            "ircopt_convergence_gradient_rms": 0.0001,
            "ircopt_convergence_requirement": 3,
            "ircopt_convergence_delta_value": 1e-06,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.2,
            "opt_convergence_max_iterations": 800,
            "opt_convergence_step_max_coefficient": 0.002,
            "opt_convergence_step_rms": 0.001,
            "opt_convergence_gradient_max_coefficient": 0.0002,
            "opt_convergence_gradient_rms": 0.0001,
            "opt_convergence_requirement": 3,
            "opt_convergence_delta_value": 1e-06,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_use_trust_radius": True,
            "opt_bfgs_trust_radius": 0.4,
            "imaginary_wavenumber_threshold": -30,
            "nt_nt_associations": [
                3,
                17,
                16,
                20
            ],
            "nt_nt_dissociations": [],
            "rc_x_alignment_0": [
                -0.405162890256947,
                0.382026105406949,
                0.830601641670804,
                -0.382026105406949,
                0.754648889886101,
                -0.533442693999341,
                -0.830601641670804,
                -0.533442693999341,
                -0.159811780143048
            ],
            "rc_x_alignment_1": [
                0.881555658365378,
                -0.439248859713409,
                -0.172974161204658,
                -0.311954059087669,
                -0.817035288655183,
                0.484910303160151,
                -0.354322291456116,
                -0.373515429845424,
                -0.857287546535394
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 3.48715302740618,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False
        }

        calculation = add_calculation(self.manager, model, job,
                                      [reactant_one_guess.id()],
                                      settings)
        auxiliaries = {
            "transition-state-id": ts_guess.id()
        }
        calculation.set_auxiliaries(auxiliaries)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineStepRefinement()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        assert structures.count(json.dumps({})) == 7
        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 2
        complex_structure = structures.get_one_structure(json.dumps(selection))
        complex_structure.link(structures)
        selection = {"label": "ts_optimized"}
        assert structures.count(json.dumps(selection)) == 2
        assert properties.count(json.dumps({})) == 14
        assert elementary_steps.count(json.dumps({})) == 2
        results = calculation.get_results()
        assert len(results.property_ids) == 10
        assert len(results.structure_ids) == 5
        assert len(results.elementary_step_ids) == 2
        # The regular elementary step should be the last one in the list.
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)

        reactants = new_elementary_step.get_reactants(db.Side.BOTH)
        lhs_energies = [-31.6823167498272]
        rhs_energies = [-31.6786171872375]
        all_energies = lhs_energies + rhs_energies
        structure_ids = reactants[0] + reactants[1]
        for i, s_id in enumerate(structure_ids):
            structure = db.Structure(s_id, structures)
            assert structure.get_model() == model
            assert structure.has_property('electronic_energy')
            assert structure.has_property('bond_orders')
            energy_props = structure.get_properties("electronic_energy")
            energy = db.NumberProperty(energy_props[0], properties)
            self.assertAlmostEqual(energy.get_data(), all_energies[i], delta=1e-1)
