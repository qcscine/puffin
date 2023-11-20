#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
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
    add_structure
)

from ..resources import resource_path


class ScineBsplineOptimizationJobTest(JobTestCase):
    @skip_without('database', 'readuct', 'molassembler')
    def test_energy_starting_from_complex(self):
        # import Job
        from scine_puffin.jobs.scine_bspline_optimization import ScineBsplineOptimization
        import scine_database as db
        self.tearDown()
        self.setUp()

        # Amine-addition of the proline derivative to the aldehyde group of propanal.
        # These structures where originally optimized with dftb3
        reactant_one_path = os.path.join(resource_path(), "proline_acid_propanal_complex.xyz")
        reactant_one_guess = add_structure(self.manager, reactant_one_path, db.Label.COMPLEX_OPTIMIZED)
        product_path = os.path.join(resource_path(), "proline_acid_propanal_product.xyz")
        product_guess = add_structure(self.manager, product_path, db.Label.MINIMUM_OPTIMIZED)

        model = db.Model('dftb3', 'dftb3', 'none')
        job = db.Job('scine_bspline_optimization')
        settings = {
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.05,
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
            "bspline_num_integration_points": 40,
            "spin_propensity_check": 0
        }

        calculation = add_calculation(self.manager, model, job,
                                      [reactant_one_guess.id(), product_guess.id()],
                                      settings)
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid_propanal_complex.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "proline_acid_propanal_product.json"), "r"))
        # add some graphs because we will need them later. It is not important whether they actually match the
        # structure
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        product_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        original_elementary_step = db.ElementaryStep.make([reactant_one_guess.id()], [product_guess.id()],
                                                          self.manager.get_collection("elementary_steps"))
        calculation.set_auxiliaries({"elementary-step-id": original_elementary_step.id()})

        # Run calculation/job
        config = self.get_configuration()
        job = ScineBsplineOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # # Check results
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
        assert structures.count(json.dumps(selection)) == 1
        assert properties.count(json.dumps({})) == 15
        # 1 initial step + 2 steps from the calculation (1 x complex fragmentation, 1 x regular).
        print("N-Steps", elementary_steps.count(json.dumps({})))
        assert elementary_steps.count(json.dumps({})) == 3
        results = calculation.get_results()
        assert len(results.property_ids) == 11
        assert len(results.structure_ids) == 5
        assert len(results.elementary_step_ids) == 2
        # The regular elementary step should be the last one in the list.
        # The first step is the complex fragmentation, second is the regular one, third is the embedding for the
        # LHS.
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[1], elementary_steps)
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-2)

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
            self.assertAlmostEqual(energy.get_data(), all_energies[i], delta=1e-3)

    @skip_without('database', 'readuct', 'molassembler')
    def test_collapsing_spline_ends(self):
        # import Job
        from scine_puffin.jobs.scine_bspline_optimization import ScineBsplineOptimization
        import scine_database as db

        # Amine-addition of the proline derivative to the aldehyde group of propanal.
        # These structures where originally optimized with dftb3
        reactant_one_path = os.path.join(resource_path(), "proline_acid_propanal_complex.xyz")
        reactant_one_guess = add_structure(self.manager, reactant_one_path, db.Label.COMPLEX_OPTIMIZED)
        product_path = os.path.join(resource_path(), "proline_acid_propanal_complex.xyz")
        product_guess = add_structure(self.manager, product_path, db.Label.MINIMUM_OPTIMIZED)

        model = db.Model('dftb3', 'dftb3', 'none')
        job = db.Job('scine_bspline_optimization')
        settings = {
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.05,
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
            "bspline_num_integration_points": 40,
            "sp_expect_charge_separation": True,
        }

        calculation = add_calculation(self.manager, model, job,
                                      [reactant_one_guess.id(), product_guess.id()],
                                      settings)
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        # add some graphs because we will need them later. It is not important whether they actually match the structure
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        product_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        original_elementary_step = db.ElementaryStep.make([reactant_one_guess.id()], [product_guess.id()],
                                                          self.manager.get_collection("elementary_steps"))
        calculation.set_auxiliaries({"elementary-step-id": original_elementary_step.id()})

        # Run calculation/job
        config = self.get_configuration()
        job = ScineBsplineOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE

    @skip_without('database', 'readuct', 'molassembler')
    def test_charges(self):
        # import Job
        from scine_puffin.jobs.scine_bspline_optimization import ScineBsplineOptimization
        import scine_database as db

        # Amine-addition of the proline derivative to the aldehyde group of propanal.
        # These structures where originally optimized with dftb3
        reactant_one_path = os.path.join(resource_path(), "spline.start.xyz")
        reactant_one_guess = add_structure(self.manager, reactant_one_path, db.Label.COMPLEX_OPTIMIZED)
        product_path = os.path.join(resource_path(), "spline.end.xyz")
        product_guess = add_structure(self.manager, product_path, db.Label.MINIMUM_OPTIMIZED)

        model = db.Model('dftb3', 'dftb3', 'none')
        job = db.Job('scine_bspline_optimization')
        settings = {
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.05,
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
            "bspline_num_integration_points": 40,
            "sp_expect_charge_separation": True
        }

        calculation = add_calculation(self.manager, model, job,
                                      [reactant_one_guess.id(), product_guess.id()],
                                      settings)
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        # add some graphs because we will need them later. It is not important whether they actually match the structure
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        product_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        original_elementary_step = db.ElementaryStep.make([reactant_one_guess.id()], [product_guess.id()],
                                                          self.manager.get_collection("elementary_steps"))
        calculation.set_auxiliaries({"elementary-step-id": original_elementary_step.id()})

        # Run calculation/job
        config = self.get_configuration()
        job = ScineBsplineOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)
        assert calculation.get_status() == db.Status.COMPLETE

    @skip_without('database', 'readuct', 'molassembler')
    def test_complex_to_complex(self):
        # import Job
        from scine_puffin.jobs.scine_bspline_optimization import ScineBsplineOptimization
        import scine_database as db

        reactant_one_path = os.path.join(resource_path(), "acetal_methanol_complex.xyz")
        reactant_one_guess = add_structure(self.manager, reactant_one_path, db.Label.MINIMUM_GUESS)
        product_path = os.path.join(resource_path(), "methanol_enamine_complex.xyz")
        product_guess = add_structure(self.manager, product_path, db.Label.MINIMUM_GUESS)

        model = db.Model('pm6', 'pm6', 'none')
        job = db.Job('scine_bspline_optimization')
        settings = {
            "tsopt_convergence_max_iterations": 100,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.1,
            "tsopt_bofill_follow_mode": 0,
            "irc_convergence_max_iterations": 75,
            "irc_sd_factor": 2.0,
            "irc_irc_initial_step_size": 0.7,
            "irc_stop_on_error": False,
            "irc_convergence_step_max_coefficient": 0.002,
            "irc_convergence_step_rms": 0.001,
            "irc_convergence_gradient_max_coefficient": 0.0002,
            "irc_convergence_gradient_rms": 0.0001,
            "irc_convergence_delta_value": 1e-06,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 1000,
            "ircopt_convergence_step_max_coefficient": 0.002,
            "ircopt_convergence_step_rms": 0.001,
            "ircopt_convergence_gradient_max_coefficient": 0.0002,
            "ircopt_convergence_gradient_rms": 0.0001,
            "ircopt_convergence_requirement": 3,
            "ircopt_convergence_delta_value": 1e-06,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.2,
            "opt_convergence_max_iterations": 1000,
            "opt_convergence_step_max_coefficient": 0.002,
            "opt_convergence_step_rms": 0.001,
            "opt_convergence_gradient_max_coefficient": 0.0002,
            "opt_convergence_gradient_rms": 0.0001,
            "opt_convergence_requirement": 3,
            "opt_convergence_delta_value": 1e-06,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_use_trust_radius": True,
            "opt_bfgs_trust_radius": 0.4,
            "sp_expect_charge_separation": True,
            "spin_propensity_check": 0,
        }

        calculation = add_calculation(self.manager, model, job,
                                      [reactant_one_guess.id(), product_guess.id()],
                                      settings)
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        # add some graphs because we will need them later. It is not important whether they actually match the structure
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        product_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        original_elementary_step = db.ElementaryStep.make([reactant_one_guess.id()], [product_guess.id()],
                                                          self.manager.get_collection("elementary_steps"))
        calculation.set_auxiliaries({"elementary-step-id": original_elementary_step.id()})

        # Run calculation/job
        config = self.get_configuration()
        job = ScineBsplineOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)
