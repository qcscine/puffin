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
    add_compound_and_structure,
)

from ..resources import resource_path


class ScineReactComplexNt2JobTest(JobTestCase):

    @skip_without('database', 'readuct', 'molassembler')
    def test_propensity(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        import scine_database as db

        # The parallel numerical Hessian makes problems on some machines
        # Therefore, we enforce the use of the serial Hessian
        omp = os.getenv("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"

        model = db.Model('pm6', 'pm6', '')
        model.spin_mode = "unrestricted"
        model.program = "sparrow"

        structures = self.manager.get_collection('structures')
        reactant_one_path = os.path.join(resource_path(), "FeO_H2.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path, charge=1, multiplicity=6,
                                                  model=model)
        reactant_one_guess = db.Structure(compound_one.get_centroid(), structures)
        graph_one = json.load(open(os.path.join(resource_path(), "FeO_H2.json"), "r"))
        # graph_one = json.load(open(os.path.join(resource_path(), "FeO_H2_lhs.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])

        job = db.Job('scine_react_complex_nt2')
        settings = {
            "max_scf_iterations": 1000,
            "self_consistence_criterion": 1e-6,
            "density_rmsd_criterion": 1e-4,
            "spin_propensity_check_for_unimolecular_reaction": False,
            "spin_propensity_energy_range_to_save": 200.0,
            "spin_propensity_energy_range_to_optimize": 500.0,
            "spin_propensity_optimize_all": True,
            "spin_propensity_check": 2,
            "nt_nt_total_force_norm": 0.05,
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.2,
            "tsopt_convergence_delta_value": 1e-6,
            "irc_stop_on_error": False,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "irc_convergence_delta_value": 1e-6,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_delta_value": 1e-6,
            "ircopt_convergence_max_iterations": 1000,
            "opt_convergence_max_iterations": 1000,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_trust_radius": 0.2,
            "opt_convergence_delta_value": 1e-6,
            "nt_nt_associations": [2, 3],
            "nt_nt_dissociations": [
                0,
                2,
                1,
                3
            ]
        }

        calculation = add_calculation(self.manager, model, job, [reactant_one_guess.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()

        assert len(results.structure_ids) == 5  # TS + product + 1 spin state changed product + 2 spin changed TS
        assert len(results.elementary_step_ids) == 1
        assert structures.count("{}") == 6  # reactant + TS + product + 1 spin changed product + 2 spin changed TS
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        assert len(new_elementary_step.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.get_multiplicity() == 6
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.get_multiplicity() == 6
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids

        assert any(structure.get_multiplicity() == 2 for structure in structures.query_structures("{}"))
        assert any(structure.get_multiplicity() == 4 for structure in structures.query_structures("{}"))
        assert any(structure.get_multiplicity() == 6 for structure in structures.query_structures("{}"))

        os.environ["OMP_NUM_THREADS"] = omp

    def test_propensity_hit(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        import scine_database as db

        # The parallel numerical Hessian makes problems on some machines
        # Therefore, we enforce the use of the serial Hessian
        omp = os.getenv("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"

        model = db.Model('pm6', 'pm6', '')
        model.spin_mode = "unrestricted"
        model.program = "sparrow"

        structures = self.manager.get_collection('structures')
        reactant_one_path = os.path.join(resource_path(), "peroxide.xyz")
        # reactant_one_path = os.path.join(resource_path(), "FeO_H2_lhs.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path, charge=0, multiplicity=1,
                                                  model=model)
        reactant_one_guess = db.Structure(compound_one.get_centroid(), structures)
        graph_one = json.load(open(os.path.join(resource_path(), "peroxide.json"), "r"))
        # graph_one = json.load(open(os.path.join(resource_path(), "FeO_H2_lhs.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])

        job = db.Job('scine_react_complex_nt2')
        settings = {
            "max_scf_iterations": 1000,
            "self_consistence_criterion": 1e-7,
            "density_rmsd_criterion": 1e-4,
            "spin_propensity_check_for_unimolecular_reaction": False,
            "spin_propensity_energy_range_to_save": 250.0,
            "spin_propensity_energy_range_to_optimize": 500.0,
            "spin_propensity_optimize_all": True,
            "spin_propensity_check": 1,
            "nt_nt_total_force_norm": 0.05,
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_bofill_trust_radius": 0.2,
            "tsopt_convergence_delta_value": 1e-6,
            "irc_stop_on_error": False,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "irc_convergence_delta_value": 1e-6,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_delta_value": 1e-6,
            "ircopt_bfgs_trust_radius": 0.2,
            "ircopt_convergence_requirement": 1,
            "ircopt_convergence_max_iterations": 1000,
            "opt_convergence_max_iterations": 1000,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_trust_radius": 0.2,
            "opt_convergence_delta_value": 1e-6,
            "nt_nt_associations": [],
            "nt_nt_dissociations": [
                0, 1,
                2, 3,
                4, 5
            ]
        }

        calculation = add_calculation(self.manager, model, job, [reactant_one_guess.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()

        assert len(results.structure_ids) == 8  # TS + 3 product + 3 spin state changed product + 1 spin changed TS
        assert len(results.elementary_step_ids) == 1
        # Total structure count: reactant + TS + 3 product + 3 spin state changed product + 1 spin changed TS
        assert structures.count("{}") == 9
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        assert len(new_elementary_step.get_reactants(db.Side.RHS)[1]) == 3
        product = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.get_multiplicity() == 3
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        product_1 = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][1], structures)
        assert product_1.get_multiplicity() == 1
        assert product_1.has_property('bond_orders')
        assert product_1.has_graph('masm_cbor_graph')
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.get_multiplicity() == 1
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids

        assert any(structure.get_multiplicity() == 1 for structure in structures.query_structures("{}"))
        assert any(structure.get_multiplicity() == 3 for structure in structures.query_structures("{}"))

        os.environ["OMP_NUM_THREADS"] = omp
