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
)

from ..resources import resource_path


class ScineReactComplexNtJobTest(JobTestCase):

    @skip_without('database', 'readuct', 'molassembler')
    def test_energy_and_structure(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt import ScineReactComplexNt
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
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
        job = db.Job('scine_react_complex_nt')
        settings = {
            "nt_convergence_max_iterations": 600,
            "nt_nt_total_force_norm": 0.05,
            "nt_sd_factor": 1.0,
            "nt_nt_use_micro_cycles": True,
            "nt_nt_fixed_number_of_micro_cycles": True,
            "nt_nt_number_of_micro_cycles": 30,
            "nt_nt_filter_passes": 10,
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.3,
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
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
            "nt_nt_lhs_list": [
                3,
                16,
            ],
            "nt_nt_rhs_list": [
                0,
                3
            ],
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
            "rc_minimal_spin_multiplicity": True,
            "spin_propensity_check": 3
        }

        calculation = add_calculation(self.manager, model, job, [reactant_one_guess.id(), reactant_two_guess.id()],
                                      settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.property_ids) == 10
        assert len(results.structure_ids) == 3 + 2  # re-optimized reactants (x2) + complex + TS + product
        assert len(results.elementary_step_ids) == 2
        new_elementary_step_one = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert new_elementary_step_one.get_type() == db.ElementaryStepType.BARRIERLESS
        new_elementary_step_two = db.ElementaryStep(results.elementary_step_ids[1], elementary_steps)
        assert new_elementary_step_two.get_type() == db.ElementaryStepType.REGULAR
        assert new_elementary_step_one.get_reactants(db.Side.RHS)[1][0] == \
            new_elementary_step_two.get_reactants(db.Side.LHS)[0][0]
        s_complex = db.Structure(new_elementary_step_two.get_reactants(db.Side.LHS)[0][0])
        s_complex.link(structures)
        assert s_complex.get_label() == db.Label.COMPLEX_OPTIMIZED
        assert len(new_elementary_step_two.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step_two.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        bonds = utils.BondOrderCollection()
        bonds.matrix = db.SparseMatrixProperty(product.get_property('bond_orders'), properties).get_data()
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step_two.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.659096385274275, delta=1e-1)
