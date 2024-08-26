#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import json
from typing import Dict

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


class ScineReactComplexNt2ObserverTests(JobTestCase):

    # all subject to numerics and changes in optimization algorithms
    # lower and upper bound
    n_opt = (520, 1560)
    n_nt = (50, 60)
    n_tsopt = (20, 30)
    n_irc = (150, 150)

    def execute_job(self, additional_settings) -> tuple:
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        import scine_database as db

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
        db_job = db.Job('scine_react_complex_nt2')
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
        for k, v in additional_settings.items():
            settings[k] = v

        calculation = add_calculation(self.manager, model, db_job,
                                      [reactant_one_guess.id(), reactant_two_guess.id()],
                                      settings)
        config = self.get_configuration()
        job = ScineReactComplexNt2()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)
        return calculation, job

    def standard_check(self, calculation, connectivity_settings) -> None:
        import scine_database as db
        import scine_utilities as utils
        from scine_puffin.utilities.masm_helper import get_molecules_result

        ts_reference_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        ts_reference = add_structure(self.manager, ts_reference_path, db.Label.TS_OPTIMIZED)
        product_reference_path = os.path.join(resource_path(), "proline_acid_propanal_product.xyz")
        product_reference = utils.io.read(str(product_reference_path))[0]

        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.property_ids) == 11
        # Structure counts: (complex + TS + product) + re-optimized reactants (x2)
        assert len(results.structure_ids) == 3 + 2
        assert len(results.elementary_step_ids) == 2
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        assert len(new_elementary_step.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        bonds = utils.BondOrderCollection()
        bonds.matrix = db.SparseMatrixProperty(product.get_property('bond_orders'), properties).get_data()
        assert len(get_molecules_result(product.get_atoms(), bonds, connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)
        fit = utils.QuaternionFit(ts_reference.get_atoms().positions, new_ts.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2
        fit = utils.QuaternionFit(product_reference.positions, product.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

    def observer_check(self, factors: Dict[str, float], more_relaxed: bool = False) -> None:
        from ...utilities.task_to_readuct_call import SubTaskToReaductCall

        lee_way = 10 if more_relaxed else 0

        structures = self.manager.get_collection("structures")
        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 1
        selection = {"label": "minimum_guess"}
        n_opt_structures = structures.count(json.dumps(selection))
        assert (self.n_opt[0] * factors[SubTaskToReaductCall.OPT.name] - lee_way <= n_opt_structures <=
                self.n_opt[1] * factors[SubTaskToReaductCall.OPT.name] + lee_way)
        selection = {"label": "reactive_complex_scanned"}
        n_nt_structures = structures.count(json.dumps(selection))
        assert (self.n_nt[0] * factors[SubTaskToReaductCall.NT2.name] - lee_way <= n_nt_structures <=
                self.n_nt[1] * factors[SubTaskToReaductCall.NT2.name] + lee_way)
        selection = {"label": "ts_guess"}
        n_tsopt_structures = structures.count(json.dumps(selection))
        assert (self.n_tsopt[0] * factors[SubTaskToReaductCall.TSOPT.name] - lee_way <= n_tsopt_structures <=
                self.n_tsopt[1] * factors[SubTaskToReaductCall.TSOPT.name] + lee_way)
        selection = {"label": "elementary_step_optimized"}
        n_irc_structures = structures.count(json.dumps(selection))
        if factors[SubTaskToReaductCall.IRC.name] == 1:
            assert n_irc_structures == self.n_irc[0]  # IRCScan algorithm does not converge
        else:
            # relax a bit due to numerics of fractions
            assert (self.n_irc[0] * factors[SubTaskToReaductCall.IRC.name] - 10 <= n_irc_structures <=
                    self.n_irc[1] * factors[SubTaskToReaductCall.IRC.name] + 10)
        total = 3 + 3 + 2 + n_opt_structures + n_nt_structures + n_tsopt_structures + n_irc_structures
        assert structures.count("{}") == total

    @skip_without('database', 'readuct', 'molassembler')
    def test_full_storage(self):
        from ...utilities.task_to_readuct_call import SubTaskToReaductCall

        calculation, job = self.execute_job({"store_all_structures": True})
        self.standard_check(calculation, job.connectivity_settings)
        self.observer_check(
            {
                SubTaskToReaductCall.OPT.name: 1,
                SubTaskToReaductCall.NT2.name: 1,
                SubTaskToReaductCall.TSOPT.name: 1,
                SubTaskToReaductCall.IRC.name: 1
            }
        )

    @skip_without('database', 'readuct', 'molassembler')
    def test_fraction_storage(self):
        from ...utilities.task_to_readuct_call import SubTaskToReaductCall
        import scine_utilities as utils
        fractions = utils.ValueCollection({
            SubTaskToReaductCall.OPT.name: 0.1,
            SubTaskToReaductCall.NT2.name: 0.5,
            SubTaskToReaductCall.TSOPT.name: 0.2,
            SubTaskToReaductCall.IRC.name: 0.2
        })
        calculation, job = self.execute_job(
            {"store_structures_with_fraction": fractions}
        )
        self.standard_check(calculation, job.connectivity_settings)
        self.observer_check(fractions.as_dict(), True)  # more relaxed check due to randomness

    @skip_without('database', 'readuct', 'molassembler')
    def test_frequency_storage(self):
        from ...utilities.task_to_readuct_call import SubTaskToReaductCall
        import scine_utilities as utils
        frequencies = utils.ValueCollection({
            SubTaskToReaductCall.OPT.name: 10,
            SubTaskToReaductCall.NT2.name: 2,
            SubTaskToReaductCall.TSOPT.name: 5,
            SubTaskToReaductCall.IRC.name: 5
        })
        calculation, job = self.execute_job(
            {"store_structures_with_frequency": frequencies}
        )
        self.standard_check(calculation, job.connectivity_settings)
        fractions = {
            k: 1 / v for k, v in frequencies.items()
        }
        self.observer_check(fractions)
