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


class ScineReactComplexNt2JobTest(JobTestCase):

    @skip_without('database', 'readuct', 'molassembler')
    def test_energy_and_structure(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
        ts_reference_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        product_reference_path = os.path.join(resource_path(), "proline_acid_propanal_product.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path)
        reactant_one_guess = db.Structure(compound_one.get_centroid())
        reactant_two_guess = db.Structure(compound_two.get_centroid())
        reactant_one_guess.link(self.manager.get_collection('structures'))
        reactant_two_guess.link(self.manager.get_collection('structures'))
        ts_reference = add_structure(self.manager, ts_reference_path, db.Label.TS_OPTIMIZED)
        product_reference = utils.io.read(str(product_reference_path))[0]
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_complex_nt2')
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

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
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
        assert structures.count("{}") == 3 + 3 + 2
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        assert len(new_elementary_step.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        bonds = utils.BondOrderCollection()
        bonds.matrix = db.SparseMatrixProperty(product.get_property('bond_orders'), properties).get_data()
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)

        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 1

        fit = utils.QuaternionFit(ts_reference.get_atoms().positions, new_ts.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

        fit = utils.QuaternionFit(product_reference.positions, product.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

    @skip_without('database', 'readuct', 'molassembler')
    def test_structure_deduplication(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
        ts_reference_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        product_reference_path = os.path.join(resource_path(), "proline_acid_propanal_product.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path)
        reactant_one_guess = db.Structure(compound_one.get_centroid())
        reactant_two_guess = db.Structure(compound_two.get_centroid())
        reactant_one_guess.link(self.manager.get_collection('structures'))
        reactant_two_guess.link(self.manager.get_collection('structures'))
        ts_reference = add_structure(self.manager, ts_reference_path, db.Label.TS_OPTIMIZED)
        product_reference = utils.io.read(str(product_reference_path))[0]
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        # Add fakes of the expected re-optimized starting structures too.
        isomer_of_r_one = add_structure(self.manager, reactant_one_path, db.Label.MINIMUM_OPTIMIZED)
        isomer_of_r_two = add_structure(self.manager, reactant_two_path, db.Label.MINIMUM_OPTIMIZED)
        isomer_of_r_one.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        isomer_of_r_one.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        isomer_of_r_one.set_graph("masm_decision_list", "(46, 51, 57, 1):(-144, -138, -133, 1)")
        isomer_of_r_two.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        isomer_of_r_two.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        isomer_of_r_two.set_graph("masm_decision_list", "(-134, -127, -123, 1):(28, 34, 39, 3)")
        compound_one.add_structure(isomer_of_r_one.get_id())
        compound_two.add_structure(isomer_of_r_two.get_id())
        isomer_of_r_one.set_compound(compound_one.get_id())
        isomer_of_r_two.set_compound(compound_two.get_id())

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_complex_nt2')
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

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
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
        assert structures.count("{}") == 3 + 3 + 2
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[-1], elementary_steps)
        assert len(new_elementary_step.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        bonds = utils.BondOrderCollection()
        bonds.matrix = db.SparseMatrixProperty(product.get_property('bond_orders'), properties).get_data()
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)

        # Check deduplication
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        duplicates = new_elementary_step.get_reactants(db.Side.LHS)[0]
        assert str(isomer_of_r_one.get_id()) in [str(d) for d in duplicates]
        assert str(isomer_of_r_two.get_id()) in [str(d) for d in duplicates]

        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 1

        fit = utils.QuaternionFit(ts_reference.get_atoms().positions, new_ts.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

        fit = utils.QuaternionFit(product_reference.positions, product.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

    @skip_without('database', 'readuct', 'molassembler')
    def test_mep_storage(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
        ts_reference_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        product_reference_path = os.path.join(resource_path(), "proline_acid_propanal_product.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path)
        reactant_one_guess = db.Structure(compound_one.get_centroid())
        reactant_two_guess = db.Structure(compound_two.get_centroid())
        reactant_one_guess.link(self.manager.get_collection('structures'))
        reactant_two_guess.link(self.manager.get_collection('structures'))
        ts_reference = add_structure(self.manager, ts_reference_path, db.Label.TS_OPTIMIZED)
        product_reference = utils.io.read(str(product_reference_path))[0]
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_complex_nt2')
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
            "store_full_mep": True,
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

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.property_ids) == 10
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
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)

        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 1
        selection = {"label": "elementary_step_optimized"}
        n_mep_structures = structures.count(json.dumps(selection))
        assert n_mep_structures >= 580
        assert n_mep_structures <= 620
        assert structures.count("{}") == 3 + 3 + 2 + n_mep_structures

        fit = utils.QuaternionFit(ts_reference.get_atoms().positions, new_ts.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

        fit = utils.QuaternionFit(product_reference.positions, product.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

    @skip_without('database', 'readuct', 'molassembler')
    def test_full_storage(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
        ts_reference_path = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        product_reference_path = os.path.join(resource_path(), "proline_acid_propanal_product.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path)
        reactant_one_guess = db.Structure(compound_one.get_centroid())
        reactant_two_guess = db.Structure(compound_two.get_centroid())
        reactant_one_guess.link(self.manager.get_collection('structures'))
        reactant_two_guess.link(self.manager.get_collection('structures'))
        ts_reference = add_structure(self.manager, ts_reference_path, db.Label.TS_OPTIMIZED)
        product_reference = utils.io.read(str(product_reference_path))[0]
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_complex_nt2')
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
            "store_all_structures": True,
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

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.property_ids) == 10
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
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.6595724342182, delta=1e-1)

        selection = {"label": "complex_optimized"}
        assert structures.count(json.dumps(selection)) == 1
        selection = {"label": "minimum_guess"}
        n_opt_structures = structures.count(json.dumps(selection))
        assert n_opt_structures >= 520  # Subject to numerics and changes in GeoOpt algorithm
        assert n_opt_structures <= 560  # Subject to numerics and changes in GeoOpt algorithm
        selection = {"label": "reactive_complex_scanned"}
        n_nt_structures = structures.count(json.dumps(selection))
        assert n_nt_structures >= 50  # Subject to numerics and changes in NT2 algorithm
        assert n_nt_structures <= 60  # Subject to numerics and changes in NT2 algorithm
        selection = {"label": "ts_guess"}
        n_tsopt_structures = structures.count(json.dumps(selection))
        assert n_tsopt_structures >= 20  # Subject to numerics and changes in TSOpt algorithm
        assert n_tsopt_structures <= 30  # Subject to numerics and changes in TSOpt algorithm
        selection = {"label": "elementary_step_optimized"}
        n_irc_structures = structures.count(json.dumps(selection))
        assert n_irc_structures == 150  # Subject to numerics and changes in IRCScan algorithm but does not converge
        total = 3 + 3 + 2 + n_opt_structures + n_nt_structures + n_tsopt_structures + n_irc_structures
        assert structures.count("{}") == total

        fit = utils.QuaternionFit(ts_reference.get_atoms().positions, new_ts.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

        fit = utils.QuaternionFit(product_reference.positions, product.get_atoms().positions)
        assert fit.get_rmsd() < 1e-2

    @skip_without('database', 'readuct', 'molassembler')
    def test_charged_reactants(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt2 import ScineReactComplexNt2
        import scine_database as db

        reactant_one_path = os.path.join(resource_path(), "proline_deprotonated.xyz")
        reactant_two_path = os.path.join(resource_path(), "imine.xyz")

        reactant_one_guess = add_structure(self.manager, reactant_one_path, db.Label.MINIMUM_OPTIMIZED)
        reactant_two_guess = add_structure(self.manager, reactant_two_path, db.Label.MINIMUM_OPTIMIZED)
        graph_one = json.load(open(os.path.join(resource_path(), "proline_deprotonated.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "imine.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])
        reactant_one_guess.charge = -1
        reactant_two_guess.charge = +1

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_complex_nt2')
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
            "tsopt_bofill_trust_radius": 0.1,
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
                15,
                37
            ],
            "nt_nt_dissociations": [],
            "rc_x_alignment_0": [
                0.0,
                -0.812898850334768,
                -0.582404892771698,
                0.812898850334768,
                0.339195459124413,
                -0.473436267763457,
                0.582404892771698,
                -0.473436267763457,
                0.660804540875587
            ],
            "rc_x_alignment_1": [
                0.145609985067041,
                0.967519900776486,
                0.206646978807424,
                -0.967519900776486,
                0.182885300756383,
                -0.174522801858975,
                -0.206646978807424,
                -0.174522801858975,
                0.962724684310658
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 2.46609259372247,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False
        }

        calculation = add_calculation(self.manager, model, job, [reactant_one_guess.id(), reactant_two_guess.id()],
                                      settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt2()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        try:
            _ = job.run(self.manager, calculation, config)
        except BaseException as e:
            print(calculation.get_comment())
            raise e

        assert "The chosen spin multiplicity (1) is not compatible with the molecular charge (0)."\
               not in calculation.get_comment()
