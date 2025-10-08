#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import pytest

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_structure
)

from ..resources import resource_path


class ScineKingfisherJobTest(JobTestCase):
    @staticmethod
    def _add_sfam_parameters_to_structure(filename: str, sfam_model,
                                          structure, properties):
        import scine_database as db
        with open(os.path.join(resource_path(), filename)) as f:
            sfam = f.read()
        sfam_prop = db.StringProperty.make("sfam_parameters", sfam_model,
                                           sfam, properties)
        structure.add_property("sfam_parameters", sfam_prop.id())

    @skip_without('database', 'swoose', 'readuct', 'sparrow')
    def test_qmmm_kingfisher(self):
        import scine_utilities as utils
        import scine_database as db
        from scine_puffin.jobs.scine_kingfisher import ScineKingfisher
        import scine_puffin.utilities.swoose_helper as swoose_helper

        properties = self.manager.get_collection("properties")

        sfam_param_model = db.Model("dft", "pbe-d3", "def2-svpd")

        # Prepare CO and H2
        co_path = os.path.join(resource_path(), "co.xyz")
        co_struct = add_structure(self.manager, co_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "co.sfam"),
                                               sfam_param_model, co_struct, properties)
        h2_path = os.path.join(resource_path(), "h2.sfam.xyz")
        h2_struct = add_structure(self.manager, h2_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "h2.sfam"),
                                               sfam_param_model, h2_struct, properties)
        # # Settings for CO + H2
        settings = utils.ValueCollection({
            "max_scf_iterations": 250,
            "kf_silence_underlying_calculators": True,
            "kf_seed": 5,
            "kfp_shells": 1,
            "kfp_coverage_threshold": 0.65,  # 0.85 was good in 4min
            "kf_nt_qm_region_radius": 4.0,
            "kf_tsopt_qm_region_radius": 2.5,
            "kf_tsextract_qm_region_radius": 2.6,
            "kf_tsextract_qm_region_hbond_based": True,
            "kf_nt_associations": [0, 2, 0, 3],
            "kf_nt_dissociations": [],
            "rc_x_alignment_0": [
                -0.823414333, 0.198504080, 0.531587215,
                -0.198504080, 0.776856919, -0.597569626,
                -0.531587215, -0.597569626, -0.600271252
            ],
            "rc_x_alignment_1": [
                -0.604296725, -0.143512434, 0.783728046,
                0.143512434, 0.947951356, 0.284240053,
                -0.783728046, 0.284240053, -0.552248081
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 1.5,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False,
            "mmopt_convergence_max_iterations": 150,
            "mmopt_stop_on_error": False,
            "preopt_qmmm_opt_max_macroiterations": 20,
            "preopt_qmmm_opt_max_full_microiterations": 15,
            "preopt_qmmm_opt_max_env_microiterations": 100,
            "nt_nt_number_of_micro_cycles": 20,
            "nt_nt_total_force_norm": 0.1,
            # TS Opt
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.1,
            "tsopt_qmmm_opt_max_macroiterations": 50,
            "tsopt_convergence_requirement": 3,
            # IRC
            "irc_irc_initial_step_size": 0.5,
            "irc_convergence_max_iterations": 50,
            # IRC Opt
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
            # For optimization of QM Products
            "opt_convergence_max_iterations": 150,
            "opt_convergence_requirement": 2,
        })

        # Set up + Run QM/MM
        qmmm_model = db.Model('pm6/SFAM', 'pm6', '')
        qmmm_model.program = "sparrow/swoose"
        job = db.Job('scine_kingfisher')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job,
                                           [co_struct.id(), h2_struct.id(), h2_struct.id()],
                                           settings)

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineKingfisher()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config, clear_job=False)

        # Check results
        # Reference parameters
        ref_para = swoose_helper.SFAMParameters()
        ref_para.bonds = [
            ["C1_O1", "O1_C1", "1.14179", "1596.88"],
            ["H_H", "H_H_H", "0.76638", "381.637"]
        ]
        ref_para.charges = [
            ['C1_O1', '0.157448'], ['H_H', '0.00615727'],
            ['H_H_H', '-0.00615727'], ['O1_C1', '-0.157448']
        ]
        # Read combined parameters
        combined_para = swoose_helper.SFAMParameters()
        swoose_helper.parse_parameter(config["daemon"]["job_dir"] + "/" +
                                      qmmm_calculation.id().string() + "/Parameters.dat",
                                      combined_para)
        assert combined_para.bonds == ref_para.bonds
        assert combined_para.charges == ref_para.charges

        assert qmmm_calculation.get_status() == db.Status.COMPLETE
        results = qmmm_calculation.get_results()
        # 2 reactants + 2 products + 2 complexes + 1 QM/MM TS + 1 QM TS Guess
        assert len(results.structure_ids) == 8
        # 2 barrierless reactions and one regular transition state
        assert len(results.elementary_step_ids) == 3
        # 3 properties per min. product structures, 4 for TS, 4 for QM TS Guess, 3 properties per complex structure
        assert len(results.property_ids) == 4 * 3 + 4 + 4 + 3 * 2
        # Clear job after checking if files are there
        job.clear()

    @skip_without('database', 'swoose', 'readuct', 'sparrow')
    @pytest.mark.slow  # type: ignore[misc]
    def test_qmmm_kingfisher_2(self):
        import scine_database as db
        from scine_puffin.jobs.scine_kingfisher import ScineKingfisher

        # Setup DB for calculation
        sfam_param_model = db.Model("dft", "pbe-d3", "def2-svp")
        # Set QM atoms
        properties = self.manager.get_collection("properties")

        co = os.path.join(resource_path(), "co.xyz")
        co_str = add_structure(self.manager, co, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "co.sfam"),
                                               sfam_param_model, co_str, properties)

        hf = os.path.join(resource_path(), "hf.xyz")
        hf_str = add_structure(self.manager, hf, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "hf.sfam"),
                                               sfam_param_model, hf_str, properties)
        # # Settings for CO + HF
        settings = {
            "kf_silence_underlying_calculators": True,
            "kfp_shells": 1,  # 2 success
            "kfp_coverage_threshold": 0.85,  # 0.85 was good in 4min
            "kf_nt_qm_region_radius": 4.0,
            "kf_nt_associations": [0, 3
                                   # 1, 5
                                   ],
            "kf_nt_dissociations": [2, 3],
            "rc_x_alignment_0": [
                0.57011828, 0.1150395, 0.81346854,
                -0.1150395, 0.99157128, -0.05960125,
                -0.81346854, -0.05960125, 0.578547
            ],
            "rc_x_alignment_1": [
                -0.93210718, 0.03197254, -0.36076857,
                -0.09638425, 0.93827897, 0.33217865,
                0.34912216, 0.34439851, -0.87149491,
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 2.597011752055531,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False,
            "mmopt_convergence_max_iterations": 150,
            "mmopt_stop_on_error": False,
            "preopt_qmmm_opt_max_macroiterations": 20,
            "preopt_qmmm_opt_max_full_microiterations": 15,
            "preopt_qmmm_opt_max_env_microiterations": 100,
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.1,
            "tsopt_qmmm_opt_max_macroiterations": 50,
            "tsopt_convergence_requirement": 2,
            "irc_irc_initial_step_size": 0.5,  # 1.5,  # worked with 2.5, because of some tunneling
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
        }

        # Set up + Run QM/MM
        qmmm_model = db.Model('pm6/SFAM', 'pm6', '')
        qmmm_model.program = "sparrow/swoose"
        job = db.Job('scine_kingfisher')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job,
                                           [co_str.id(), hf_str.id(), hf_str.id()],
                                           settings)
        settings = qmmm_calculation.get_settings()

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineKingfisher()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config)

        assert qmmm_calculation.get_status() == db.Status.COMPLETE
        results = qmmm_calculation.get_results()
        # 2 reactants + 2 products + 2 complexes + 1 QM/MM TS + 1 QM TS Guess
        assert len(results.structure_ids) == 8
        # 2 barrierless reactions and one regular transition state
        assert len(results.elementary_step_ids) == 3
        # 3 properties per min. product structures, 4 for TS, 4 for QM TS Guess, 3 properties per complex structure
        assert len(results.property_ids) == 4 * 3 + 4 + 4 + 3 * 2
        # Clear job after checking if files are there
        job.clear()

    @skip_without('database', 'swoose', 'readuct', 'sparrow')
    @pytest.mark.slow  # type: ignore[misc]
    def test_mixed_solvents_kingfisher(self):
        import scine_utilities as utils
        import scine_database as db
        from scine_puffin.jobs.scine_kingfisher import ScineKingfisher
        # import scine_puffin.utilities.swoose_helper as swoose_helper

        properties = self.manager.get_collection("properties")

        sfam_param_model = db.Model("dft", "pbe-d3", "def2-svpd")

        # Prepare CO and H2
        co_path = os.path.join(resource_path(), "co.xyz")
        co_struct = add_structure(self.manager, co_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "co.sfam"),
                                               sfam_param_model, co_struct, properties)
        h2_path = os.path.join(resource_path(), "h2.sfam.xyz")
        h2_struct = add_structure(self.manager, h2_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "h2.sfam"),
                                               sfam_param_model, h2_struct, properties)

        meoh_path = os.path.join(resource_path(), "meoh.xyz")
        meoh_struct = add_structure(self.manager, meoh_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "meoh.sfam"),
                                               sfam_param_model, meoh_struct, properties)
        # # Settings for CO + H2
        settings = utils.ValueCollection({
            "max_scf_iterations": 250,
            "allow_exhaustive_product_decomposition": True,
            "kf_silence_underlying_calculators": True,
            "kf_seed": 42,  # 42,  # 49,
            "kfp_shells": 1,  # 2 success
            "kfp_coverage_threshold": 0.65,  # 0.85 was good in 4min
            "kf_solvent_types": 2,
            "kf_solvent_ratio": [4, 1],
            "kf_nt_qm_region_radius": 4.0,
            "kf_tsopt_qm_region_radius": 2.5,
            "kf_tsextract_qm_region_radius": 2.6,
            "kf_tsextract_qm_region_hbond_based": True,
            "kf_nt_associations": [0, 2, 0, 3
                                   # 1, 5
                                   ],
            "kf_nt_dissociations": [],
            "rc_x_alignment_0": [
                -0.823414333, 0.198504080, 0.531587215,
                -0.198504080, 0.776856919, -0.597569626,
                -0.531587215, -0.597569626, -0.600271252
            ],
            "rc_x_alignment_1": [
                -0.604296725, -0.143512434, 0.783728046,
                0.143512434, 0.947951356, 0.284240053,
                -0.783728046, 0.284240053, -0.552248081
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 1.5,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False,
            "mmopt_convergence_max_iterations": 150,
            "mmopt_stop_on_error": False,
            "preopt_qmmm_opt_max_macroiterations": 20,
            "preopt_qmmm_opt_max_full_microiterations": 15,
            "preopt_qmmm_opt_max_env_microiterations": 100,
            "nt_nt_number_of_micro_cycles": 20,
            "nt_nt_total_force_norm": 0.1,
            # TS Opt
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.1,
            "tsopt_qmmm_opt_max_macroiterations": 50,
            "tsopt_convergence_requirement": 3,
            # IRC
            "irc_irc_initial_step_size": 0.5,
            "irc_convergence_max_iterations": 50,  # 1.5,  # worked with 2.5, because of some tunneling
            # IRC Opt
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
            # For optimization of QM Products
            "opt_convergence_max_iterations": 150,
            "opt_convergence_requirement": 2,
        })

        # Set up + Run QM/MM
        qmmm_model = db.Model('pm6/SFAM', 'pm6', '')
        qmmm_model.program = "sparrow/swoose"
        job = db.Job('scine_kingfisher')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job,
                                           [co_struct.id(), h2_struct.id(),  # reactants
                                            h2_struct.id(), meoh_struct.id()],  # solvents
                                           settings.as_dict())

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineKingfisher()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config, clear_job=False)

        assert qmmm_calculation.get_status() == db.Status.COMPLETE
        results = qmmm_calculation.get_results()
        # # 5 reactants + 5 products + 2 complexes + 1 QM/MM TS + 1 QM TS Guess
        assert len(results.structure_ids) == 14
        # # 2 barrierless reactions and one regular transition state
        assert len(results.elementary_step_ids) == 3
        # # 3 properties per min. product structures, 4 for TS, 4 for QM TS Guess + properties of complexes
        assert len(results.property_ids) >= 7 * 3 + 4 + 4
        # Clear job after checking if files are there
        job.clear()

    @skip_without('database', 'swoose', 'readuct', 'xtb_wrapper')
    @pytest.mark.slow  # type: ignore[misc]
    def test_slow_hydrolysis(self):
        import scine_utilities as utils
        import scine_database as db
        from scine_puffin.jobs.scine_kingfisher import ScineKingfisher
        from scine_puffin.jobs.scine_react_ts_guess import ScineReactTsGuess

        # Setup DB for calculation
        sfam_param_model = db.Model("dft", "pbe-d3", "def2-svp")
        # Set QM atoms
        properties = self.manager.get_collection("properties")

        ch2o = os.path.join(resource_path(), "ch2o.xyz")
        ch2o_str = add_structure(self.manager, ch2o, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "ch2o.sfam"),
                                               sfam_param_model, ch2o_str, properties)
        water = os.path.join(resource_path(), "good_water.xyz")
        water_str = add_structure(self.manager, water, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "water.sfam"),
                                               sfam_param_model, water_str, properties)
        # # # Settings for CH2O + H2O
        settings = utils.ValueCollection({
            # "kf_silence_underlying_calculators": True,
            "kfp_shells": 2,  # 2 success
            "kfp_coverage_threshold": 0.75,  # 0.75 success
            "kf_nt_qm_region_radius": 4.0,
            "kf_tsextract_qm_region_radius": 1.6,
            "kf_nt_associations": [0, 4,
                                   # 1, 5
                                   ],
            "kf_nt_dissociations": [4, 5],
            "rc_x_alignment_0": [
                3.11659008e-06, -1.00000000e+00, -3.11659330e-06,
                1.00000000e+00, 3.11659980e-06, -3.11658359e-06,
                3.11659330e-06, -3.11658359e-06, 1.00000000e+00
            ],
            "rc_x_alignment_1": [
                0.85678467, -0.48449087, 0.17660303,
                0.51546805, 0.79496555, -0.31987883,
                0.01458505, 0.3651005, 0.93085386,
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 2.597011752055531,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False,
            "mmopt_convergence_max_iterations": 500,
            "mmopt_stop_on_error": False,
            "preopt_qmmm_opt_max_macroiterations": 20,
            "preopt_qmmm_opt_max_full_microiterations": 15,
            "preopt_qmmm_opt_max_env_microiterations": 200,
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.1,
            "tsopt_qmmm_opt_max_macroiterations": 50,
            "tsopt_convergence_requirement": 2,
            "irc_irc_initial_step_size": 0.5,  # 1.5,  # worked with 2.5, because of some tunneling
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
        })

        # Set up + Run QM/MM
        qmmm_model = db.Model('gfn2/SFAM', 'gfn2', '')
        qmmm_model.program = "xtb/swoose"
        job = db.Job('scine_kingfisher')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job,
                                           [ch2o_str.id(), water_str.id(), water_str.id()],
                                           settings.as_dict())
        settings = qmmm_calculation.get_settings()

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineKingfisher()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config, clear_job=False)

        # Check results of QM/MM
        assert qmmm_calculation.get_status() == db.Status.COMPLETE
        qmmm_results = qmmm_calculation.get_results()
        # 2 barrierless reactions and one regular transition state
        assert len(qmmm_results.elementary_step_ids) == 3

        ts_guess_id = qmmm_results.structure_ids[-1]
        ts_guess = db.Structure(ts_guess_id, self.manager.get_collection('structures'))
        qm_model = db.Model('gfn2', 'gfn2', '')
        qm_model.program = "xtb"
        qm_model.solvent = "water"
        qm_model.solvation = "gbsa"
        job = db.Job('scine_react_ts_guess')
        qm_settings = {
            'tsopt_automatic_mode_selection': [0, 4, 5],
            "tsopt_convergence_requirement": 3,
            "irc_irc_initial_step_size": 0.5,
            "irc_convergence_max_iterations": 250,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 500,
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
            "ircopt_convergence_step_max_coefficient": 5.0e-3,
            "ircopt_convergence_step_rms": 1.0e-3,
            "ircopt_convergence_gradient_max_coefficient": 5.0e-4,
            "ircopt_convergence_gradient_rms": 1.0e-4,
            'spin_propensity_check': 0
        }
        qm_calculation = add_calculation(self.manager, qm_model, job, [ts_guess.id()], qm_settings)
        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactTsGuess()
        job.prepare(config["daemon"]["job_dir"], qm_calculation.id())
        self.run_job(job, qm_calculation, config, clear_job=False)

        # Check results of QM
        assert qm_calculation.get_status() == db.Status.COMPLETE
        qm_results = qm_calculation.get_results()
        # 2 barrierless reactions and one regular transition state
        assert len(qm_results.elementary_step_ids) == 3
        # Clear job after checking if files are there
        job.clear()

    @skip_without('database', 'swoose', 'readuct', 'xtb_wrapper')
    @pytest.mark.slow  # type: ignore[misc]
    def test_slow_chlorination(self):
        import scine_utilities as utils
        import scine_database as db
        from scine_puffin.jobs.scine_kingfisher import ScineKingfisher
        from scine_puffin.jobs.scine_react_ts_guess import ScineReactTsGuess

        # Setup DB for calculation
        sfam_param_model = db.Model("dft", "pbe-d3", "def2-svp")
        # Set QM atoms
        properties = self.manager.get_collection("properties")

        phenol_path = os.path.join(resource_path(), "phenol.xyz")
        phenol = add_structure(self.manager, phenol_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "phenol.sfam"),
                                               sfam_param_model, phenol, properties)
        hocl_path = os.path.join(resource_path(), "hocl.xyz")
        hocl = add_structure(self.manager, hocl_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "hocl.sfam"),
                                               sfam_param_model, hocl, properties)
        water_path = os.path.join(resource_path(), "good_water.xyz")
        water = add_structure(self.manager, water_path, db.Label.USER_OPTIMIZED)
        self._add_sfam_parameters_to_structure(os.path.join(resource_path(), "water.sfam"),
                                               sfam_param_model, water, properties)
        # # Settings for Phenol + HOCl + H2O
        settings = utils.ValueCollection({
            # "kf_silence_underlying_calculators": True,
            "kfp_shells": 2,  # 2 success
            "kfp_coverage_threshold": 0.75,  # 0.75 success
            "kf_nt_qm_region_radius": 4.0,
            "kf_tsextract_qm_region_radius": 1.6,
            "kf_nt_associations": [5, 15],
            "kf_nt_dissociations": [13, 15],
            "kf_include_acceptors": True,
            "kf_acceptors": [1, 2],  # OH Group
            "rc_x_alignment_0": [
                0.63956913, -0.295613, 0.70962264,
                0.295613, 0.94670121, 0.12794439,
                -0.70962264, 0.12794439, 0.69286791
            ],
            "rc_x_alignment_1": [
                0.34660598, -0.49755845, 0.79517286,
                0.8496858, 0.52565754, -0.04145115,
                -0.39736424, 0.6900143, 0.6049644
            ],
            "rc_x_rotation": 1.5,
            "rc_x_spread": 2.597011752055531,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": False,
            "mmopt_convergence_max_iterations": 500,
            "mmopt_stop_on_error": False,
            "preopt_qmmm_opt_max_macroiterations": 20,
            "preopt_qmmm_opt_max_full_microiterations": 15,
            "preopt_qmmm_opt_max_env_microiterations": 200,
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.1,
            "tsopt_qmmm_opt_max_macroiterations": 50,
            "tsopt_convergence_requirement": 2,
            "irc_irc_initial_step_size": 0.5,  # 1.5,  # worked with 2.5, because of some tunneling
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
        })

        # Set up + Run QM/MM
        qmmm_model = db.Model('gfn2/SFAM', 'gfn2', '')
        qmmm_model.program = "xtb/swoose"
        job = db.Job('scine_kingfisher')
        qmmm_calculation = add_calculation(self.manager, qmmm_model, job,
                                           [phenol.id(), hocl.id(), water.id()],
                                           settings.as_dict())
        settings = qmmm_calculation.get_settings()

        qmmm_calculation.set_settings(settings)

        config = self.get_configuration()
        job = ScineKingfisher()
        job.prepare(config["daemon"]["job_dir"], qmmm_calculation.id())
        self.run_job(job, qmmm_calculation, config)

        # Check results of QM/MM
        assert qmmm_calculation.get_status() == db.Status.COMPLETE
        results = qmmm_calculation.get_results()

        ts_guess_id = results.structure_ids[-1]
        ts_guess = db.Structure(ts_guess_id, self.manager.get_collection('structures'))
        qm_model = db.Model('gfn2', 'gfn2', '')
        qm_model.program = "xtb"
        qm_model.solvent = "water"
        qm_model.solvation = "gbsa"
        job = db.Job('scine_react_ts_guess')
        qm_settings = {
            'tsopt_automatic_mode_selection': [5, 13, 15],
            "tsopt_convergence_requirement": 3,
            "irc_irc_initial_step_size": 0.5,
            "irc_convergence_max_iterations": 250,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 500,
            "ircopt_optimizer": "bfgs",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.15,
            "ircopt_convergence_step_max_coefficient": 5.0e-3,
            "ircopt_convergence_step_rms": 1.0e-3,
            "ircopt_convergence_gradient_max_coefficient": 5.0e-4,
            "ircopt_convergence_gradient_rms": 1.0e-4,
            'spin_propensity_check': 0
        }
        qm_calculation = add_calculation(self.manager, qm_model, job, [ts_guess.id()], qm_settings)
        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactTsGuess()
        job.prepare(config["daemon"]["job_dir"], qm_calculation.id())
        self.run_job(job, qm_calculation, config)
