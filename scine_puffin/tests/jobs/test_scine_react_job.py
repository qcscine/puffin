#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List
import unittest


import os
import numpy as np

from scine_puffin.config import Configuration
from scine_puffin.jobs.templates.job import calculation_context
from scine_puffin.jobs.templates.scine_react_job import ReactJob
from scine_puffin.utilities.imports import module_exists
from scine_puffin.tests.testcases import skip_without, JobTestCase

from ..db_setup import add_calculation, add_structure
from ..resources import resource_path

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db


class _Dummy(ReactJob):

    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        return True


class ScineReactTest(unittest.TestCase):

    def _split_atoms(self, atoms, compound_map):
        import scine_utilities as utils
        compound_set = list(set(compound_map[:]))
        split_structures = []
        for c in compound_set:
            tmp_index = np.where(compound_map == c)[0]
            tmp_atoms = utils.AtomCollection()
            for i in tmp_index:
                tmp_atoms.push_back(atoms[i])
            split_structures.append(tmp_atoms)
        return split_structures

    def test_custom_round(self):
        react_job = _Dummy()
        assert (react_job._custom_round(2.45) == (2.0, False))
        assert (react_job._custom_round(2.45, 0.4) == (3.0, True))
        assert (react_job._custom_round(-2.45) == (-2.0, False))
        assert (react_job._custom_round(-2.45, 0.45) == (-3.0, True))
        assert (react_job._custom_round(-2.45, 0.46) == (-2.0, False))

    def test_distribute_charge(self):
        # import Job
        react_job = _Dummy()
        assert (react_job._distribute_charge(0, [-1, 0], [-0.42, 0.2]) == [0, 0])  # Remove electron
        assert (react_job._distribute_charge(0, [-1, 1], [-0.42, 0.43]) == [-1, 1])  # Pass without change
        assert (react_job._distribute_charge(1, [-1, 1], [-0.42, 0.43]) == [0, 1])  # Remove electron
        assert (react_job._distribute_charge(-1, [-1, 1], [-0.42, 0.43]) == [-1, 0])  # Add electron

        assert (react_job._distribute_charge(0, [-1, 0, 0], [-0.42, 0.3, 0.2]) == [0, 0, 0])  # Remove electron
        assert (react_job._distribute_charge(1, [-1, 0, 0], [-0.42, 0.3, 0.2])
                == [0, 1, 0])  # Remove electrons, charge at 1
        assert (react_job._distribute_charge(-1, [0, 0, 1], [-0.42, 0.3, 0.2])
                == [-1, 0, 0])  # Add electrons, charge at 0

        assert (react_job._distribute_charge(0, [-1, 0, 0, 0],
                [-0.8, 0.05, 0.1, 0.45]) == [-1, 0, 0, 1])  # Remove electron at 3
        assert (react_job._distribute_charge(1, [-1, 0, 0, 0], [-0.8, 0.05,
                0.1, 0.45]) == [0, 0, 0, 1])  # Remove electrons at 0 and 3
        assert (react_job._distribute_charge(-2, [-1, 0, 0, 0],
                [-1.3, 0.05, 0.1, 0.45]) == [-2, 0, 0, 0])  # Add electron at 0

    @skip_without("utilities")
    def test_two_molecules_in_complex(self):
        import scine_utilities as utils
        # Load test case
        total_charge = 0
        positions = [[-2.091417412302329, -1.404388718237768, 0.6029258660694673, ],
                     [-2.816756527689213, -1.3231929424307944, 2.3162727689619786, ],
                     [-0.769035793823446, 0.8329482716606773, -3.8768877780419366, ],
                     [-3.703318756505382, 2.005585668744448, -0.8974914033853223, ],
                     [-5.618159670890759, 2.259089229026567, 2.1310261412101372, ],
                     [2.787061473021499, 1.6632939643599012, 0.2488945227380528, ],
                     [2.925544159007961, 2.2718128647175626, 1.9998956415464775, ],
                     [0.31572125387546973, -1.3201494353154004, -3.8157417425209093, ],
                     [2.7617385506121117, -2.153256226612508, 0.40935040851094545, ],
                     [6.208622724694082, -2.8317426759127162, 0.881755574911138, ],
                     ]
        elements = [
            utils.ElementType.O,
            utils.ElementType.H,
            utils.ElementType.O,
            utils.ElementType.I,
            utils.ElementType.O,
            utils.ElementType.O,
            utils.ElementType.H,
            utils.ElementType.O,
            utils.ElementType.I,
            utils.ElementType.O]
        compound_map = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1])
        partial_charges = [-0.65685, 0.09501, -0.44847, 1.61043, -
                           0.8496, -0.52153, 0.11467, -0.32257, 1.77924, -0.80032]

        atoms = utils.AtomCollection(elements, positions)
        split_structures = self._split_atoms(atoms, compound_map)

        react_job = _Dummy()
        react_job.settings['sp']['expect_charge_separation'] = True
        react_job.settings['sp']['charge_separation_threshold'] = 0.6
        # Test: no charge assigned
        c, e, r = react_job._integrate_charges(compound_map, partial_charges, split_structures, total_charge)
        assert (c == [0, 0])
        assert (e == [86, 70])
        ref_r = [-0.57205, 0.57206]
        for i in range(len(r)):
            self.assertAlmostEqual(r[i], ref_r[i], delta=1e-6)
        # Test: charges assigned
        react_job.settings['sp']['charge_separation_threshold'] = 0.4
        c2, e2, r2 = react_job._integrate_charges(compound_map, partial_charges, split_structures, total_charge)
        assert (c2 == [-1, 1])
        assert (e2 == [87, 69])
        ref_r2 = [0.42795, -0.42794]
        for i in range(len(r2)):
            self.assertAlmostEqual(r2[i], ref_r2[i], delta=1e-6)

    @skip_without("utilities")
    def test_three_molecules_in_complex(self):
        import scine_utilities as utils
        # Load test case
        total_charge = 0

        total_charge = 0
        positions = [[-3.8285425220374076, 2.501569916268017, -1.0942637578410739, ],
                     [-3.499223402721347, 3.7924089929727063, 0.200112785510394, ],
                     [-3.8864676468204076, -2.621016392091094, -2.3699658667477745, ],
                     [-2.452045940530779, -0.8971756379752788, 0.3166879043898848, ],
                     [0.03063393916145471, -3.7125817375070587, 1.4120257924996842, ],
                     [0.07616066380648903, 2.367289623435106, 3.2738556814906894, ],
                     [1.0713516991177325, -4.346031681596603, 0.007480457031388776, ],
                     [5.146936751280798, 0.5610680307465189, 1.165516524172823, ],
                     [2.9459360374464914, 3.318316167686782, 1.359894627469893, ],
                     [1.3302599640812325, 0.3101084857682949, -2.2347737364826714, ],
                     [3.0650004572157394, -1.2739557677073863, -2.0365704114932495, ],
                     ]
        elements = [
            utils.ElementType.O,
            utils.ElementType.H,
            utils.ElementType.O,
            utils.ElementType.I,
            utils.ElementType.O,
            utils.ElementType.O,
            utils.ElementType.H,
            utils.ElementType.O,
            utils.ElementType.I,
            utils.ElementType.O,
            utils.ElementType.O,
        ]

        compound_map = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 2])
        partial_charges = [-0.53305, 0.07042, -0.74429, 1.95638, -
                           0.52344, -0.87023, 0.07851, -0.67674, 1.81967, -0.44495, -0.13228]
        atoms = utils.AtomCollection(elements, positions)
        split_structures = self._split_atoms(atoms, compound_map)

        react_job = _Dummy()
        react_job.settings['sp']['expect_charge_separation'] = True
        react_job.settings['sp']['charge_separation_threshold'] = 0.6
        # Test: no charge assigned
        c, e, r = react_job._integrate_charges(compound_map, partial_charges, split_structures, total_charge)
        assert (c == [0, 0, 0])
        assert (e == [79, 69, 16])
        ref_r = [0.30453, 0.2727, -0.57723]
        for i in range(len(r)):
            self.assertAlmostEqual(r[i], ref_r[i], delta=1e-6)

        react_job.settings['sp']['charge_separation_threshold'] = 0.5
        # Test: charge assigend
        c2, e2, r2 = react_job._integrate_charges(compound_map, partial_charges, split_structures, total_charge)
        assert (c2 == [1, 0, -1])
        assert (e2 == [78, 69, 17])
        ref_r2 = [-0.69547, 0.2727, 0.42277]
        for i in range(len(r2)):
            self.assertAlmostEqual(r2[i], ref_r2[i], delta=1e-6)


class _DummyJob(ReactJob):

    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Test React Job"
        self.exploration_key = "nt"
        opt_defaults = {
            "convergence_max_iterations": 500,
        }
        self.settings = {
            **self.settings,
            self.opt_key: opt_defaults
        }

        self.settings[self.propensity_key]['check'] = 1
        self.settings[self.single_point_key]['expect_charge_separation'] = True
        self.settings[self.single_point_key]['charge_separation_threshold'] = 0.5

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "readuct", "utils"]


class ScineReactJobTest(JobTestCase):

    @skip_without("utilities", "readuct", "xtb_wrapper")
    def test_analyze_side_reject_exhaustive_product_decomposition(self):
        import scine_utilities as utils

        # # # Load system into calculator
        xyz_name = os.path.join(resource_path(), "iodine_in_water.xyz")
        test_model_method = "gfn2"
        calculator_settings = utils.ValueCollection({
            'program': 'xtb',
            'method': test_model_method,
            'spin_multiplicity': 1,
            'molecular_charge': 0,
            'spin_mode': 'restricted_open_shell',
            'self_consistence_criterion': 1e-07,
            'electronic_temperature': 300.0,
            'max_scf_iterations': 100,
            'solvent': 'water',
            'solvation': 'gbsa',
        })
        test_calc = utils.core.load_system_into_calculator(
            xyz_name,
            test_model_method,
            **calculator_settings)

        test_model = db.Model(test_model_method, test_model_method, "")
        test_model.solvation = "gbsa"
        test_model.solvent = "water"

        dummy_structure = add_structure(self.manager, xyz_name, db.Label.MINIMUM_OPTIMIZED, 0, 1, test_model)
        dummy_calculation = add_calculation(self.manager, model=test_model, job=db.Job("dummy_test"),
                                            structures=[dummy_structure.id()])

        config = self.get_configuration()
        react_job = _DummyJob()
        react_job.configure_run(self.manager, dummy_calculation, config)
        react_job.prepare(config["daemon"]["job_dir"], dummy_calculation.id())
        react_job.systems["test_irc"] = test_calc

        react_job.settings[react_job.opt_key]['geoopt_coordinate_system'] = "cartesianWithoutRotTrans"

        with calculation_context(react_job):
            (
                test_graph,
                test_charges,
                test_decision_lists,
                test_names
            ) = react_job.analyze_side("test_irc", 0, "test_forward", calculator_settings)

        assert test_graph is None
        assert test_charges is None
        assert test_decision_lists is None
        assert test_names is None

    @skip_without("utilities", "readuct", "molassembler", "xtb_wrapper")
    def test_analyze_side_allow_exhaustive_product_decomposition(self):
        import scine_utilities as utils
        import scine_molassembler as masm

        # # # Load system into calculator
        xyz_name = os.path.join(resource_path(), "iodine_in_water.xyz")
        test_model_method = "gfn2"
        calculator_settings = utils.ValueCollection({
            'program': 'xtb',
            'method': test_model_method,
            'spin_multiplicity': 1,
            'molecular_charge': 0,
            'spin_mode': 'restricted_open_shell',
            'self_consistence_criterion': 1e-07,
            'electronic_temperature': 300.0,
            'max_scf_iterations': 100,
            'solvent': 'water',
            'solvation': 'gbsa',
        })
        test_calc = utils.core.load_system_into_calculator(
            xyz_name,
            test_model_method,
            **calculator_settings)

        test_model = db.Model(test_model_method, test_model_method, "")
        test_model.solvation = "gbsa"
        test_model.solvent = "water"

        dummy_structure = add_structure(self.manager, xyz_name, db.Label.MINIMUM_OPTIMIZED, 0, 1, test_model)
        dummy_calculation = add_calculation(self.manager, model=test_model, job=db.Job("dummy_test"),
                                            structures=[dummy_structure.id()])

        config = self.get_configuration()
        react_job = _DummyJob()
        react_job.configure_run(self.manager, dummy_calculation, config)
        react_job.prepare(config["daemon"]["job_dir"], dummy_calculation.id())
        react_job.systems["test_irc"] = test_calc

        react_job.settings[react_job.opt_key]['geoopt_coordinate_system'] = "cartesianWithoutRotTrans"
        react_job.settings[react_job.job_key]["allow_exhaustive_product_decomposition"] = True

        with calculation_context(react_job):
            (
                test_graph,
                test_charges,
                test_decision_lists,
                test_names
            ) = react_job.analyze_side("test_irc", 0, "test_forward", calculator_settings)

        assert len(test_names) == 33  # 32 water molecules + 1 iodine
        assert len(set(test_names)) == 2  # 2 type of molecules
        assert len(test_charges) == 33
        assert len(test_decision_lists) == 33
        ref_graph_1 = "o2FjD2FnomFFgYMAAQBhWoIZP7UZP7VhdoMCAAE="  # Iodine
        ref_graph_2 = "pGFhgaVhYQBhYwJhb4GCAAFhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAgAB"
        assert masm.JsonSerialization.equal_molecules(test_graph.split(";")[0], ref_graph_1)
        for tmp_graph in test_graph.split(";")[1:]:
            assert masm.JsonSerialization.equal_molecules(tmp_graph, ref_graph_2)
