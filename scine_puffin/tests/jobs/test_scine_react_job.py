#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""
import numpy as np
import unittest


class ScineReactJobTest(unittest.TestCase):

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
        from scine_puffin.jobs.templates.scine_react_job import ReactJob
        react_job = ReactJob()
        assert (react_job._custom_round(2.45) == 2.0)
        assert (react_job._custom_round(2.45, 0.4) == 3.0)
        assert (react_job._custom_round(-2.45) == -2.0)
        assert (react_job._custom_round(-2.45, 0.45) == -3.0)
        assert (react_job._custom_round(-2.45, 0.46) == -2.0)

    def test_distribute_charge(self):
        # import Job
        from scine_puffin.jobs.templates.scine_react_job import ReactJob
        react_job = ReactJob()
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

    def test_two_molecules_in_complex(self):
        import scine_utilities as utils
        from scine_puffin.jobs.templates.scine_react_job import ReactJob
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

        react_job = ReactJob()
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

    def test_three_molecules_in_complex(self):
        import scine_utilities as utils
        from scine_puffin.jobs.templates.scine_react_job import ReactJob
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

        react_job = ReactJob()
        react_job.settings['sp']['expect_charge_separation'] = True
        react_job.settings['sp']['charge_separation_threshold'] = 0.4
        # Test: no charge assigned
        c, e, r = react_job._integrate_charges(compound_map, partial_charges, split_structures, total_charge)
        assert (c == [0, 0, 0])
        assert (e == [79, 69, 16])
        ref_r = [0.30453, 0.2727, -0.57723]
        for i in range(len(r)):
            self.assertAlmostEqual(r[i], ref_r[i], delta=1e-6)

        react_job.settings['sp']['charge_separation_threshold'] = 0.3
        # Test: charge assigend
        c2, e2, r2 = react_job._integrate_charges(compound_map, partial_charges, split_structures, total_charge)
        assert (c2 == [1, 0, -1])
        assert (e2 == [78, 69, 17])
        ref_r2 = [-0.69547, 0.2727, 0.42277]
        for i in range(len(r2)):
            self.assertAlmostEqual(r2[i], ref_r2[i], delta=1e-6)
