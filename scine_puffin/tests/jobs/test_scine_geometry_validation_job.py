#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import numpy as np
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


class ScineGeometryValidationJobTest(JobTestCase):

    @skip_without('database', 'readuct')
    def test_non_valid_minimum(self):
        from scine_puffin.jobs.scine_geometry_validation import ScineGeometryValidation
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water_distorted.xyz")
        structure = add_structure(self.manager, water, db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph(
            "masm_cbor_graph",
            "pGFhgaVhYQBhYwJhb4GCAAFhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQIB")
        structure.set_graph("masm_decision_list", "")

        model = db.Model('dftb3', 'dftb3', '')
        model.program = "sparrow"
        job = db.Job('scine_geometry_validation')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineGeometryValidation()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        success = job.run(self.manager, calculation, config)
        assert not success

        # Check comment of calculation
        ref_comment = "\nError: Scine Geometry Validation Job failed with message:\n" + \
            "Structure could not be validated to be a minimum. Hessian information is stored anyway."
        assert calculation.get_comment() == ref_comment

        # Check results
        assert calculation.get_status() == db.Status.FAILED
        assert structure.has_property("electronic_energy")
        assert structure.has_property("hessian")
        assert structure.has_property("normal_modes")
        assert structure.has_property("frequencies")
        assert structure.has_property("gibbs_free_energy")
        assert structure.has_property("gibbs_energy_correction")
        energy_props = structure.get_properties("electronic_energy")
        hessian_props = structure.get_properties("hessian")
        normal_modes_props = structure.get_properties("normal_modes")
        frequencies_props = structure.get_properties("frequencies")
        gibbs_free_energy_props = structure.get_properties("gibbs_free_energy")
        gibbs_energy_correction_props = structure.get_properties("gibbs_energy_correction")

        assert len(energy_props) == 1
        assert len(hessian_props) == 1
        assert len(normal_modes_props) == 1
        assert len(frequencies_props) == 1
        assert len(gibbs_free_energy_props) == 1
        assert len(gibbs_energy_correction_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 6
        assert energy_props[0] in results.property_ids
        assert hessian_props[0] in results.property_ids
        assert frequencies_props[0] in results.property_ids
        assert normal_modes_props[0] in results.property_ids
        assert gibbs_free_energy_props[0] in results.property_ids
        assert gibbs_energy_correction_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -3.8236653931331333, delta=1e-1)
        # Hessian
        ref_hessian = np.array([
            [1.32061502e+00, 1.55335561e+00, -9.74047400e-08, -3.39090813e-01, 9.51106255e-02, 3.43230048e-07,
             -9.81525023e-01, -1.64846785e+00, 3.31603713e-07, ],
            [1.55335561e+00, 1.26247487e+00, -8.85166508e-08, 5.39764221e-02, 1.95699661e-02, -7.51007690e-08,
             -1.60733254e+00, -1.28204482e+00, -7.55222799e-08, ],
            [-9.74047400e-08, -8.85166508e-08, -4.20492614e-01, -1.33637278e-06, 1.43254119e-06, -5.61467929e-02,
             1.43377752e-06, -1.34402454e-06, 4.76637885e-01, ],
            [-3.39090813e-01, 5.39764221e-02, -1.33637278e-06, 6.61626880e-01, -4.82760057e-01, -2.63388409e-07,
             -3.22535721e-01, 4.28784515e-01, -2.69506583e-07, ],
            [9.51106255e-02, 1.95699661e-02, 1.43254119e-06, -4.82760057e-01, 3.05309881e-01, 3.24476937e-09,
             3.87647711e-01, -3.24879754e-01, 1.75076689e-08, ],
            [3.43230048e-07, -7.51007690e-08, -5.61467929e-02, -2.63388409e-07, 3.24476937e-09, -1.58028581e-01,
             -7.98416446e-08, 7.18560128e-08, 2.14175438e-01, ],
            [-9.81525023e-01, -1.60733254e+00, 1.43377752e-06, -3.22535721e-01, 3.87647711e-01, -7.98416446e-08,
             1.30406122e+00, 1.21968557e+00, -6.20971299e-08, ],
            [-1.64846785e+00, -1.28204482e+00, -1.34402454e-06, 4.28784515e-01, -3.24879754e-01, 7.18560128e-08,
             1.21968557e+00, 1.60692447e+00, 5.80145901e-08, ],
            [3.31603713e-07, -7.55222799e-08, 4.76637885e-01, -2.69506583e-07, 1.75076689e-08, 2.14175438e-01,
             -6.20971299e-08, 5.80145901e-08, -6.90811867e-01, ],
        ])
        hessian_prop = db.DenseMatrixProperty(hessian_props[0])
        hessian_prop.link(properties)
        hessian = hessian_prop.get_data()
        assert hessian.shape == (9, 9)
        assert np.allclose(ref_hessian, hessian, atol=1e-1)
        # Normal modes
        ref_normal_modes = np.array([
            [-6.57555073e-02, 1.26620092e-02, 4.40344617e-02, ],
            [1.75882705e-03, -4.50720615e-03, 4.28132636e-02, ],
            [6.25238992e-18, -4.82124061e-18, 2.08537396e-18, ],
            [6.80766002e-01, -6.84858319e-01, -6.00657576e-02, ],
            [3.90929440e-01, 4.12321086e-01, 4.06809503e-02, ],
            [2.59753169e-16, -1.21379778e-17, -1.53607653e-17, ],
            [3.63010525e-01, 4.83866667e-01, -6.38919610e-01, ],
            [-4.18848354e-01, -3.40775505e-01, -7.20281511e-01, ],
            [-4.33840517e-16, 1.20815437e-17, 2.94634947e-17, ],
        ])
        normal_mode_prop = db.DenseMatrixProperty(normal_modes_props[0])
        normal_mode_prop.link(properties)
        normal_modes = normal_mode_prop.get_data()
        assert normal_modes.shape == (9, 3)
        assert np.allclose(np.abs(ref_normal_modes), np.abs(normal_modes), atol=1e-1)

        # Frequencies
        ref_frequencies = [-0.00054616, 0.00440581, 0.00628707]
        frequencies_prop = db.VectorProperty(frequencies_props[0])
        frequencies_prop.link(properties)
        frequencies = frequencies_prop.get_data()
        assert len(frequencies) == 3
        self.assertAlmostEqual(ref_frequencies[0], frequencies[0], delta=1e-1)
        self.assertAlmostEqual(ref_frequencies[1], frequencies[1], delta=1e-1)
        self.assertAlmostEqual(ref_frequencies[2], frequencies[2], delta=1e-1)

        # Gibbs free energy
        gibbs_free_energy_prop = db.NumberProperty(gibbs_free_energy_props[0])
        gibbs_free_energy_prop.link(properties)
        gibbs_free_energy = gibbs_free_energy_prop.get_data()
        self.assertAlmostEqual(gibbs_free_energy, -3.807695814121228, delta=1e-1)

        # Gibbs energy correction
        gibbs_energy_correction_prop = db.NumberProperty(gibbs_energy_correction_props[0])
        gibbs_energy_correction_prop.link(properties)
        gibbs_energy_correction = gibbs_energy_correction_prop.get_data()
        self.assertAlmostEqual(gibbs_energy_correction, 0.01596957901190521, delta=1e-5)

    @skip_without('database', 'readuct')
    def test_valid_minimum(self):
        from scine_puffin.jobs.scine_geometry_validation import ScineGeometryValidation
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph(
            "masm_cbor_graph",
            "pGFhgaVhYQBhYwJhb4GCAAFhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAgAA")
        structure.set_graph("masm_decision_list", "")

        model = db.Model('dftb3', 'dftb3', '')
        model.program = "sparrow"
        job = db.Job('scine_geometry_validation')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineGeometryValidation()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        success = job.run(self.manager, calculation, config)
        assert success

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        assert structure.has_property("bond_orders")
        assert structure.has_property("hessian")
        assert structure.has_property("normal_modes")
        assert structure.has_property("frequencies")
        assert structure.has_property("gibbs_free_energy")
        assert structure.has_property("gibbs_energy_correction")
        energy_props = structure.get_properties("electronic_energy")
        bond_props = structure.get_properties("bond_orders")
        hessian_props = structure.get_properties("hessian")
        normal_modes_props = structure.get_properties("normal_modes")
        frequencies_props = structure.get_properties("frequencies")
        gibbs_free_energy_props = structure.get_properties("gibbs_free_energy")
        gibbs_energy_correction_props = structure.get_properties("gibbs_energy_correction")

        assert len(energy_props) == 1
        assert len(bond_props) == 1
        assert len(hessian_props) == 1
        assert len(normal_modes_props) == 1
        assert len(frequencies_props) == 1
        assert len(gibbs_free_energy_props) == 1
        assert len(gibbs_energy_correction_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 7
        assert energy_props[0] in results.property_ids
        assert bond_props[0] in results.property_ids
        assert hessian_props[0] in results.property_ids
        assert frequencies_props[0] in results.property_ids
        assert normal_modes_props[0] in results.property_ids
        assert gibbs_free_energy_props[0] in results.property_ids
        assert gibbs_energy_correction_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -4.061143327, delta=1e-1)
        # Bond orders
        bond_prop = db.SparseMatrixProperty(bond_props[0])
        bond_prop.link(properties)
        bond = bond_prop.get_data()
        assert bond.shape == (3, 3)
        self.assertAlmostEqual(bond[0, 1], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bond[0, 2], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bond[1, 0], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bond[2, 0], +0.881347660, delta=1e-1)
        self.assertAlmostEqual(bond[0, 0], +0.000000000)
        self.assertAlmostEqual(bond[1, 1], +0.000000000)
        self.assertAlmostEqual(bond[2, 2], +0.000000000)
        self.assertAlmostEqual(bond[2, 1], +0.003157004, delta=1e-1)
        self.assertAlmostEqual(bond[1, 2], +0.003157004, delta=1e-1)
        # Hessian
        ref_hessian = np.array(
            [[4.33647320e-01, -1.92895860e-02, 6.74264637e-09, -3.82458865e-01, -2.36584777e-02, -2.82825003e-07,
              -5.11883523e-02, 4.29483194e-02, 2.77110166e-07],
             [-1.92895860e-02, 4.33647320e-01, 6.74262344e-09, 4.30982652e-02, -5.13382980e-02,
                 2.77110141e-07, -2.38084235e-02, -3.82308920e-01, -2.82824980e-07],
             [6.74264637e-09, 6.74262344e-09, 5.50602001e-02, -5.27561104e-09, -1.46703795e-09, -
                 2.75300925e-02, -1.46703552e-09, -5.27558506e-09, -2.75300925e-02],
             [-3.82458865e-01, 4.30982652e-02, -5.27561104e-09, 3.92889518e-01, -2.32293721e-02,
                 2.55186625e-07, -1.04307157e-02, -1.98690895e-02, -2.50496798e-07],
             [-2.36584777e-02, -5.13382980e-02, -1.46703795e-09, -2.32293721e-02,
                 6.16190275e-02, -1.38730691e-08, 4.68877925e-02, -1.02807711e-02, 1.48980846e-08],
             [-2.82825003e-07, 2.77110141e-07, -2.75300925e-02, 2.55186625e-07, -1.38730691e-08,
                 1.47316279e-02, 2.76383853e-08, -2.63237086e-07, 1.27984576e-02],
             [-5.11883523e-02, -2.38084235e-02, -1.46703552e-09, -1.04307157e-02, 4.68877925e-02,
                 2.76383853e-08, 6.16190282e-02, -2.30794283e-02, -2.66133757e-08],
             [4.29483194e-02, -3.82308920e-01, -5.27558506e-09, -1.98690895e-02, -
                 1.02807711e-02, -2.63237086e-07, -2.30794283e-02, 3.92589630e-01, 2.67926909e-07],
             [2.77110166e-07, -2.82824980e-07, -2.75300925e-02, -2.50496798e-07, 1.48980846e-08, 1.27984576e-02,
              -2.66133757e-08, 2.67926909e-07, 1.47316269e-02]])
        hessian_prop = db.DenseMatrixProperty(hessian_props[0])
        hessian_prop.link(properties)
        hessian = hessian_prop.get_data()
        assert hessian.shape == (9, 9)
        assert np.allclose(ref_hessian, hessian, atol=1e-1)
        # Normal modes
        ref_normal_modes = np.array(
            [[-4.47497118e-02, -3.85208122e-02, -4.31593322e-02],
             [-4.47122166e-02, -3.88386563e-02, 4.29126332e-02],
             [-8.49613600e-18, 5.24295770e-20, 2.59912639e-19],
             [2.95571230e-02, 6.81190413e-01, 6.85647964e-01],
             [6.80781838e-01, -6.97265040e-02, -5.54050376e-04],
             [1.70377797e-16, 1.60989274e-16, 1.17463267e-16],
             [6.80781838e-01, -6.97265040e-02, -5.54050376e-04],
             [2.89619394e-02, 6.86235743e-01, -6.80623863e-01],
             [-2.83048332e-16, -1.53196255e-16, -2.05063421e-16]]
        )
        normal_mode_prop = db.DenseMatrixProperty(normal_modes_props[0])
        normal_mode_prop.link(properties)
        normal_modes = normal_mode_prop.get_data()
        assert normal_modes.shape == (9, 3)
        assert np.allclose(np.abs(ref_normal_modes), np.abs(normal_modes), atol=1e-1)

        # Frequencies
        ref_frequencies = [0.00124259, 0.00233505, 0.00246372]
        frequencies_prop = db.VectorProperty(frequencies_props[0])
        frequencies_prop.link(properties)
        frequencies = frequencies_prop.get_data()
        assert len(frequencies) == 3
        self.assertAlmostEqual(ref_frequencies[0], frequencies[0], delta=1e-1)
        self.assertAlmostEqual(ref_frequencies[1], frequencies[1], delta=1e-1)
        self.assertAlmostEqual(ref_frequencies[2], frequencies[2], delta=1e-1)

        # Gibbs free energy
        gibbs_free_energy_prop = db.NumberProperty(gibbs_free_energy_props[0])
        gibbs_free_energy_prop.link(properties)
        gibbs_free_energy = gibbs_free_energy_prop.get_data()
        self.assertAlmostEqual(gibbs_free_energy, -4.060596474205113, delta=1e-1)

        # Gibbs energy correction
        gibbs_energy_correction_prop = db.NumberProperty(gibbs_energy_correction_props[0])
        gibbs_energy_correction_prop.link(properties)
        gibbs_energy_correction = gibbs_energy_correction_prop.get_data()
        self.assertAlmostEqual(gibbs_energy_correction, 0.0005468527260594769, delta=1e-5)

    @skip_without('database', 'readuct')
    def test_fail_to_optimize_non_valid_minimum(self):
        # fails because of different graph
        from scine_puffin.jobs.scine_geometry_validation import ScineGeometryValidation
        import scine_database as db

        # Setup DB for calculation
        h2o2 = os.path.join(resource_path(), "h2o2_distorted.xyz")
        ref_graph = "pGFhgaVhYQBhYwNhb4GDAAECYXKjYWyDgQCBAYECYmxygoIAAYECYXOCgg" + \
                    "ABgQJhcwNhYw9hZ6JhRYODAAMAgwEDAIMCAwBhWoQBAQgIYXaDAQIB"
        structure = add_structure(self.manager, h2o2, db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", ref_graph)
        structure.set_graph("masm_decision_list", "")

        model = db.Model('dftb3', 'dftb3', '')
        model.program = "sparrow"
        job = db.Job('scine_geometry_validation')
        settings = {
            "optimization_attempts": 2
        }

        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineGeometryValidation()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        success = job.run(self.manager, calculation, config)
        assert not success
        # Check comment of calculation
        ref_comment = "Scine Geometry Validation Job: End structure does not match starting structure." + \
            "\nError: Scine Geometry Validation Job failed with message:" + \
            "\nStructure could not be validated to be a minimum. Hessian information is stored anyway."
        assert calculation.get_comment() == ref_comment

        # Check results
        assert calculation.get_status() == db.Status.FAILED
        assert structure.has_property("electronic_energy")
        assert structure.has_property("hessian")
        assert structure.has_property("normal_modes")
        assert structure.has_property("frequencies")
        assert structure.has_property("gibbs_free_energy")
        assert structure.has_property("gibbs_energy_correction")
        energy_props = structure.get_properties("electronic_energy")
        hessian_props = structure.get_properties("hessian")
        normal_modes_props = structure.get_properties("normal_modes")
        frequencies_props = structure.get_properties("frequencies")
        gibbs_free_energy_props = structure.get_properties("gibbs_free_energy")
        gibbs_energy_correction_props = structure.get_properties("gibbs_energy_correction")

        assert len(energy_props) == 1
        assert len(hessian_props) == 1
        assert len(normal_modes_props) == 1
        assert len(frequencies_props) == 1
        assert len(gibbs_free_energy_props) == 1
        assert len(gibbs_energy_correction_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 6
        assert energy_props[0] in results.property_ids
        assert hessian_props[0] in results.property_ids
        assert frequencies_props[0] in results.property_ids
        assert normal_modes_props[0] in results.property_ids
        assert gibbs_free_energy_props[0] in results.property_ids
        assert gibbs_energy_correction_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -7.168684600560611, delta=1e-1)
        # Normal modes
        ref_normal_modes = np.array([
            [-4.31688799e-02, -6.91523883e-03, -1.82492631e-02, -3.86771134e-02, -3.45645862e-02, 3.81068441e-03, ],
            [-1.65111999e-03, 2.83228734e-03, 2.10319980e-02, -4.31722218e-02, 1.36502269e-03, -5.94798269e-02, ],
            [-2.99343119e-02, 1.78068975e-01, -2.59408246e-04, -1.58605276e-02, -1.23374087e-02, 2.33946275e-03, ],
            [4.03868933e-02, -1.78581009e-03, -4.13741993e-03, -6.89669850e-03, -1.88649585e-02, 9.27050089e-04, ],
            [-3.48727166e-03, -6.65257470e-03, -4.16577386e-02, 5.81676804e-03, 2.25155821e-03, -2.87158874e-04, ],
            [-2.68061138e-02, -1.69761922e-01, 5.77588045e-02, -5.30648270e-03, 1.05427368e-02, -9.30432318e-04, ],
            [-2.90044217e-02, 7.59482476e-02, 5.02213061e-02, 1.25262116e-01, 9.65015716e-01, -5.88417133e-02, ],
            [7.31645599e-02, 6.21687101e-02, 3.05135914e-01, 5.98158280e-01, -1.16896572e-01, -1.63631824e-02, ],
            [9.47125787e-01, -4.54672494e-03, -7.34029615e-02, -1.01582077e-01, 3.33256908e-02, -7.34742518e-03, ],
            [7.31645599e-02, 6.21687101e-02, 3.05135914e-01, 5.98158280e-01, -1.16896572e-01, -1.63631824e-02, ],
            [8.40020643e-03, -1.52700207e-03, 2.22688135e-02, -5.19280259e-03, 5.94884190e-02, 9.65080322e-01, ],
            [-4.64510457e-02, -1.27316095e-01, -8.39319373e-01, 4.37578701e-01, -4.83778945e-03, -1.50189582e-02, ],
        ])

        normal_mode_prop = db.DenseMatrixProperty(normal_modes_props[0])
        normal_mode_prop.link(properties)
        normal_modes = normal_mode_prop.get_data()
        assert normal_modes.shape == (12, 6)
        assert np.allclose(np.abs(ref_normal_modes), np.abs(normal_modes), atol=1e-1)

        # Frequencies
        ref_frequencies = [-0.00117588, 0.00064638, 0.00083982, 0.00106649, 0.00156229, 0.00239611]
        frequencies_prop = db.VectorProperty(frequencies_props[0])
        frequencies_prop.link(properties)
        frequencies = frequencies_prop.get_data()
        assert len(frequencies) == 6
        for i, ref in enumerate(ref_frequencies):
            self.assertAlmostEqual(ref, frequencies[i], delta=1e-1)

        # Gibbs free energy
        gibbs_free_energy_prop = db.NumberProperty(gibbs_free_energy_props[0])
        gibbs_free_energy_prop.link(properties)
        gibbs_free_energy = gibbs_free_energy_prop.get_data()
        self.assertAlmostEqual(gibbs_free_energy, -7.170441957688447, delta=1e-1)

        # Gibbs energy correction
        gibbs_energy_correction_prop = db.NumberProperty(gibbs_energy_correction_props[0])
        gibbs_energy_correction_prop.link(properties)
        gibbs_energy_correction = gibbs_energy_correction_prop.get_data()
        self.assertAlmostEqual(gibbs_energy_correction, -0.001757357127836201, delta=1e-5)

        # Graph
        assert structure.get_graph("masm_cbor_graph") == ref_graph

    @skip_without('database', 'readuct')
    @pytest.mark.filterwarnings("ignore:.+The structure had a graph already")
    def test_optimize_non_valid_minimum(self):
        # fails because of different graph
        from scine_puffin.jobs.scine_geometry_validation import ScineGeometryValidation
        import scine_database as db
        import scine_utilities as utils

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water_distorted_2.xyz")
        ref_graph = "pGFhgaVhYQBhYwJhb4GCAAFhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAgAA"
        structure = add_structure(self.manager, water, db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", ref_graph)
        structure.set_graph("masm_decision_list", "")

        model = db.Model('dftb3', 'dftb3', '')
        model.program = "sparrow"
        job = db.Job('scine_geometry_validation')
        settings = {
            "optimization_attempts": 2
        }

        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineGeometryValidation()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        success = job.run(self.manager, calculation, config)
        assert success

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert structure.has_property("electronic_energy")
        assert structure.has_property("bond_orders")
        assert structure.has_property("hessian")
        assert structure.has_property("normal_modes")
        assert structure.has_property("frequencies")
        assert structure.has_property("gibbs_free_energy")
        assert structure.has_property("gibbs_energy_correction")
        assert structure.has_property("position_shift")
        energy_props = structure.get_properties("electronic_energy")
        bond_props = structure.get_properties("bond_orders")
        hessian_props = structure.get_properties("hessian")
        normal_modes_props = structure.get_properties("normal_modes")
        frequencies_props = structure.get_properties("frequencies")
        gibbs_free_energy_props = structure.get_properties("gibbs_free_energy")
        gibbs_energy_correction_props = structure.get_properties("gibbs_energy_correction")
        position_shift_props = structure.get_properties("position_shift")

        assert len(energy_props) == 1
        assert len(bond_props) == 1
        assert len(hessian_props) == 1
        assert len(normal_modes_props) == 1
        assert len(frequencies_props) == 1
        assert len(gibbs_free_energy_props) == 1
        assert len(gibbs_energy_correction_props) == 1
        assert len(position_shift_props) == 1
        results = calculation.get_results()
        assert len(results.property_ids) == 8
        assert energy_props[0] in results.property_ids
        assert bond_props[0] in results.property_ids
        assert hessian_props[0] in results.property_ids
        assert frequencies_props[0] in results.property_ids
        assert normal_modes_props[0] in results.property_ids
        assert gibbs_free_energy_props[0] in results.property_ids
        assert gibbs_energy_correction_props[0] in results.property_ids
        assert position_shift_props[0] in results.property_ids

        # Check generated properties
        # Energy
        properties = self.manager.get_collection("properties")
        energy = db.NumberProperty(energy_props[0])
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -4.071575644461411, delta=1e-1)
        # Bond orders
        bond_prop = db.SparseMatrixProperty(bond_props[0])
        bond_prop.link(properties)
        bond = bond_prop.get_data()
        assert bond.shape == (3, 3)
        self.assertAlmostEqual(bond[1, 0], +0.8541060592278684, delta=1e-1)
        self.assertAlmostEqual(bond[2, 0], +0.8541060835647227, delta=1e-1)
        self.assertAlmostEqual(bond[0, 1], +0.8541060592278684, delta=1e-1)
        self.assertAlmostEqual(bond[0, 2], +0.8541060835647227, delta=1e-1)
        self.assertAlmostEqual(bond[2, 1], +0.015945768676323538, delta=1e-1)
        self.assertAlmostEqual(bond[1, 2], +0.015945768676323538, delta=1e-1)
        self.assertAlmostEqual(bond[0, 0], +0.000000000)
        self.assertAlmostEqual(bond[1, 1], +0.000000000)
        self.assertAlmostEqual(bond[2, 2], +0.000000000)
        # Hessian
        ref_hessian = np.array([
            [6.81547839e-01, -1.01826427e-01, 2.01519849e-10, -4.78354633e-01, -1.53645723e-01, -3.88607902e-07,
             -2.03194076e-01, 2.55471871e-01, 3.91793089e-07, ],
            [-1.01826427e-01, 4.23459677e-01, 6.58626116e-10, -9.28932439e-02, -7.42712067e-02, 1.39593626e-07,
             1.94720190e-01, -3.49188697e-01, -1.31228326e-07, ],
            [2.01519849e-10, 6.58626116e-10, 4.44106624e-06, -5.17270348e-10, -1.75271200e-10, -2.02383752e-06,
             3.15750472e-10, -4.83355620e-10, -2.15141967e-06, ],
            [-4.78354633e-01, -9.28932439e-02, -5.17270348e-10, 4.77068864e-01, 1.20898325e-01, 2.27650618e-07,
             1.28613515e-03, -2.80051173e-02, -2.30263842e-07, ],
            [-1.53645723e-01, -7.42712067e-02, -1.75271200e-10, 1.20898325e-01, 6.68427367e-02, 5.48141018e-08,
             3.27473460e-02, 7.42862870e-03, -5.87412969e-08, ],
            [-3.88607902e-07, 1.39593626e-07, -2.02383752e-06, 2.27650618e-07, 5.48141018e-08, 1.86869915e-06,
             1.60957288e-07, -1.94407719e-07, 2.25367506e-08, ],
            [-2.03194076e-01, 1.94720190e-01, 3.15750472e-10, 1.28613515e-03, 3.27473460e-02, 1.60957288e-07,
             2.01908446e-01, -2.27467221e-01, -1.61529252e-07, ],
            [2.55471871e-01, -3.49188697e-01, -4.83355620e-10, -2.80051173e-02, 7.42862870e-03, -1.94407719e-07,
             -2.27467221e-01, 3.41760135e-01, 1.89969615e-07, ],
            [3.91793089e-07, -1.31228326e-07, -2.15141967e-06, -2.30263842e-07, -5.87412969e-08, 2.25367506e-08,
             -1.61529252e-07, 1.89969615e-07, 1.99567548e-06, ],
        ])
        hessian_prop = db.DenseMatrixProperty(hessian_props[0])
        hessian_prop.link(properties)
        hessian = hessian_prop.get_data()
        assert hessian.shape == (9, 9)
        assert np.allclose(ref_hessian, hessian, atol=1e-1)
        # Normal modes
        ref_normal_modes = np.array([
            [-2.89597029e-02, -3.15134055e-02, 6.80792263e-02, ],
            [-6.12657343e-02, -3.78163196e-02, -3.58186536e-02, ],
            [4.20375520e-18, 2.34897827e-18, 1.00104275e-17, ],
            [-1.79731098e-01, 7.85076713e-01, -5.36167737e-01, ],
            [5.37311156e-01, 1.05145171e-01, 8.02545212e-03, ],
            [-1.04237003e-16, -1.01785409e-16, 7.12699448e-17, ],
            [6.39425796e-01, -2.84845564e-01, -5.44494573e-01, ],
            [4.35196516e-01, 4.95135906e-01, 5.60545477e-01, ],
            [-1.32320360e-16, 7.19185833e-17, -4.43320067e-16, ],
        ])
        normal_mode_prop = db.DenseMatrixProperty(normal_modes_props[0])
        normal_mode_prop.link(properties)
        normal_modes = normal_mode_prop.get_data()
        assert normal_modes.shape == (9, 3)
        assert np.allclose(np.abs(ref_normal_modes), np.abs(normal_modes), atol=1e-1)

        # Frequencies
        ref_frequencies = [0.00099327, 0.00261741, 0.00276749]
        frequencies_prop = db.VectorProperty(frequencies_props[0])
        frequencies_prop.link(properties)
        frequencies = frequencies_prop.get_data()
        assert len(frequencies) == 3
        self.assertAlmostEqual(ref_frequencies[0], frequencies[0], delta=1e-1)
        self.assertAlmostEqual(ref_frequencies[1], frequencies[1], delta=1e-1)
        self.assertAlmostEqual(ref_frequencies[2], frequencies[2], delta=1e-1)

        # Gibbs free energy
        gibbs_free_energy_prop = db.NumberProperty(gibbs_free_energy_props[0])
        gibbs_free_energy_prop.link(properties)
        gibbs_free_energy = gibbs_free_energy_prop.get_data()
        self.assertAlmostEqual(gibbs_free_energy, -4.069523427319414, delta=1e-1)

        # Gibbs energy correction
        gibbs_energy_correction_prop = db.NumberProperty(gibbs_energy_correction_props[0])
        gibbs_energy_correction_prop.link(properties)
        gibbs_energy_correction = gibbs_energy_correction_prop.get_data()
        self.assertAlmostEqual(gibbs_energy_correction, 0.002052217141994994, delta=1e-5)

        # Position Shift
        ref_position_shift = np.array([
            [-2.28040644e-01, -6.57123762e-01, 4.61543866e-17],
            [-3.75344241e-01, -1.57403825e-01, 1.12412725e-16],
            [-3.41478178e-01, -1.30335476e-01, 1.14223117e-16]])
        position_shift_prop = db.DenseMatrixProperty(position_shift_props[0])
        position_shift_prop.link(properties)
        position_shift = position_shift_prop.get_data()
        assert position_shift.shape == (3, 3)
        assert np.allclose(ref_position_shift, position_shift, atol=1e-1)
        new_positions = structure.get_atoms().positions
        water_atoms, _ = utils.io.read(water)
        assert np.allclose(water_atoms.positions, new_positions - position_shift)
