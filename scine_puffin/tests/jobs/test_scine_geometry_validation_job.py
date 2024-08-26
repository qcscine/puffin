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
            [-6.82647619e-02, 1.27298723e-02, 4.55149739e-02, ],
            [1.82667950e-03, -4.53157634e-03, 4.42513393e-02, ],
            [6.49103407e-18, -4.84694439e-18, 2.15525997e-18, ],
            [7.06741648e-01, -6.88502360e-01, -6.21030118e-02, ],
            [4.05861756e-01, 4.14501053e-01, 4.20445836e-02, ],
            [2.69669044e-16, -1.22067159e-17, -1.58816392e-17, ],
            [3.76865779e-01, 4.86433477e-01, -6.60383413e-01, ],
            [-4.34857733e-01, -3.42568629e-01, -7.44472580e-01, ],
            [-4.50402118e-16, 1.21528609e-17, 3.04612875e-17, ],
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
        ref_normal_modes = np.array([
            [-4.63375323e-02, -3.95703688e-02, -4.45850571e-02, ],
            [-4.62987096e-02, -3.98968446e-02, 4.43302315e-02, ],
            [-8.79759548e-18, 5.38614204e-20, 2.68498367e-19, ],
            [3.06061064e-02, 6.99750720e-01, 7.08297607e-01, ],
            [7.04937280e-01, -7.16265701e-02, -5.72307806e-04, ],
            [1.76423192e-16, 1.65375651e-16, 1.21343530e-16, ],
            [7.04937280e-01, -7.16265701e-02, -5.72307806e-04, ],
            [2.99898507e-02, 7.04933066e-01, -7.03107992e-01, ],
            [-2.93091491e-16, -1.57370285e-16, -2.11837496e-16, ],
        ])
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

    @skip_without('database', 'readuct', 'molassembler')
    def test_fail_to_optimize_non_valid_minimum(self):
        # fails because of different graph
        from scine_puffin.jobs.scine_geometry_validation import ScineGeometryValidation
        import scine_database as db
        import scine_molassembler as masm

        # Setup DB for calculation
        h2o2 = os.path.join(resource_path(), "h2o2_distorted.xyz")
        ref_graph = "pGFhgaVhYQBhYwNhb4GDAAECYXKjYWyDgQCBAYECYmxygoIAAYECYXOCgg" + \
                    "ABgQJhcwNhYw9hZ6JhRYODAAMAgwEDAIMCAwBhWoQBAQgIYXaDAQIB"
        structure = add_structure(self.manager, h2o2, db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", ref_graph)
        structure.set_graph("masm_decision_list", "")

        model = db.Model('dftb3', 'dftb3', '')
        model.program = "sparrow"
        model.spin_mode = "restricted"
        job = db.Job('scine_geometry_validation')
        settings = {
            "val_optimization_attempts": 2
        }

        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineGeometryValidation()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        success = job.run(self.manager, calculation, config)
        assert not success
        # Check comment of calculation
        ref_comment = "\nError: Scine Geometry Validation Job failed with message:" + \
            "\nFinal structure does not match starting structure. " + \
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
        self.assertAlmostEqual(energy.get_data(), -7.168684600560611, delta=1e-1)
        # Normal modes
        ref_normal_modes = np.array([
            [-4.51056224e-02, -2.30053324e-02, -1.91838298e-02, -3.99609071e-02, -3.51840370e-02, 3.93200055e-03, ],
            [-1.72509150e-03, 9.42119128e-03, 2.21094036e-02, -4.46041374e-02, 1.39010219e-03, -6.13743055e-02, ],
            [-3.12783219e-02, 5.92381295e-01, -2.71106437e-04, -1.63870079e-02, -1.25590502e-02, 2.41365634e-03, ],
            [4.21994858e-02, -5.94026789e-03, -4.34931029e-03, -7.12559616e-03, -1.92028040e-02, 9.56529286e-04, ],
            [-3.64372163e-03, -2.21299691e-02, -4.37910660e-02, 6.00944990e-03, 2.29206345e-03, -2.96324400e-04, ],
            [-2.80083927e-02, -5.64748740e-01, 6.07151008e-02, -5.48202434e-03, 1.07317405e-02, -9.59698746e-04, ],
            [-3.03145349e-02, 2.52662209e-01, 5.27982951e-02, 1.29431627e-01, 9.82317479e-01, -6.07141499e-02, ],
            [7.64453798e-02, 2.06808631e-01, 3.20757324e-01, 6.18000623e-01, -1.19002597e-01, -1.68844103e-02, ],
            [9.89629279e-01, -1.51090374e-02, -7.71596608e-02, -1.04947466e-01, 3.39323358e-02, -7.58173211e-03, ],
            [7.64453798e-02, 2.06808631e-01, 3.20757324e-01, 6.18000623e-01, -1.19002597e-01, -1.68844103e-02, ],
            [8.77700467e-03, -5.07459285e-03, 2.34086814e-02, -5.36404689e-03, 6.05533777e-02, 9.95819233e-01, ],
            [-4.85357718e-02, -4.23519054e-01, -8.82304042e-01, 4.52087705e-01, -4.92635570e-03, -1.54978070e-02, ],
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
        assert masm.JsonSerialization.equal_molecules(structure.get_graph("masm_cbor_graph"), ref_graph)

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
        model.spin_mode = "restricted"
        job = db.Job('scine_geometry_validation')
        settings = {
            "val_optimization_attempts": 2
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
            [-3.00191566e-02, -3.23938595e-02, 7.15634599e-02, ],
            [-6.37968518e-02, -3.85915640e-02, -3.76619702e-02, ],
            [4.38956694e-18, 2.37844772e-18, 1.05262604e-17, ],
            [-1.87877770e-01, 8.04193332e-01, -5.61620285e-01, ],
            [5.59025142e-01, 1.07533595e-01, 7.82721537e-03, ],
            [-1.08329864e-16, -1.04272618e-16, 7.48675733e-17, ],
            [6.64389811e-01, -2.89986210e-01, -5.74349350e-01, ],
            [4.53660473e-01, 5.05053400e-01, 5.90003781e-01, ],
            [-1.38379756e-16, 7.46124958e-17, -4.65817026e-16, ],
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
