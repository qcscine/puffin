#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os
import numpy as np

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_structure
)

from ..resources import resource_path


class ScineHessianJobTest(JobTestCase):

    @skip_without('database', 'readuct')
    def test_water(self):
        # import Job
        from scine_puffin.jobs.scine_hessian import ScineHessian
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_OPTIMIZED)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_hessian')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineHessian()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
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
        self.assertAlmostEqual(energy.get_data(), -4.061143327, delta=1e-1)
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
    def test_single_atom_fail(self):
        # import Job
        from scine_puffin.jobs.scine_hessian import ScineHessian
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "h.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_OPTIMIZED, 0, 2)
        model = db.Model('dftb3', '', '')
        job = db.Job('scine_hessian')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineHessian()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert not job.run(self.manager, calculation, config)
        assert calculation.get_status() == db.Status.FAILED
        assert calculation.has_comment()
        assert calculation.get_comment() != ""
