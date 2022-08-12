#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_structure
)

from ..resources import resource_path


class ScineIrcScanJobTest(JobTestCase):

    @skip_without('database', 'readuct')
    def test_correct_stop_on_error_fail(self):
        # import Job
        from scine_puffin.jobs.scine_irc_scan import ScineIrcScan
        import scine_database as db

        # Setup DB for calculation
        ts_opt = os.path.join(resource_path(), "CH3ClBr_ts.xyz")
        structure = add_structure(self.manager, ts_opt, db.Label.TS_OPTIMIZED, charge=-1, multiplicity=1)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_irc_scan')
        # Settings
        settings = {
            "convergence_max_iterations": 10
        }
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineIrcScan()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        assert not job.run(self.manager, calculation, config)
        # Check results
        assert calculation.get_status() == db.Status.FAILED
        assert calculation.has_comment()
        assert calculation.get_comment() == "\nProblem: IRC optimization did not converge."

    @skip_without('database', 'readuct')
    def test_full_optimization(self):
        # import Job
        from scine_puffin.jobs.scine_irc_scan import ScineIrcScan
        import scine_database as db

        # Setup DB for calculation
        ts_opt = os.path.join(resource_path(), "CH3ClBr_ts.xyz")
        structure = add_structure(self.manager, ts_opt, db.Label.TS_OPTIMIZED, charge=-1, multiplicity=1)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_irc_scan')
        # Settings
        # Enhanced SD Optimizer to reduce number of steps
        settings = {
            "irc_initial_step_size": 0.3,
            "convergence_max_iterations": 250,
            "sd_use_trust_radius": True,
            "sd_trust_radius": 0.05,
            "sd_dynamic_multiplier": 1.2
        }
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineIrcScan()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        assert not calculation.has_comment()
        results = calculation.get_results()
        assert len(results.structure_ids) == 2
        # Retrieve resulting structures
        structures = self.manager.get_collection("structures")
        forward_struct = structures.get_structure(results.structure_ids[0])
        forward_struct.link(structures)
        backward_struct = structures.get_structure(results.structure_ids[1])
        backward_struct.link(structures)
        # Check labels
        assert forward_struct.get_label() == db.Label.MINIMUM_OPTIMIZED
        assert backward_struct.get_label() == db.Label.MINIMUM_OPTIMIZED
        # Check electronic energies are correctly assigned (forward, backward)
        assert forward_struct.has_property("electronic_energy")
        forward_energy_prop = forward_struct.get_properties("electronic_energy")
        assert backward_struct.has_property("electronic_energy")
        backward_energy_prop = backward_struct.get_properties("electronic_energy")
        assert len(results.property_ids) == 2
        assert len(forward_energy_prop) == 1
        assert len(backward_energy_prop) == 1
        assert forward_energy_prop[0] == results.property_ids[0]
        assert backward_energy_prop[0] == results.property_ids[1]
        # Check numerics of energies
        properties = self.manager.get_collection("properties")
        forward_energy = db.NumberProperty(forward_energy_prop[0])
        forward_energy.link(properties)
        backward_energy = db.NumberProperty(backward_energy_prop[0])
        backward_energy.link(properties)
        self.assertAlmostEqual(forward_energy.get_data(), -9.076183551, delta=1e-1)
        self.assertAlmostEqual(backward_energy.get_data(), -9.075089877, delta=1e-1)

    @skip_without('database', 'readuct')
    def test_partial_optimization(self):
        # import Job
        from scine_puffin.jobs.scine_irc_scan import ScineIrcScan
        import scine_database as db

        # Setup DB for calculation
        ts_opt = os.path.join(resource_path(), "CH3ClBr_ts.xyz")
        structure = add_structure(self.manager, ts_opt, db.Label.TS_OPTIMIZED, charge=-1, multiplicity=1)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_irc_scan')
        # Settings
        # Enhanced SD Optimizer to reduce number of steps
        settings = {
            "stop_on_error": False,
            "convergence_max_iterations": 20,
        }
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineIrcScan()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert calculation.has_comment()
        assert calculation.get_comment() == "Optimization did not fully converge for one or both sides. 'forward' and "\
                                            "'backward' structures are stored as 'MINIMUM_GUESS'."
        assert len(results.structure_ids) == 2
        # Retrieve resulting structures
        structures = self.manager.get_collection("structures")
        forward_struct = structures.get_structure(results.structure_ids[0])
        forward_struct.link(structures)
        backward_struct = structures.get_structure(results.structure_ids[1])
        backward_struct.link(structures)
        # Check labels
        assert forward_struct.get_label() == db.Label.MINIMUM_GUESS
        assert backward_struct.get_label() == db.Label.MINIMUM_GUESS
        # Check electronic energies are correctly assigned (forward, backward)
        assert forward_struct.has_property("electronic_energy")
        forward_energy_prop = forward_struct.get_properties("electronic_energy")
        assert backward_struct.has_property("electronic_energy")
        backward_energy_prop = backward_struct.get_properties("electronic_energy")
        assert len(results.property_ids) == 2
        assert len(forward_energy_prop) == 1
        assert len(backward_energy_prop) == 1
        assert forward_energy_prop[0] == results.property_ids[0]
        assert backward_energy_prop[0] == results.property_ids[1]
        # # Check numerics of energies
        properties = self.manager.get_collection("properties")
        forward_energy = db.NumberProperty(forward_energy_prop[0])
        forward_energy.link(properties)
        backward_energy = db.NumberProperty(backward_energy_prop[0])
        backward_energy.link(properties)
        self.assertAlmostEqual(forward_energy.get_data(), -9.064454196, delta=1e-1)
        self.assertAlmostEqual(backward_energy.get_data(), -9.070640750, delta=1e-1)
