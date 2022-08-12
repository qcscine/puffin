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


class ScineGeometryOptimizationJobTest(JobTestCase):

    def run_by_label(self, input_label, expected_label):
        # import Job
        from scine_puffin.jobs.scine_ts_optimization import ScineTsOptimization
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "ts_proline_acid_propanal.xyz")
        structure = add_structure(self.manager, water, input_label)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_geometry_optimization')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = ScineTsOptimization()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.structure_ids) == 1
        structure = db.Structure(results.structure_ids[0])
        structures = self.manager.get_collection("structures")
        assert structures.count("{}") == 2
        structure.link(structures)
        assert expected_label == structure.get_label()
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        assert len(energy_props) == 1
        assert len(results.property_ids) == 1
        assert energy_props[0] == results.property_ids[0]
        energy = db.NumberProperty(energy_props[0])
        properties = self.manager.get_collection("properties")
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -31.659571862, delta=1e-1)

    @skip_without('database', 'readuct')
    def test_user_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.USER_GUESS, db.Label.TS_OPTIMIZED)

    @skip_without('database', 'readuct')
    def test_minimum_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.MINIMUM_GUESS, db.Label.TS_OPTIMIZED)

    @skip_without('database', 'readuct')
    def test_surface_guess(self):
        import scine_database as db
        self.run_by_label(db.Label.SURFACE_GUESS, db.Label.TS_OPTIMIZED)
