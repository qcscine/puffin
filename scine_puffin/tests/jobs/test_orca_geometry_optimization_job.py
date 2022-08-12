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


class OrcaGeometryOptimizationJobTest(JobTestCase):

    @skip_without('database', 'orca')
    def test_energy(self):
        # import Job
        from scine_puffin.jobs.orca_geometry_optimization import OrcaGeometryOptimization
        import scine_database as db

        # Setup DB for calculation
        water = os.path.join(resource_path(), "water.xyz")
        structure = add_structure(self.manager, water, db.Label.USER_GUESS)
        model = db.Model('dft', 'pbe', 'def2-SVP')
        job = db.Job('orca_geometry_optimization')
        calculation = add_calculation(self.manager, model, job, [structure.id()])

        # Run calculation/job
        config = self.get_configuration()
        job = OrcaGeometryOptimization()
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
        assert db.Label.USER_OPTIMIZED == structure.get_label()
        assert structure.has_property("electronic_energy")
        energy_props = structure.get_properties("electronic_energy")
        assert len(energy_props) == 1
        assert len(results.property_ids) == 1
        assert energy_props[0] == results.property_ids[0]
        energy = db.NumberProperty(energy_props[0])
        properties = self.manager.get_collection("properties")
        energy.link(properties)
        self.assertAlmostEqual(energy.get_data(), -76.272650836328, delta=1e-1)
