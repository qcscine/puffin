#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
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


class ScineReactTsGuessJobTest(JobTestCase):

    @skip_without('database', 'readuct')
    def test_job_sn2(self):
        # import Job
        from scine_puffin.jobs.scine_react_ts_guess import ScineReactTsGuess
        import scine_database as db
        import scine_utilities as su

        # Setup DB for calculation
        ts_opt = os.path.join(resource_path(), "CH3ClBr_ts.xyz")
        structure = add_structure(self.manager, ts_opt, db.Label.TS_OPTIMIZED, charge=-1, multiplicity=1)
        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_ts_guess')
        # Settings
        # Enhanced SD Optimizer to reduce number of steps
        settings = {
            "irc_irc_initial_step_size": 0.3,
            "irc_convergence_max_iterations": 250,
            "irc_sd_use_trust_radius": True,
            "irc_sd_trust_radius": 0.05,
            "irc_sd_dynamic_multiplier": 1.2,
            'spin_propensity_check': 0
        }
        calculation = add_calculation(self.manager, model, job, [structure.id()], settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactTsGuess()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.structure_ids) == 7
        assert len(results.property_ids) >= 10
        assert len(results.elementary_step_ids) == 3
        found_ts = False
        for sid in results.elementary_step_ids:
            step = db.ElementaryStep(sid, self.manager.get_collection('elementary_steps'))
            if not step.has_transition_state():
                continue
            found_ts = True
            ts = db.Structure(step.get_transition_state(), self.manager.get_collection("structures"))
            fit = su.QuaternionFit(ts.get_atoms().positions, structure.get_atoms().positions)
            assert fit.get_rmsd() < 0.1
            assert step.has_spline()
        assert found_ts
