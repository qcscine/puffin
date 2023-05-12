#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from json import dumps

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_compound_and_structure,
    add_reaction,
    add_flask_and_structure
)
from ...utilities.compound_and_flask_helpers import get_compound_or_flask


class KinetxKineticModelingJobTest(JobTestCase):

    @skip_without('database', 'kinetx')
    def test_concentrations(self):
        # import Job
        from scine_puffin.jobs.kinetx_kinetic_modeling import KinetxKineticModeling
        import scine_database as db

        # This reaction network is made up and converges fairly quickly.
        n_compounds = 5
        all_compounds = [add_compound_and_structure(self.manager) for _ in range(n_compounds - 1)]
        flask = add_flask_and_structure(self.manager)
        dummy_compound = add_compound_and_structure(self.manager)
        all_compounds.append(flask)
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        c_ids = [c.id() for c in all_compounds]
        all_reaction_ids = [
            add_reaction(self.manager, [c_ids[0]], [c_ids[2]]).id(),
            add_reaction(self.manager, [c_ids[1]], [c_ids[3]]).id(),
            add_reaction(self.manager, [c_ids[2], c_ids[3]], [c_ids[0], c_ids[1]]).id(),
            add_reaction(self.manager, [c_ids[0], c_ids[1]], [c_ids[4], c_ids[4]]).id()
        ]
        _ = add_reaction(self.manager, [dummy_compound.id()], [flask.id()])
        lhs_rates = [0.1, 0.05, 0.02, 0.02]
        rhs_rates = [0.05, 0.05, 0.001, 0.0000001]
        start_concentrations = [0.5, 0.4, 0.0, 0.0, 0.0]
        reference_data = [3.337327e-02, 5.991047e-05, 6.674503e-02, 5.839152e-05, 2 * 3.998817e-01]
        reference_max = [0.5, 0.4, 0.07103765, 0.00325947, 2 * 0.3998817]
        reference_flux = [1.85802001, 1.78338488, 1.45451295, 1.37987782, 2 * 0.40350706]

        for c in all_compounds:
            c.enable_exploration()

        model = db.Model('FAKE', '', '')
        job = db.Job('kinetx_kinetic_modeling')
        settings = {
            "time_step": 1e-08,
            "solver": "cash_karp_5",
            "batch_interval": 1000,
            "n_batches": 50000,
            "energy_label": "electronic_energy",
            "convergence": 1e-10,
            "lhs_rates": lhs_rates,
            "rhs_rates": rhs_rates,
            "start_concentrations": start_concentrations,
            "reaction_ids": [str(oid) for oid in all_reaction_ids],
            "aggregate_ids": [str(oid) for oid in c_ids],
            "aggregate_types": [db.CompoundOrFlask.COMPOUND for _ in range(4)] + [db.CompoundOrFlask.FLASK],
            "energy_model_program": "DUMMY",
            "instant_barrierless": False
        }

        calculation = add_calculation(self.manager, model, job, all_structure_ids, settings)

        # Run calculation/job
        config = self.get_configuration()
        job = KinetxKineticModeling()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        properties = self.manager.get_collection("properties")
        reactions = self.manager.get_collection("reactions")
        structures = self.manager.get_collection("structures")
        compounds = self.manager.get_collection("compounds")
        flasks = self.manager.get_collection("flasks")
        results = calculation.get_results()
        assert properties.count(dumps({})) == n_compounds * 3 + len(all_reaction_ids) * 3
        assert len(results.property_ids) == n_compounds * 3 + len(all_reaction_ids) * 3
        assert len(results.structure_ids) == 0
        assert len(results.elementary_step_ids) == 0
        for c in all_compounds:
            assert not c.explore()
        for i, s_id in enumerate(all_structure_ids):
            structure = db.Structure(s_id, structures)
            assert structure.has_property("final_concentration")
            assert structure.has_property("max_concentration")
            assert structure.has_property("concentration_flux")
            final_concentration = db.NumberProperty(structure.get_properties("final_concentration")[0], properties)
            max_concentration = db.NumberProperty(structure.get_properties("max_concentration")[0], properties)
            concentration_flux = db.NumberProperty(structure.get_properties("concentration_flux")[0], properties)
            self.assertAlmostEqual(final_concentration.get_data(), reference_data[i], delta=1e-2)
            self.assertAlmostEqual(max_concentration.get_data(), reference_max[i], delta=1e-2)
            self.assertAlmostEqual(concentration_flux.get_data(), reference_flux[i], delta=1e-1)

        for r_id in all_reaction_ids:
            total_flux_label = r_id.string() + "_reaction_edge_flux"
            forward_flux_label = r_id.string() + "_forward_edge_flux"
            backward_flux_label = r_id.string() + "_backward_edge_flux"
            a_id = db.Reaction(r_id, reactions).get_reactants(db.Side.LHS)[0][0]
            a_type = db.Reaction(r_id, reactions).get_reactant_types(db.Side.LHS)[0][0]
            aggregate = get_compound_or_flask(a_id, a_type, compounds, flasks)
            centroid = aggregate.get_centroid(self.manager)
            assert centroid.has_property(total_flux_label)
            assert centroid.has_property(forward_flux_label)
            assert centroid.has_property(backward_flux_label)
