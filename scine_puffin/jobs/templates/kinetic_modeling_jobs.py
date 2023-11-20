# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import numpy as np
from typing import List

import scine_database as db

from ..templates.job import Job, job_configuration_wrapper
from ...utilities.compound_and_flask_helpers import get_compound_or_flask
from scine_puffin.config import Configuration


class KineticModelingJob(Job):
    """
    Abstract base class for the RMS kinetic modeling and KiNetX kinetic modeling jobs.
    """
    def __init__(self):
        super().__init__()
        self.name = "KineticModelingJob"
        self.model: db.Model = db.Model("PM6", "PM6", "")
        self.c_max_label = "max_concentration"
        self.c_final_label = "final_concentration"
        self.c_flux_label = "concentration_flux"
        self.r_flux_label = "_reaction_edge_flux"
        self.r_forward_label = "_forward_edge_flux"
        self.r_backward_label = "_backward_edge_flux"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs():
        raise NotImplementedError

    def _write_concentrations_to_centroids(self, aggregate_ids: List[db.ID], aggregate_types: List[db.CompoundOrFlask],
                                           reaction_ids: List[db.ID], aggregate_wise_concentrations: List[np.ndarray],
                                           reaction_wise_concentrations: List[np.ndarray],
                                           aggregate_wise_labels: List[str],
                                           reaction_wise_labels: List[str],
                                           results: db.Results,
                                           post_fix: str = "",
                                           add_flask_result_to_compounds: bool = False):
        assert len(aggregate_wise_concentrations) == len(aggregate_wise_labels)
        assert len(reaction_wise_concentrations) == len(reaction_wise_labels)
        if add_flask_result_to_compounds:
            original_concentration_data = np.zeros((len(aggregate_ids), len(aggregate_wise_concentrations)))
            for i, concentrations in enumerate(aggregate_wise_concentrations):
                original_concentration_data[:, 0] = concentrations
            concentration_data = self._resolve_flask_to_compound_mapping(original_concentration_data, aggregate_ids,
                                                                         aggregate_types)
            for i, concentrations in enumerate(aggregate_wise_concentrations):
                concentrations = concentration_data[:, i]

        print("Concentration Properties")
        for i, (a_id, a_type) in enumerate(zip(aggregate_ids, aggregate_types)):
            centroid = self._get_aggregate_centroid(a_id, a_type)
            for concentrations, concentration_label in zip(aggregate_wise_concentrations,
                                                           aggregate_wise_labels):
                c = concentrations[i]
                label = concentration_label + post_fix
                self._write_concentration_property(centroid, label, c, results)

        print("Reaction flux properties")
        for i, r_id in enumerate(reaction_ids):
            centroid = self._get_reaction_centroid(r_id)
            for concentrations, concentration_label in zip(reaction_wise_concentrations,
                                                           reaction_wise_labels):
                c = concentrations[i]
                label = r_id.string() + concentration_label + post_fix
                self._write_concentration_property(centroid, label, c, results)

    def _write_concentration_property(self, centroid: db.Structure, label: str, value: float, results: db.Results):
        prop = db.NumberProperty.make(label, self.model, value, self._properties)
        results.add_property(prop.id())
        centroid.add_property(label, prop.id())
        prop.set_structure(centroid.id())
        print("struc", centroid.id().string(), "   prop ", prop.id().string(), "    ", label, "    ", value)

    def _resolve_flask_to_compound_mapping(self, concentration_data, aggregate_id_list,
                                           aggregate_type_list):
        i = 0
        new_concentration_data = np.copy(concentration_data)
        for a_id, a_type in zip(aggregate_id_list, aggregate_type_list):
            if a_type == db.CompoundOrFlask.FLASK:
                flask = db.Flask(a_id, self._flasks)
                compounds_in_flask = flask.get_compounds()
                for c_id in compounds_in_flask:
                    if c_id in aggregate_id_list:
                        j = aggregate_id_list.index(c_id)
                        new_concentration_data[j, :] += concentration_data[i, :]
            i += 1
        return new_concentration_data

    def _disable_all_aggregates(self):
        """
        Disable the exploration of all aggregates.
        """
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            compound.disable_exploration()
        for flask in self._flasks.iterate_all_flasks():
            flask.link(self._flasks)
            flask.disable_exploration()

    def _get_reaction_centroid(self, r_id):
        a_id = db.Reaction(r_id, self._reactions).get_reactants(db.Side.LHS)[0][0]
        a_type = db.Reaction(r_id, self._reactions).get_reactant_types(db.Side.LHS)[0][0]
        aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
        return db.Structure(aggregate.get_centroid(), self._structures)

    def _get_aggregate_centroid(self, a_id, a_type):
        aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
        return db.Structure(aggregate.get_centroid(), self._structures)
