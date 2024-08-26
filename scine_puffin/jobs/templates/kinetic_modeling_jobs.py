# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from typing import List, TYPE_CHECKING

import numpy as np

from ..templates.job import Job
from ...utilities.compound_and_flask_helpers import get_compound_or_flask
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class KineticModelingJob(Job, ABC):
    """
    Abstract base class for the RMS kinetic modeling and KiNetX kinetic modeling jobs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "KineticModelingJob"
        self.model: db.Model = db.Model("PM6", "PM6", "")
        self.c_max_label = "max_concentration"
        self.c_final_label = "final_concentration"
        self.c_flux_label = "concentration_flux"
        self.r_flux_label = "_reaction_edge_flux"
        self.r_forward_label = "_forward_edge_flux"
        self.r_backward_label = "_backward_edge_flux"

    def _write_concentrations_to_centroids(self, aggregate_ids: List[db.ID], aggregate_types: List[db.CompoundOrFlask],
                                           reaction_ids: List[db.ID], aggregate_wise_concentrations: List[np.ndarray],
                                           reaction_wise_concentrations: List[np.ndarray],
                                           aggregate_wise_labels: List[str],
                                           reaction_wise_labels: List[str],
                                           results: db.Results,
                                           post_fix: str = ""):
        assert len(aggregate_wise_concentrations) == len(aggregate_wise_labels)
        assert len(reaction_wise_concentrations) == len(reaction_wise_labels)

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

    @requires("database")
    def _write_concentration_property(self, centroid: db.Structure, label: str, value: float, results: db.Results) \
            -> None:
        prop = db.NumberProperty.make(label, self.model, value, self._properties)
        results.add_property(prop.id())
        centroid.add_property(label, prop.id())
        prop.set_structure(centroid.id())
        print("struc", centroid.id().string(), "   prop ", prop.id().string(), "    ", label, "    ", value)

    def _disable_all_aggregates(self) -> None:
        """
        Disable the exploration of all aggregates.
        """
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            compound.disable_exploration()
        for flask in self._flasks.iterate_all_flasks():
            flask.link(self._flasks)
            flask.disable_exploration()

    @requires("database")
    def _get_reaction_centroid(self, r_id: db.ID) -> db.Structure:
        a_id = db.Reaction(r_id, self._reactions).get_reactants(db.Side.LHS)[0][0]
        a_type = db.Reaction(r_id, self._reactions).get_reactant_types(db.Side.LHS)[0][0]
        aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
        return db.Structure(aggregate.get_centroid(), self._structures)

    @requires("database")
    def _get_aggregate_centroid(self, a_id: db.ID, a_type: db.CompoundOrFlask) -> db.Structure:
        aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
        return db.Structure(aggregate.get_centroid(), self._structures)
