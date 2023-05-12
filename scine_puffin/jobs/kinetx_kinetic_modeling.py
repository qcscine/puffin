# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List
import numpy as np

import scine_database as db

from .templates.job import Job, breakable, calculation_context, job_configuration_wrapper
from ..utilities.compound_and_flask_helpers import get_compound_or_flask
from scine_puffin.config import Configuration


class KinetxKineticModeling(Job):
    """
    A job that performs the kinetic modeling using KiNetX given a set of reactions and
    an electronic structure model. The reaction rates are calculated from transition
    state theory. The final concentration and the maximum concentration reached
    (for at least a given set of time steps) is added as properties to the centroids
    of each compound. As starting concentrations, the sum of all "start_concentration"
    properties of each structure for the given compounds is used.

    **Order Name**
      ``kinetx_kinetic_modeling``

    **Required Input**
      All reactions, concentrations, rate constants, etc. are parsed through the
      settings. The numerical integration is done through KiNetX. The final and
      maximum concentration for each compound is written to its centroid. The
      concentrations trajectories are written to the raw output by KiNetX.

      model :: db.Model
         The electronic structure model to flag the new properties with.

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.

      The following options are available:

      time_step ::float
          The integration time step.
      solver :: str
          The name of the numerical differential equation solver. Options are "CashKarp5" (default),
          "ImplicitEuler", and "ExplicitEuler".
      batch_interval :: int
          The numerical integration is done in batches of time steps. After each step the maximum
          concentration for each compound is updated. This is the size of each time-step batch.
      n_batches :: int
          The numerical integration is done in batches of time steps. After each step the maximum
          concentration for each compound is updated. This is the number of time-step batches.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - SCINE: KiNetX

    **Generated Data**
      If successful (technically and chemically) the following data will be
      generated and added to the database:

      Properties
        The maximum and final concentration of each compound is added to
        its centroid.
    """

    def __init__(self):
        super().__init__()
        self.name = "KiNetX kinetic modeling job"
        self.settings = {
            "energy_label": "electronic_energy",
            "time_step": 1e-8,
            "solver": "cash_karp_5",
            "batch_interval": 1000,
            "n_batches": 1000,
            "convergence": 1e-10,
            "energy_model_program": "any",
            "concentration_label_postfix": ""
        }
        self.model = db.Model("PM6", "PM6", "")
        self._flask_decomposition = dict()

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        import scine_kinetx as kinetx

        with breakable(calculation_context(self)):
            self.settings.update(calculation.get_settings())
            aggregate_id_list = [db.ID(c_id_str) for c_id_str in self.settings["aggregate_ids"]]
            aggregate_type_list = [db.CompoundOrFlask(a_type) for a_type in self.settings["aggregate_types"]]
            reaction_ids = [db.ID(r_id_str) for r_id_str in self.settings["reaction_ids"]]
            lhs_rates_per_reaction = self.settings["lhs_rates"]
            rhs_rates_per_reaction = self.settings["rhs_rates"]
            concentrations = self.settings["start_concentrations"]
            self.model = calculation.get_model()
            n_reactions = len(reaction_ids)
            n_aggregates = len(aggregate_id_list)
            if len(reaction_ids) != len(lhs_rates_per_reaction) or len(reaction_ids) != len(rhs_rates_per_reaction):
                raise RuntimeError("The number of reaction rates differs from the number of reactions.")
            network_builder = kinetx.NetworkBuilder()
            # Prepare the data arrays / network
            network_builder.reserve(n_compounds=n_aggregates, n_reactions=n_reactions, n_channels_per_reaction=1)
            # Add compounds and reactions
            self._add_all_aggregates(aggregate_id_list, aggregate_type_list, network_builder)
            self._add_all_reactions(reaction_ids, network_builder, aggregate_id_list, lhs_rates_per_reaction,
                                    rhs_rates_per_reaction, aggregate_type_list)
            network = network_builder.generate()
            # Solve network
            time_step = self.settings["time_step"]
            solver = kinetx.get_integrator(self.settings["solver"])
            batch_interval = self.settings["batch_interval"]
            n_batches = self.settings["n_batches"]
            convergence = self.settings["convergence"]
            if "max_time" in self.settings:
                concentration_data, reaction_flux, reaction_flux_forward, reaction_flux_backward = kinetx.integrate(
                    network, concentrations, 0.0, time_step, solver, batch_interval, n_batches, convergence, True,
                    self.settings["max_time"])
            else:
                concentration_data, reaction_flux, reaction_flux_forward, reaction_flux_backward = kinetx.integrate(
                    network, concentrations, 0.0, time_step, solver, batch_interval, n_batches, convergence)
            # Save the concentrations
            results = calculation.get_results()
            self._write_concentrations_to_centroids(aggregate_id_list, aggregate_type_list, concentration_data,
                                                    reaction_flux, reaction_flux_forward, reaction_flux_backward,
                                                    reaction_ids, manager, results)
            calculation.set_results(results)
            self._disable_all_aggregates()
            self.complete_job()

        return self.postprocess_calculation_context()

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

    def _write_concentrations_to_centroids(self, aggregate_id_list, aggregate_type_list, original_concentration_data,
                                           total_reaction_flux, forward_reaction_flux, backward_reaction_flux,
                                           reaction_ids, manager, results) -> None:
        """
        Write the final and maximum concentrations to the centroids of each compound.
        """
        self.model.program = self.settings["energy_model_program"]
        concentration_data = self._resolve_flask_to_compound_mapping(original_concentration_data, aggregate_id_list,
                                                                     aggregate_type_list)
        i = 0
        post_fix = self.settings["concentration_label_postfix"]
        print(post_fix)
        for a_id, a_type in zip(aggregate_id_list, aggregate_type_list):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = aggregate.get_centroid(manager)
            flux_c = concentration_data[i, 2]
            max_c = concentration_data[i, 1]
            final_c = concentration_data[i, 0]
            max_concentration_label = "max_concentration" + post_fix
            final_concentration_label = "final_concentration" + post_fix
            concentration_flux_label = "concentration_flux" + post_fix
            self._write_concentration_property(centroid, max_concentration_label, max_c, results)
            self._write_concentration_property(centroid, final_concentration_label, final_c, results)
            self._write_concentration_property(centroid, concentration_flux_label, flux_c, results)
            i += 1
        # Save edge flux (for the time being I will save it as a property to the centroid of the first LHS aggregate).
        for i, r_id in enumerate(reaction_ids):
            r_flux_total = total_reaction_flux[i, 0]
            r_flux_forward = forward_reaction_flux[i, 0]
            r_flux_backward = backward_reaction_flux[i, 0]
            total_flux_label = r_id.string() + "_reaction_edge_flux" + post_fix
            forward_flux_label = r_id.string() + "_forward_edge_flux" + post_fix
            backward_flux_label = r_id.string() + "_backward_edge_flux" + post_fix
            a_id = db.Reaction(r_id, self._reactions).get_reactants(db.Side.LHS)[0][0]
            a_type = db.Reaction(r_id, self._reactions).get_reactant_types(db.Side.LHS)[0][0]
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = aggregate.get_centroid(manager)
            self._write_concentration_property(centroid, total_flux_label, r_flux_total, results)
            self._write_concentration_property(centroid, forward_flux_label, r_flux_forward, results)
            self._write_concentration_property(centroid, backward_flux_label, r_flux_backward, results)

    def _write_concentration_property(self, centroid: db.Structure, label: str, value: float, results: db.Results):
        prop = db.NumberProperty.make(label, self.model, value, self._properties)
        results.add_property(prop.id())
        centroid.add_property(label, prop.id())
        prop.set_structure(centroid.id())

    def _add_all_aggregates(self, aggregate_id_list: List[db.ID], aggregate_type_list: List[db.CompoundOrFlask],
                            network_builder) -> None:
        """
        Add all aggregates to the kinetic model.
        """
        for a_id, a_type in zip(aggregate_id_list, aggregate_type_list):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = aggregate.get_centroid()
            mass = self._calculate_weight(centroid)
            network_builder.add_compound(mass, a_id.string())

    def _calculate_weight(self, structure_id) -> float:
        import scine_utilities as utils
        """
        Calculates the molecular weight, given a DB structure id.

        Attributes
        ----------
        structure :: db.Structure
            The structure of which to calculate the molecular weight.

        Returns
        -------
        weight :: float
            The molecular weight in a.u. .
        """
        structure = db.Structure(structure_id, self._structures)
        atoms = structure.get_atoms()
        weight = 0.0
        for e in atoms.elements:
            weight += utils.ElementInfo.mass(e)
        return weight

    def _add_all_reactions(self, reaction_ids, network_builder, aggregate_id_list, lhs_rates_per_reaction,
                           rhs_rates_per_reaction, aggregate_type_list) -> None:
        """
        Add all reactions to the kinetic model.
        """
        for i, reaction_id in enumerate(reaction_ids):
            reaction = db.Reaction(reaction_id, self._reactions)
            lhs_rates = [lhs_rates_per_reaction[i]]
            rhs_rates = [rhs_rates_per_reaction[i]]
            lhs_rhs_compound_or_flask_ids = reaction.get_reactants(db.Side.BOTH)
            lhs_stoichiometry = [(aggregate_id_list.index(c_id), 1) for c_id in lhs_rhs_compound_or_flask_ids[0]]
            rhs_stoichiometry = [(aggregate_id_list.index(c_id), 1) for c_id in lhs_rhs_compound_or_flask_ids[1]]
            self.check_mass_balance(lhs_stoichiometry, rhs_stoichiometry, aggregate_id_list, aggregate_type_list)
            network_builder.add_reaction(lhs_rates, rhs_rates, lhs_stoichiometry, rhs_stoichiometry)

    def check_mass_balance(self, lhs_stoichiometry, rhs_stoichiometry, aggregate_id_list, aggregate_type_list):
        lhs_mass = 0.0
        for a_index, n in lhs_stoichiometry:
            a_id = aggregate_id_list[a_index]
            a_type = aggregate_type_list[a_index]
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = aggregate.get_centroid()
            lhs_mass += n * self._calculate_weight(centroid)
        rhs_mass = 0.0
        for a_index, n in rhs_stoichiometry:
            a_id = aggregate_id_list[a_index]
            a_type = aggregate_type_list[a_index]
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = aggregate.get_centroid()
            rhs_mass += n * self._calculate_weight(centroid)
        if abs(rhs_mass - lhs_mass) > 1e-6:
            raise RuntimeError("Unbalanced masses in reaction. You are destroying/creating atoms!")

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

    @staticmethod
    def required_programs():
        return ["database", "kinetx", "utils"]
