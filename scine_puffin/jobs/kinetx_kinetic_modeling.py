# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Any, TYPE_CHECKING, List, Dict

from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.kinetic_modeling_jobs import KineticModelingJob
from ..utilities.compound_and_flask_helpers import get_compound_or_flask
from scine_puffin.config import Configuration
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class KinetxKineticModeling(KineticModelingJob):
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
      model : db.Model
        The electronic structure model to flag the new properties with.

    **Required Settings**
      aggregate_ids : List[str]
        The aggregate IDs (as strings).
      reaction_ids : List[str]
        The reaction IDs (as strings).
      aggregate_types : List[int]
        The aggregate types. 0 for compounds, 1 for flasks.
      lhs_rates : List[float]
        The reaction rates for the forward reactions.
      rhs_rates : List[float]
        The reaction rates for the backward reactions.
    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.

      The following options are available:

      time_step : float
          The integration time step.
      solver : str
          The name of the numerical differential equation solver. Options are "CashKarp5" (default),
          "ImplicitEuler", and "ExplicitEuler".
      batch_interval : int
          The numerical integration is done in batches of time steps. After each step the maximum
          concentration for each compound is updated. This is the size of each time-step batch.
      n_batches : int
          The numerical integration is done in batches of time steps. After each step the maximum
          concentration for each compound is updated. This is the number of time-step batches.
      energy_model_program : str
          The program with which the electronic structure model should be flagged. Default any.
      convergence : float
          Stop the numerical integration if the concentrations do not change more than this threshold between
          intervals.
      concentration_label_postfix : str
          Post fix to the property label. Default "".

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - SCINE: KiNetX

    **Generated Data**
      If successful (technically and chemically) the following data will be
      generated and added to the database:

      Properties
        The maximum and final concentration, and the vertex flux of each aggregate is added to
        its centroid. The edge flux, forward + backward flux for each reaction is added to the centroid of the first
        aggregate on the reaction's LHS. Note, that the properties are NOT listed in the results to avoid large DB
        documents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "KiNetX kinetic modeling job"
        self.settings: Dict[str, Any] = {
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

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        import scine_kinetx as kinetx

        with breakable(calculation_context(self)):
            self.settings.update(calculation.get_settings())
            aggregate_id_list = [db.ID(c_id_str) for c_id_str in self._get_setting_value_list("aggregate_ids")]
            aggregate_type_list = [db.CompoundOrFlask(a_type)
                                   for a_type in self._get_setting_value_list("aggregate_types")]
            reaction_ids = [db.ID(r_id_str) for r_id_str in self._get_setting_value_list("reaction_ids")]
            lhs_rates_per_reaction = self._get_setting_value_list("lhs_rates")
            rhs_rates_per_reaction = self._get_setting_value_list("rhs_rates")
            concentrations = self._get_setting_value_list("start_concentrations")
            self.model = calculation.get_model()
            n_reactions = len(reaction_ids)
            n_aggregates = len(aggregate_id_list)
            if len(reaction_ids) != len(lhs_rates_per_reaction) or len(reaction_ids) != len(rhs_rates_per_reaction):
                raise AssertionError("The number of reaction rates differs from the number of reactions.")
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
            final_concentrations = concentration_data[:, 0]
            max_concentrations = concentration_data[:, 1]
            flux_concentrations = concentration_data[:, 2]
            results = calculation.get_results()
            self.model.program = self.settings["energy_model_program"]
            self._write_concentrations_to_centroids(aggregate_id_list, aggregate_type_list, reaction_ids,
                                                    [max_concentrations, final_concentrations, flux_concentrations],
                                                    [reaction_flux, reaction_flux_forward, reaction_flux_backward],
                                                    [self.c_max_label, self.c_final_label, self.c_flux_label],
                                                    [self.r_flux_label, self.r_forward_label, self.r_backward_label],
                                                    results, self.settings["concentration_label_postfix"])
            # calculation.set_results(results)
            self._disable_all_aggregates()
            self.complete_job()

        return self.postprocess_calculation_context()

    def _get_setting_value_list(self, name: str) -> List[Any]:
        val = self.settings[name]
        if not isinstance(val, list):
            raise RuntimeError(f"Setting {name} must be a list.")
        return val

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

    @requires("utilities")
    def _calculate_weight(self, structure_id: db.ID) -> float:
        """
        Calculates the molecular weight, given a DB structure id.

        Attributes
        ----------
        structure_id : db.ID
            The structure of which to calculate the molecular weight.

        Returns
        -------
        weight : float
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
            raise AssertionError("Unbalanced masses in reaction. You are destroying/creating atoms!")

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "kinetx", "utils"]
