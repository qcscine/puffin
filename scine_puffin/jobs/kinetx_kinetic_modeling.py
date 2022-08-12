# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List

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
            "energy_model_program": "any"
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
                raise Exception("The number of reaction rates differs from the number of reactions.")
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
            concentration_data = kinetx.integrate(network, concentrations, 0.0, time_step, solver,
                                                  batch_interval, n_batches, convergence)
            # Save the concentrations
            results = calculation.get_results()
            self._write_concentrations_to_centroids(aggregate_id_list, aggregate_type_list, concentration_data, manager,
                                                    results)
            calculation.set_results(results)
            self._disable_all_aggregates()
            self.complete_job()

        return self.postprocess_calculation_context()

    def _write_concentrations_to_centroids(self, aggregate_id_list, aggregate_type_list, concentration_data, manager,
                                           results) -> None:
        """
        Write the final and maximum concentrations to the centroids of each compound.
        """
        self.model.program = self.settings["energy_model_program"]
        i = 0
        for a_id, a_type in zip(aggregate_id_list, aggregate_type_list):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = aggregate.get_centroid(manager)
            flux_c = concentration_data[i, 2]
            max_c = concentration_data[i, 1]
            final_c = concentration_data[i, 0]
            max_concentration_property = db.NumberProperty.make("max_concentration", self.model,
                                                                max_c, self._properties)
            final_concentration_property = db.NumberProperty.make("final_concentration", self.model,
                                                                  final_c, self._properties)
            concentration_flux_property = db.NumberProperty.make("concentration_flux", self.model,
                                                                 flux_c, self._properties)
            results.add_property(max_concentration_property.id())
            results.add_property(final_concentration_property.id())
            results.add_property(concentration_flux_property.id())
            centroid.add_property("max_concentration", max_concentration_property.id())
            centroid.add_property("final_concentration", final_concentration_property.id())
            centroid.add_property("concentration_flux", concentration_flux_property.id())
            max_concentration_property.set_structure(centroid.id())
            final_concentration_property.set_structure(centroid.id())
            concentration_flux_property.set_structure(centroid.id())
            i += 1

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
            raise Exception("Unbalanced masses in reaction. You are destroying/creating atoms!")

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
