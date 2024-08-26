# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, TYPE_CHECKING
import math

import numpy as np
from scipy import integrate

from .rms_input_file_creator import create_rms_yml_file, resolve_rms_phase, resolve_rms_solver
from ..programs.rms import JuliaPrecompiler

from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    db = MissingDependency("scine_utilities")


class RMSKineticModel:
    """
    This class provides an interface to the ReactionMechanismSimulator (RMS) for kinetic modeling.
    """

    def __init__(self, settings: Dict, manager: db.Manager, model: db.Model, rms_path: str, rms_file_name: str) -> None:
        """
        Parameters:
        -----------
        settings : Dict[str, Any]
            The settings of the kinetic modeling calculation. This must contain:
                * The activation energies 'ea'.
                * The enthalpies 'enthalpies'.
                * The entropies 'entropies'.
                * The arrhenius prefactors 'arrhenius_prefactors'.
                * The arrhenius temperature exponents 'arrhenius_temperature_exponents'.
                * The lower and upper uncertainty bounds for the activation energies and enthalpies.
                * General settings for the kinetic modeling: Diffusion limited, phase type, aggregate ids, reaction ids,
              kinetic modeling solver, start concentrations, maximum integration time.
        manager : db.Manager
            The database manager.
        model : db.Model
            The main electronic structure model (used for default temperature and pressure).
        rms_path : str
            The path to the RMS shared library.
        rms_file_name : str
            The base file name for the RMS input file.
        """
        self.settings = settings
        self.ea: List[float] = [float(s) for s in self.settings["ea"]]
        self.h: List[float] = [float(s) for s in self.settings["enthalpies"]]
        self.s: List[float] = [float(s) for s in self.settings["entropies"]]
        self.a = [float(s) for s in self.settings["arrhenius_prefactors"]]
        self.n = [float(s) for s in self.settings["arrhenius_temperature_exponents"]]
        self.uq_ea_lower: List[float] = [0.0 for _ in self.ea]
        self.uq_ea_upper: List[float] = [0.0 for _ in self.ea]
        self.uq_h_lower: List[float] = [0.0 for _ in self.h]
        self.uq_h_upper: List[float] = [0.0 for _ in self.h]
        if "ea_lower_uncertainty" in self.settings:
            self.uq_ea_lower = [float(u) for u in self.settings["ea_lower_uncertainty"]]
        if "ea_upper_uncertainty" in self.settings:
            self.uq_ea_upper = [float(u) for u in self.settings["ea_upper_uncertainty"]]
        if "enthalpy_lower_uncertainty" in self.settings:
            self.uq_h_lower = [float(u) for u in self.settings["enthalpy_lower_uncertainty"]]
        if "enthalpy_upper_uncertainty" in self.settings:
            self.uq_h_upper = [float(u) for u in self.settings["enthalpy_upper_uncertainty"]]
        self.reactants: List[Tuple[List[str], List[str]]] = []
        self.viscosity: Optional[float] = None
        self.solvent_index: Optional[int] = None
        self.solvent_aggregate_str_id: Optional[str] = None
        self.diffusion_limited: bool = self.settings["diffusion_limited"]
        self.phase_type: str = self.settings["phase_type"]
        self._phase_options = ["ideal_dilute_solution", "ideal_gas"]
        self.a_str_ids: List[str] = self.settings["aggregate_ids"]
        self.r_str_ids: List[str] = self.settings["reaction_ids"]
        self.aggregate_types = [db.CompoundOrFlask(a_type) for a_type in self.settings["aggregate_types"]]
        self.solver: str = self.settings["solver"]
        self.start_concentrations = [float(s) for s in self.settings["start_concentrations"]]
        self.temperature = float(model.temperature) if self.settings["reactor_temperature"] == "none" else float(
            self.settings["reactor_temperature"])
        self.pressure = float(model.pressure) if self.settings["reactor_pressure"] == "none" else float(
            self.settings["reactor_pressure"])
        self.site_density = None
        if "site_density" in self.settings and self.settings["site_density"] != "none":
            self.site_density = float(self.settings["site_density"])

        self.solvent_species_added: bool = False

        self.solvent: Optional[str] = self.settings["reactor_solvent"]
        if self.solvent == "none":
            self.solvent = model.solvent if model.solvent != "none" else None
        if self.settings["solvent_aggregate_str_id"] != "none":
            self.solvent_index = self.settings["aggregate_ids"].index(self.settings["solvent_aggregate_str_id"])
            self.solvent_aggregate_str_id = self.settings["solvent_aggregate_str_id"]

        reaction_collection = manager.get_collection("reactions")
        for r_id_str in self.settings["reaction_ids"]:
            reactants = db.Reaction(db.ID(r_id_str), reaction_collection).get_reactants(db.Side.BOTH)
            self.reactants.append(([a_id.string() for a_id in reactants[0]], [a_id.string() for a_id in reactants[1]]))
        self.a_str_ids = self.settings["aggregate_ids"]
        self.max_time = float(self.settings["max_time"])
        self.abs_tol = float(self.settings["absolute_tolerance"])
        self.rel_tol = float(self.settings["relative_tolerance"])

        self._aggregates_to_reactions: Optional[Dict[str, List[str]]] = None
        self._rms_path = rms_path
        self._rms_file_name: str = rms_file_name

        self.calculate_adjoined = self.settings["sensitivity_analysis"] == "adjoined_sensitivities"
        if "adjpined_sensitivites" in self.settings:
            self.calculate_adjoined = self.settings["adjoined_sensitivities"]

        enforce_mass_balance = True
        if "enforce_mass_balance" in settings:
            enforce_mass_balance = bool(settings["enforce_mass_balance"])
        self.__sanity_checks(manager, enforce_mass_balance)

    def create_yml_file(self, file_name: Optional[str], h: Optional[List[float]] = None,
                        s: Optional[List[float]] = None, a: Optional[List[float]] = None,
                        n: Optional[List[float]] = None, ea: Optional[Union[List[float], np.ndarray]] = None):
        """
        Create a RMS input file with the given enthalpies (h), entropies (s), prefactors (a), temperature exponents (n),
        and activation energies (ea). Arguments not provided upon function call are set to their default values in
        saved upon class construction.

        All values musst be provided in SI units (e.g., J/mol for the enthalpies).
        """
        if file_name is None:
            file_name = self._rms_file_name
        if h is None:
            h = self.h
        if s is None:
            s = self.s
        if a is None:
            a = self.a
        if n is None:
            n = self.n
        if ea is None:
            ea = self.ea
        create_rms_yml_file(self.a_str_ids, h, s, a, n, ea, self.reactants, file_name, self.solvent, self.viscosity,
                            self.solvent_index)

    def _get_phase(self, file_name: Optional[str] = None, h: Optional[List[float]] = None,
                   s: Optional[List[float]] = None, a: Optional[List[float]] = None, n: Optional[List[float]] = None,
                   ea: Optional[Union[List[float], np.ndarray]] = None):
        """
        Getter for the RMS phase object. Values other than the original settings values may be provided for
        enthalpies (h), entropies (s), etc.

        All values musst be provided in SI units (e.g., J/mol for the enthalpies).
        """
        # pylint: disable=import-error
        JuliaPrecompiler().set_root(self._rms_path)
        JuliaPrecompiler().ensure_is_compiled()
        from julia import ReactionMechanismSimulator as rms
        # pylint: enable=import-error
        if file_name is None:
            file_name = "chem.rms"

        self.create_yml_file(file_name, h, s, a, n, ea)
        phase_dict = rms.readinput(file_name)
        rms_species = phase_dict["phase"]["Species"]
        rms_reactions = phase_dict["phase"]["Reactions"]
        rms_solvent = None
        if "Solvents" in phase_dict:
            rms_solvent = phase_dict["Solvents"][0]

        n_rms_species = len(rms_species)
        self.solvent_species_added = True if n_rms_species > len(self.a_str_ids) else False

        return resolve_rms_phase(self.phase_type, rms_species, rms_reactions,
                                 rms_solvent, self.diffusion_limited,
                                 self.site_density)

    def get_initial_conditions(self) -> Dict:
        """
        Getter for the initial conditions dictionary.
        """
        initial_conditions = {"T": self.temperature}
        for a_str_id, c in zip(self.a_str_ids, self.start_concentrations):
            if c > 1e-16:
                initial_conditions[a_str_id] = c
        return initial_conditions

    def _get_domain(self, phase: Any, initial_conditions: Dict):
        """
        Getter for the RMS domain object.
        """
        # pylint: disable=import-error
        JuliaPrecompiler().set_root(self._rms_path)
        JuliaPrecompiler().ensure_is_compiled()
        from julia import ReactionMechanismSimulator as rms
        # pylint: enable=import-error
        if self.phase_type == "ideal_gas":
            initial_conditions["P"] = self.pressure
            domain, y0, p = rms.ConstantTPDomain(phase=phase, initialconds=initial_conditions)
            volume = y0[-1]  # The reactor volume is calculated as V = nRT/P and provided as the last element of y0
            return domain, y0, p, volume, initial_conditions
        if self.phase_type == "ideal_dilute_solution":
            volume = 1e-3  # 1 L
            initial_conditions["V"] = volume
            if self.solvent_index is None:
                assert self.solvent
                initial_conditions[self.solvent] = self.settings["solvent_concentration"]
            domain, y0, p = rms.ConstantTVDomain(phase=phase, initialconds=initial_conditions)
            return domain, y0, p, volume, initial_conditions
        raise RuntimeError("Error: Unknown phase type.")

    def __sanity_checks(self, manager: db.Manager, enforce_mass_balance: bool):
        """
        Perform sanity checks for the attributes.
        """
        if self.settings["max_time"] < 0.0:
            raise AssertionError("The maximum time must be larger 0.0 for kinetic modeling.")
        if self.phase_type not in self._phase_options:
            raise LookupError("Unknown phase type. Options are: " + str(self._phase_options))
        if self.phase_type == "ideal_dilute_solution" and self.solvent is None:
            raise AssertionError("An ideal solution was requested but no solvent was specified. Please add an"
                                 " appropriate\nentry to the model definition or the job settings.")
        if len(self.reactants) != len(self.ea) or len(self.reactants) != len(self.a):
            raise AssertionError("The number of activation energies/prefactors differs from the number of"
                                 " reactions.")
        if len(self.a_str_ids) != len(self.h) or len(self.a_str_ids) != len(self.s):
            raise AssertionError("The number of aggregate entropies/enthalpies differs from the number of"
                                 " aggregates.")
        compounds = manager.get_collection("compounds")
        flasks = manager.get_collection("flasks")
        structures = manager.get_collection("structures")
        if enforce_mass_balance:
            for r_str_id, reactants in zip(self.r_str_ids, self.reactants):
                if not self._balanced_reaction(reactants, compounds, flasks, structures):
                    raise RuntimeError("Mass unbalance for reaction", r_str_id)

    @staticmethod
    def _balanced_reaction(reactants: Tuple[List[str], List[str]], compounds: db.Collection, flasks: db.Collection,
                           structures: db.Collection):
        """
        Check if the reaction conserves mass balance. This is used in sanity checks.
        """
        lhs_w = sum(RMSKineticModel._calculate_weight(a_id, compounds, flasks, structures) for a_id in reactants[0])
        rhs_w = sum(RMSKineticModel._calculate_weight(a_id, compounds, flasks, structures) for a_id in reactants[1])
        return abs(lhs_w - rhs_w) < 1e-6

    @staticmethod
    def _calculate_weight(a_str_id: str, compounds: db.Collection, flasks: db.Collection,
                          structures: db.Collection) -> float:
        """
        Calculate the molecular weight of the given aggregate
        """
        a_id = db.ID(a_str_id)
        aggregate: Union[db.Compound, db.Flask]
        aggregate = db.Compound(a_id, compounds)
        if not aggregate.exists():
            aggregate = db.Flask(a_id, flasks)
        structure = db.Structure(aggregate.get_centroid(), structures)
        weight = sum(utils.ElementInfo.mass(e) for e in structure.get_atoms().elements)
        return weight

    def run_kinetic_modeling(self, rms_file_name: Optional[str] = None, h: Optional[List[float]] = None,
                             s: Optional[List[float]] = None, a: Optional[List[float]] = None,
                             n: Optional[List[float]] = None, ea: Optional[Union[List[float], np.ndarray]] = None):
        """
        Run the kinetic modeling. If no values for the enthalpies (h), entropies (s), activation energies (ea),
        Arrhenius prefactors (a), temperature exponents (n) etc. are provided, the values stored in
        this class are used.
        """
        # pylint: disable=import-error
        JuliaPrecompiler().set_root(self._rms_path)
        JuliaPrecompiler().ensure_is_compiled()
        from julia import ReactionMechanismSimulator as rms
        from diffeqpy import de
        # pylint: enable=import-error
        start = datetime.now()
        phase = self._get_phase(rms_file_name, h, s, a, n, ea)
        initial_conditions = self.get_initial_conditions()

        # We could add more domains here: ConstantVDomain, ConstantPDomain, ParametrizedTPDomain,
        # ParametrizedVDomain, ParametrizedPDomain, ConstantTVDomain, ParametrizedTConstantVDomain,
        # ConstantTAPhiDomain
        # Variables created here:
        # volume: reactor volume
        # domain: RMS reactor domain object
        # y0: initial conditions
        # p: ODE parameters: (gibbs of each point, forward rate constants of each reaction)
        domain, y0, p, volume, initial_conditions = self._get_domain(phase, initial_conditions)
        reactor = rms.Reactor(domain, y0, (0.0, self.max_time), p=p)
        solution = de.solve(reactor.ode, resolve_rms_solver(self.solver, reactor),
                            abstol=self.abs_tol,
                            reltol=self.rel_tol)
        if not self.valid_model_solution(solution):
            return None, None, None, None, None
        # RMS just hangs here if multiprocessing is used.
        simulation = rms.Simulation(solution, domain)
        end = datetime.now()
        print("RMS Input file + Solving", end - start, rms_file_name)
        return simulation, reactor, volume, p, solution

    def calculate_adjoined_sensitivities(self, simulation, reactor, absolute_vertex_flux: np.ndarray,
                                         n_params: int) -> np.ndarray:
        """
        Run an adjoined sensitivity analysis for the microkinetic model.
        """
        # pylint: disable=import-error
        from julia import ReactionMechanismSimulator as rms
        # pylint: enable=import-error
        solver = resolve_rms_solver(self.solver, reactor)
        target_species = absolute_vertex_flux > float(self.settings["adjoined_sensitivity_threshold"])
        n_target = np.count_nonzero(target_species)
        all_sensitivities: np.ndarray = np.zeros((n_target, n_params))
        high_flux_aggregates = []
        abs_tolerance = float(self.settings["absolute_tolerance_sensitivity"])
        rel_tolerance = float(self.settings["relative_tolerance_sensitivity"])
        counter = 0
        for i, (a_id, a_type) in enumerate(zip(self.a_str_ids, self.aggregate_types)):
            if absolute_vertex_flux[i] > self.settings["adjoined_sensitivity_threshold"]\
                    and a_type == db.CompoundOrFlask.COMPOUND:
                # Calculate the adjoined sensitivities of the aggregate's concentration with respect to all
                # ODE parameters (parameter sorting: Gibb's free energies, forward reaction rates)
                # More information:
                # https://docs.sciml.ai/SciMLSensitivity/stable/manual/direct_adjoint_sensitivities/#SciMLSensitivity.adjoint_sensitivities
                # https://epubs.siam.org/doi/epdf/10.1137/S1064827501380630
                all_sensitivities[counter, :] = rms.getadjointsensitivities(simulation, a_id, solver,
                                                                            abstol=abs_tolerance, reltol=rel_tolerance)
                high_flux_aggregates.append(a_id)
                counter += 1

        abs_max_sensitivities = np.amax(np.abs(all_sensitivities), axis=0)
        if math.isnan(np.sum(abs_max_sensitivities)):
            raise RuntimeError("Error: NaN detected after sensitivity analysis. The ODE solver probably ran into"
                               " problems")
        return abs_max_sensitivities

    def calculate_fluxes_and_concentrations(self, rms_file_name: Optional[str] = None,
                                            time_points: Optional[List[float]] = None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Run the microkinetic modeling simulation and analyse the resulting concentration trajectories. A set of time
        points may be provided at which the concentrations for each aggregate is extracted. Default values for
        enthalpies, entropies, activation energies etc. are assumed (see class members).
        """
        if rms_file_name is None:
            rms_file_name = self._rms_file_name
        simulation, reactor, volume, p, solution = self.run_kinetic_modeling(rms_file_name)
        if simulation is None:
            raise RuntimeError("Numerical integration of the ODE failed. The system of ODEs may be ill conditioned.")
        c_max, c_final, absolute_vertex_flux, absolute_edge_flux, additional_c = self.integrate_results(
            simulation, volume, time_points, solution)
        abs_max_sens = None
        if self.calculate_adjoined:
            abs_max_sens = self.calculate_adjoined_sensitivities(simulation, reactor, absolute_vertex_flux, len(p))
        return c_max, c_final, absolute_vertex_flux, absolute_edge_flux, abs_max_sens, additional_c

    def valid_model_solution(self, solution: Any) -> bool:
        """
        Returns true if the solution (julia object) is valid for the kinetic model, i.e., the numerical integration
        of the ODE system was successful.
        """
        if all(solution.t < 0.99 * self.max_time):
            return False
        return True

    @staticmethod
    def concentrations(simulation, volume, time_points: Optional[List[float]] = None)\
            -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Getter for the concentrations, max concentrations, and final concentrations from a finished simulation run.
        Note that this functions assumes that Julia was imported previously.
        """
        # pylint: disable=import-error
        from julia import ReactionMechanismSimulator as rms
        # pylint: enable=import-error
        concentrations: np.ndarray = rms.concentrations(simulation)
        if concentrations.shape[1] == 1:
            raise RuntimeError("The numerical integration failed! The system of ODEs is probably ill conditioned.")
        concentrations *= volume
        maximum_concentrations = np.amax(concentrations, axis=1)
        final_concentrations = concentrations[:, -1]

        additional_c: Optional[np.ndarray] = None
        if time_points is not None and time_points:
            additional_c = np.array([rms.concentrations(simulation, t) for t in time_points])
            additional_c *= volume
        return maximum_concentrations, final_concentrations, additional_c

    def integrate_results(self, simulation, volume, times: Optional[List[float]] = None,
                          solution=None):
        """
        Integrate the absolute reaction rates to get vertex and edge fluxes and calculate final and maximum
        concentrations.
        """
        start = datetime.now()
        maximum_concentrations, final_concentrations, additional_c = self.concentrations(simulation, volume, times)
        absolute_edge_flux = self._calculate_absolute_edge_flux(simulation, self.max_time, solution)
        # The initial conditions parsed to RMS are understood as particle numbers N. The concentrations are
        # therefore c = N/V. We want to retrieve the particle number again and must multiply with V.
        # Note however, that V may not be constant during the kinetic modeling if not explicitly enforced.
        # For practical reasons we use the initial reactor volume here.
        absolute_vertex_flux = self._calculate_absolute_vertex_flux(absolute_edge_flux)
        end = datetime.now()
        print("Integration timing:", end - start)
        return maximum_concentrations, final_concentrations, absolute_vertex_flux, absolute_edge_flux, additional_c

    def _calculate_absolute_vertex_flux(self, absolute_edge_flux: np.ndarray):
        """
        Calculate the absolute vertex fluxes.
        """
        n_aggregates = len(self.a_str_ids)
        vertex_flux = np.zeros(n_aggregates)
        for i, reactants in enumerate(self.reactants):
            all_reactant_str_ids = [a_id for a_id in reactants[0] + reactants[1]]
            edge_flux = absolute_edge_flux[i]
            for a_str_id in all_reactant_str_ids:
                a_index = self.a_str_ids.index(a_str_id)
                vertex_flux[a_index] += edge_flux
        return vertex_flux

    def _calculate_absolute_edge_flux(self, simulation, t_max: float, solution=None):
        """
        Calculate the absolute edge fluxes.
        """
        # pylint: disable=import-error
        from julia import ReactionMechanismSimulator as rms
        # pylint: enable=import-error

        n_reactions = len(self.reactants)
        edge_flux = np.zeros(n_reactions, dtype=float)
        # The concentrations (usually) become smooth after the first few seconds. Therefore, we use a logarithmic
        # spacing of the integration points.
        if solution is None:
            n_steps = int(np.log10(t_max) * 5e+2)
            times: Union[np.ndarray, List[float]] = np.logspace(np.log10(1e-12), np.log10(t_max), num=n_steps,
                                                                dtype=float)
            rates = np.zeros((n_reactions, n_steps))
            for i, t in enumerate(times):
                rates[:, i] = rms.rates(simulation, min(t, t_max))
        else:
            raw_times = solution.t
            raw_rates = rms.rates(simulation)
            # Sometimes the ODE solver is "stuck" for a few steps on a tiny dt. This would lead to "nan" in the
            # flux integration. Therefore, we eliminate all dt =< 1e-12 before integrating.
            time_list = [i for i in range(len(raw_times)) if i == 0 or abs(raw_times[i] - raw_times[i - 1]) > 1e-12]
            times = [raw_times[i] for i in time_list]
            rates = np.transpose(np.array([raw_rates[:, i] for i in time_list]))
        for i in range(n_reactions):
            abs_rates_i = np.abs(rates[i, :])
            edge_flux[i] = integrate.simps(abs_rates_i, times)
        return edge_flux

    def get_aggregate_to_reaction_map(self):
        """
        Getter for the aggregate string id to reaction string id map.
        """
        if self._aggregates_to_reactions is None:
            self._aggregates_to_reactions = {a_id: [] for a_id in self.a_str_ids}
            for reactants, r_str_id in zip(self.reactants, self.r_str_ids):
                for a_id in reactants[0] + reactants[1]:
                    self._aggregates_to_reactions[a_id].append(r_str_id)
        return self._aggregates_to_reactions

    def translate_minimum_change_to_barriers(self, ea: np.ndarray, h: Union[List[float], np.ndarray],
                                             s: Union[List[float], np.ndarray], a_str_id: str) -> np.ndarray:
        """
        Ensure that changing the enthalpy of the aggregate with string id a_str_id did not lead to a negative reverse
        reaction barrier. If this is the case, the reaction barrier is increased to make the reverse reaction barrier
        zero.

        Parameters
        ----------
        ea : np.ndarray
            The activation energies (in J/mol).
        h : Union[List[float], np.ndarray]
            The enthalpies.
        s : Union[List[float], np.ndarray]
            The entropies.
        a_str_id : str
            The aggregate for which the enthalpy is changed.

        Returns
        -------
        np.ndarray
            Returns the updated activation energies.
        """
        for r_str_id in self.get_aggregate_to_reaction_map()[a_str_id]:
            r_index = self.r_str_ids.index(r_str_id)
            reactants = self.reactants[r_index]
            lhs_indices = [self.a_str_ids.index(a_id) for a_id in reactants[0]]
            rhs_indices = [self.a_str_ids.index(a_id) for a_id in reactants[1]]
            g_lhs = sum([h[i] - self.temperature * s[i] for i in lhs_indices])
            g_rhs = sum([h[i] - self.temperature * s[i] for i in rhs_indices])
            g_ts = g_lhs + ea[r_index]
            g_ts = max(g_ts, g_lhs, g_rhs)
            ea[r_index] = g_ts - g_lhs
        return ea

    def ensure_non_negative_barriers(self, ea: np.ndarray, h: Union[List[float], np.ndarray],
                                     s: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Ensure that a change in the enthalpies or activation energies did not lead to negative reverse reaction barriers
        for any reaction.

        Parameters
        ----------
        ea : np.ndarray
            The activation energies (in J/mol).
        h : Union[List[float], np.ndarray]
            The enthalpies.
        s : Union[List[float], np.ndarray]
            The entropies.

        Returns
        -------
        np.ndarray
            Returns the updated activation energies.
        """
        for a_str_id in self.a_str_ids:
            ea = self.translate_minimum_change_to_barriers(ea, h, s, a_str_id)
        return ea

    def get_n_aggregates(self, with_solvent: bool = True):
        """
        Get the number of aggregates in the RMS kinetic modeling. Note that this number may be larger than the
        number of input aggregates if a solvent species was added to the kinetic modeling. This additional species
        can be excluded by with_solvent=False
        """
        n_aggregates = len(self.h)
        if self.solvent_species_added and with_solvent:
            n_aggregates += 1
        return n_aggregates

    def get_n_reactions(self):
        """
        Getter for the number of reactions.
        """
        return len(self.ea)

    def get_n_parameters(self) -> int:
        """
        Getter for the total number of microkinetic model parameters.
        """
        return self.get_n_aggregates(with_solvent=False) + self.get_n_reactions()

    def get_all_parameters(self):
        """
        Getter for the full list of parameters.
        """
        full_parameters = np.empty(self.get_n_parameters())
        full_parameters[:self.get_n_aggregates(with_solvent=False)] = self.h
        full_parameters[self.get_n_aggregates(with_solvent=False):] = self.ea
        return full_parameters
