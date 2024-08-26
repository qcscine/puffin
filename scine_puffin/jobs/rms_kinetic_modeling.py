# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from multiprocessing import Pool
from typing import Optional, List, TYPE_CHECKING, Any, Dict

import numpy as np

from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.kinetic_modeling_jobs import KineticModelingJob
from scine_puffin.config import Configuration
from ..utilities.rms_kinetic_model import RMSKineticModel
from ..utilities.kinetic_modeling_sensitivity_analysis import RMSKineticModelingSensitivityAnalysis
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class RmsKineticModeling(KineticModelingJob):
    """
    Micro-kinetic modeling with the puffin-interface to the reaction mechanism simulator (RMS).
    Note: Running jobs with RMS as a backend requires an installation of RMS (including its Python bindings). This
    is not supported through the Puffin bootstrapping. See programs/rms.py for more information.

    **Order Name**
      ``rms_kinetic_modeling``

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
      ea : List[float]
        The activation energies for each reaction as the free energy difference to the reaction LHS (in J/mol).
      enthalpies : List[float]
        The enthalpy of each aggregate (in J/mol).
      entropies : List[float]
        The entropy of each aggregate (in J/mol).
      arrhenius_prefactors : List[float]
        The exponential prefactors.
      arrhenius_temperature_exponents : List[float]
        The temperature exponents in the Arrhenius equation.
      start_concentrations : List[float
        The start concentrations of each aggregate.

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.

      The following options are available:

      solver : str
        ODE solver. Currently only "CVODE_BDF" is supported.
      phase_type : str
        The reactor phase. Options are ideal_gas (assumes P=const, T=const), ideal_dilute_solution
        (assumes V=const, T=const). Default is "ideal_gas".
      max_time : float
        Maximum integration time in seconds. Default 3600.0.
      energy_model_program : str
        The program with which the electronic structure model should be flagged. Default any.
      viscosity : float
        The solvent viscosity (in Pa s). Needs phase=ideal_dilute_solution and diffusion_limited=true. If "none", the
        viscosity is taken from tabulated values.
      reactor_solvent : str
        The reactor solvent. If "none", the solvent in the electronic structure model is used if any.
      site_density : float
        The density of surface sites. Default is "none". Requires phase=ideal_surface. Not fully supported yet.
      diffusion_limited : bool
        If true, diffusion limits are enforced. Requires phase=ideal_dilute_solution. May lead to numerical
        instability of the ODE solver. Default False.
      reactor_temperature : float
        The reactor temperature (in K). If "none", the temperature in the model is used. Default "none".
      reactor_pressure : float
        The reactor pressure (in Pa). If none, the pressure in the model is used. Default "none".
      absolute_tolerance : float
        The absolute tolerance of the ODE solver. High values lead to a faster but less reliable integration. Default
        1e-20.
      relative_tolerance : float
        The relative tolerance of the ODE solver. High values lead to a faster but less reliable integration.
        Default 1e-6.
      solvent_aggregate_str_id : str
        The aggregate ID of the solvent as a string. If "none", the solvent is assumed to be unreactive.
      solvent_concentration : float
        The solvent concentration. Default is 55.3 (mol/L).
      enforce_mass_balance : bool
        If true, an error is raised for any non-balanced reaction.
      screen_sensitivities : bool
        If true, only parameters associated to aggregates and reactions with significant concentration flux are
        considered in the sensitivity analysis (flux > oaat_vertex_flux_threshold | flux > oaat_vertex_flux_threshold).

    **Required Packages**
      - SCINE: Database (present by default)
      - rms

    **Generated Data**
      If successful (technically and chemically) the following data will be
      generated and added to the database:

      Properties
        The maximum and final concentration, and the vertex flux of each aggregate is added to
        its centroid. The edge flux for each reaction is added to the centroid of the first aggregate on the reaction's
        LHS. Note, that the properties are NOT listed in the results to avoid large DB documents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "RMS kinetic modeling job"
        self.settings: Dict[str, Any] = {
            "solver": "Recommended",
            "phase_type": "ideal_gas",
            "max_time": 3600.0,
            "energy_model_program": "any",
            "viscosity": "none",  # in Pa s
            "reactor_solvent": "none",
            "site_density": "none",
            "diffusion_limited": False,
            "reactor_temperature": "none",  # in K
            "reactor_pressure": "none",  # in Pa,
            "absolute_tolerance": 1e-20,
            "relative_tolerance": 1e-6,
            "solvent_aggregate_str_id": "none",
            "solvent_concentration": 55.3,
            "sensitivity_analysis": "none",
            "adjoined_sensitivity_threshold": 1e-2,
            "absolute_tolerance_sensitivity": 1e-6,
            "relative_tolerance_sensitivity": 1e-3,
            "oaat_vertex_flux_threshold": 1e-2,
            "oaat_edge_flux_threshold": 1e-2,
            "sample_size": 10,
            "local_sensitivities": False,
            "adjoined_sensitivities": False,
            "save_oaat_var": False,
            "distribution_shape": "truncnorm",
            "enforce_mass_balance": True,
            "screen_global_sens_size": 1e+3
        }
        self.model: db.Model = db.Model("PM6", "PM6", "")
        self._rms_file_name: str = "chem.rms"
        self._solvent_a_index: Optional[int] = None
        self._viscosity: Optional[float] = None
        self._solvent: Optional[str] = None
        self._site_density: Optional[float] = None
        self._pressure: Optional[float] = None
        self._temperature: Optional[float] = None
        self._phase_type: Optional[str] = None
        self._phase_options = ["ideal_dilute_solution", "ideal_gas"]
        self._sensitivity_options = ["none", "adjoined_sensitivities", "one_at_a_time_sensitivities", "morris", "sobol"]
        self._solvent_aggregate_str_id: Optional[str] = None
        self._solvent_species_added: bool = False
        self.reaction_ids: List[db.ID] = []
        self.aggregate_id_list: List[db.ID] = []
        self.aggregate_types: List[db.CompoundOrFlask] = []
        self.max_time: float = float(self.settings["max_time"])
        self.abs_tol: float = float(self.settings["absolute_tolerance"])
        self.rel_tol: float = float(self.settings["relative_tolerance"])
        self._sample_size: int = 50
        self._n_cores: int = 1
        self.force_parallel = False

        self.rms_kinetic_model: Optional[RMSKineticModel] = None

    def use_n_cores(self, n_cores: int) -> int:
        assert isinstance(self.rms_kinetic_model, RMSKineticModel)
        if self.force_parallel:
            return n_cores
        if self.rms_kinetic_model.get_n_parameters() > 100:
            return n_cores
        if self.settings["sensitivity_analysis"] in ["morris", "sobol"]:
            return n_cores
        return 1

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "rms"]

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        with breakable(calculation_context(self)):
            self._calculation = calculation
            self.settings.update(calculation.get_settings())
            self._resolve_default_settings()
            rms_path = config["programs"]["rms"]["root"]
            self.rms_kinetic_model = RMSKineticModel(self.settings, manager, self.model, rms_path, self._rms_file_name)
            # Importing Julia has a significant overhead. If we parallelize we have to import Julia potentially more
            # than once (i.e., once in each sub process). Therefore, we should only do this if the kinetic model is
            # large enough that the parallelization actually accelerates the calculation.
            self._n_cores = self.use_n_cores(int(config["resources"]["cores"]))
            if self._n_cores > 1:
                # In the case we parallelize the kinetic modeling, we must ensure that Julia is only imported in the
                # worker processes and NEVER in the main process.
                with Pool(1) as pool:
                    res = pool.map(self.rms_kinetic_model.calculate_fluxes_and_concentrations, [self._rms_file_name])
            else:
                res = [self.rms_kinetic_model.calculate_fluxes_and_concentrations(self._rms_file_name)]
            c_max, c_final, c_flux, r_flux, adjoined_sens, _ = res[0]

            print("Maximum concentrations")
            print(c_max)
            print("Final concentrations")
            print(c_final)
            print("Absolute edge flux")
            print(r_flux)
            print("Absolute vertex flux")
            print(c_flux)

            results = calculation.get_results()
            if self.settings["sensitivity_analysis"] == "one_at_a_time_sensitivities"\
                    or self.settings['local_sensitivities']:
                vertex_t = float(self.settings["oaat_vertex_flux_threshold"])
                edge_t = float(self.settings["oaat_edge_flux_threshold"])
                flux_replace = 10.0 * vertex_t
                sens = RMSKineticModelingSensitivityAnalysis(self.rms_kinetic_model, self._n_cores, self._sample_size)
                max_sens, sens_c_final, flux_sens, var_max, var_final, var_flux = sens.one_at_a_time_differences(
                    c_flux, r_flux, vertex_t, edge_t, flux_replace, c_max, c_final)

                self._write_sensitivities_to_database(flux_sens, "oaat_flux")
                self._write_sensitivities_to_database(max_sens, "oaat_max")
                self._write_sensitivities_to_database(sens_c_final, "oaat_final")
                if self.settings["save_oaat_var"]:
                    self._write_concentrations_to_centroids(self.aggregate_id_list, self.aggregate_types,
                                                            self.reaction_ids, [var_final, var_max, var_flux], [],
                                                            ["var_final_c", "var_max_c", "var_flux_c"], [],
                                                            results)
            if adjoined_sens is not None:
                print("Adjoined Sensitivities")
                print(adjoined_sens)
                self._write_sensitivities_to_database(adjoined_sens, "adjoined")
            salib = RMSKineticModelingSensitivityAnalysis(self.rms_kinetic_model, self._n_cores, self._sample_size,
                                                          self.settings["distribution_shape"])
            if self.settings["sensitivity_analysis"] == "morris":
                if self.settings["screen_global_sens_size"] < salib.get_n_parameters():
                    vertex_t = float(self.settings["oaat_vertex_flux_threshold"])
                    edge_t = float(self.settings["oaat_edge_flux_threshold"])
                    salib.set_prescreening_condition(c_flux, r_flux, vertex_t, edge_t)
                mu, mu_star, sigma, _ = salib.morris_sensitivities()
                m_v = salib.analyse_runs()
                self._write_sensitivities_to_database(mu['c_max'], "morris_mu_c_max")
                self._write_sensitivities_to_database(mu['c_final'], "morris_mu_c_final")
                self._write_sensitivities_to_database(mu_star['c_max'], "morris_mu_star_c_max")
                self._write_sensitivities_to_database(mu_star['c_final'], "morris_mu_star_c_final")
                self._write_sensitivities_to_database(sigma['c_max'], "morris_sigma_c_max")
                self._write_sensitivities_to_database(sigma['c_final'], "morris_sigma_c_final")
                self._write_concentrations_to_centroids(self.aggregate_id_list, self.aggregate_types, self.reaction_ids,
                                                        [m_v[0][0], m_v[1][0], m_v[2][0], m_v[0][1], m_v[1][1],
                                                         m_v[2][1]], [], ["morris_mean_c_max", "morris_mean_c_final",
                                                                          "morris_mean_c_flux", "morris_var_c_max",
                                                                          "morris_var_c_final", "morris_var_c_flux"],
                                                        [], results)

            elif self.settings["sensitivity_analysis"] == "sobol":
                st, s1, _ = salib.sobol_sensitivities()
                m_v = salib.analyse_runs()
                self._write_sensitivities_to_database(st['c_max'], "sobol_st_c_max")
                self._write_sensitivities_to_database(st['c_final'], "sobol_st_c_final")
                self._write_sensitivities_to_database(s1['c_max'], "sobol_s1_c_max")
                self._write_sensitivities_to_database(s1['c_final'], "sobol_s1_c_final")
                self._write_concentrations_to_centroids(self.aggregate_id_list, self.aggregate_types, self.reaction_ids,
                                                        [m_v[0][0], m_v[1][0], m_v[2][0], m_v[0][1], m_v[1][1],
                                                         m_v[2][1]], [], ["sobol_mean_c_max", "sobol_mean_c_final",
                                                                          "sobol_mean_c_flux", "sobol_var_c_max",
                                                                          "sobol_var_c_final", "sobol_var_c_flux"],
                                                        [], results)

            self._write_concentrations_to_centroids(self.aggregate_id_list, self.aggregate_types, self.reaction_ids,
                                                    [c_max, c_final, c_flux], [r_flux],
                                                    [self.c_max_label, self.c_final_label, self.c_flux_label],
                                                    [self.r_flux_label], results)
            self._disable_all_aggregates()
            self.complete_job()
        return self.postprocess_calculation_context()

    def _write_sensitivities_to_database(self, absolute_sensitivities: np.ndarray, prop_label: str):
        counter = 0
        results = self._calculation.get_results()
        for a_id, a_type in zip(self.aggregate_id_list, self.aggregate_types):
            centroid = self._get_aggregate_centroid(a_id, a_type)
            label = "max_free_energy_sensitivity_" + prop_label
            self._write_concentration_property(centroid, label, absolute_sensitivities[counter], results)
            counter += 1
        # The last species free energy sensitivity may be for a potentially added solvent.
        if absolute_sensitivities.shape[0] > len(self.reaction_ids) + len(self.aggregate_id_list):
            counter += 1
        for r_id in self.reaction_ids:
            centroid = self._get_reaction_centroid(r_id)
            label = r_id.string() + "_reaction_barrier_sensitivity_" + prop_label
            self._write_concentration_property(centroid, label, absolute_sensitivities[counter], results)
            counter += 1

    def _resolve_default_settings(self):
        self.model = self._calculation.get_model()
        self.model.program = self.settings["energy_model_program"]
        self._sample_size = self.settings["sample_size"]

        self.reaction_ids = [db.ID(r_id_str) for r_id_str in self.settings["reaction_ids"]]
        self.aggregate_id_list = [db.ID(c_id_str) for c_id_str in self.settings["aggregate_ids"]]
        self.aggregate_types = [db.CompoundOrFlask(a_type) for a_type in self.settings["aggregate_types"]]
