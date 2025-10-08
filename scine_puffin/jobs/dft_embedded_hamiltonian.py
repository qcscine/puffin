# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List, Optional, Dict
import numpy as np
import sys

from scine_puffin.jobs.templates.job import is_configured
from scine_puffin.jobs.scine_single_point import ScineSinglePoint
from scine_puffin.utilities.program_helper import ProgramHelper
from scine_puffin.utilities.imports import module_exists, MissingDependency
from scine_puffin.utilities.scine_helper import SettingsManager

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class DftEmbeddedHamiltonian(ScineSinglePoint):
    """
    This jobs saves a QM/QM embedding Hamiltonian after a standard single point calculation with a QM/QM/MM model.
    All settings from the scine_single_point job apply.

    **Order Name**
      ``dft_embedded_hamiltonian``

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - Serenity

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The ``core_coefficients_x`` orbital coefficients for quantum core x.
        The ``h_core__x`` core Hamiltonian for quantum core x.
        The ``fragment_charges_x`` orbital populations on the quantum core for core x.
        Partial energies for the quantum core.
    """

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "serenity", "utils"]

    @is_configured
    def sp_preprocessing(self, settings_manager: SettingsManager, structure: db.Structure, resources: Dict) -> None:
        settings = self._calculation.get_settings()
        if "cas_systems" not in settings:
            raise RuntimeError("No system for CAS calculation defined!")

        if "use_turbomole_mos" in settings:
            import scine_readuct as readuct
            # 1. Get the supersystem electronic structure method.
            method_family_list = self._calculation.get_model().method_family.lower().split("/")
            method_family_list.pop(0)
            method_list = self._calculation.get_model().method.lower().split("/")
            method_list.pop(0)
            # The calculation must at least be a QM/QM calculations, ensuring that there are at least two
            # elements separated by /
            program_list = self._calculation.get_model().program.lower().split("/")
            program_list.pop(0)
            program_list.pop(0)
            supersystem_method_family = "/".join(method_family_list)
            supersystem_method = "/".join(method_list)
            program = "/".join(["turbomole"] + program_list)
            turbomole_settings_manager = SettingsManager(supersystem_method_family, program)

            default_settings = utils.core.get_available_settings(supersystem_method_family, program)
            settings_dict = {
                key: value for key, value in settings_manager.calculator_settings.items() if key in default_settings
            }
            settings_dict["program"] = program
            settings_dict["method"] = supersystem_method
            turbomole_settings_manager.calculator_settings = utils.Settings("calc_settings", settings_dict)
            turbomole_settings_manager.task_settings = {
                "require_charges": False
            }
            model = self._calculation.get_model()
            model.program = program
            model.method_family = supersystem_method_family
            model.method = supersystem_method
            systems, keys = turbomole_settings_manager.prepare_readuct_task(
                    structure, self._calculation, utils.ValueCollection({}), resources, model)
            systems, success = readuct.run_sp_task(systems, keys, **turbomole_settings_manager.task_settings)
            if not success:
                raise RuntimeError("Initial Turbomole calculation failed.")

            calculator = self.get_calc(keys[0], systems)
            results = calculator.get_results()
            if results.file_paths is None:
                raise RuntimeError("Unable to determine the MO file path after Turbomole calculation."
                                   " An outdated scine_utilities version may have been used.")
            settings_manager.calculator_settings["external_supersystem_mo_file"] = results.file_paths["directory"]
            settings_manager.calculator_settings["external_supersystem_mo_type"] = "turbomole"

    def get_underlying_calculators(self, calculator: utils.core.Calculator) -> List[utils.core.Calculator]:
        embedding_calculator_names = ["QMMM", "QMQM"]
        if calculator.name() in embedding_calculator_names:
            underlying_calculators = calculator.to_embedded_calculator().get_underlying_calculators()
            to_return = []
            for underlying_calculator in underlying_calculators:
                to_return += self.get_underlying_calculators(underlying_calculator)
            return to_return
        else:
            return [calculator]

    @is_configured
    def sp_postprocessing(
            self,
            success: bool,
            systems: Dict[str, Optional[utils.core.Calculator]],
            keys: List[str],
            structure: db.Structure,
            program_helper: Optional[ProgramHelper],
    ) -> None:
        super().sp_postprocessing(success, systems, keys, structure, program_helper)
        program = self.get_model().program
        method_family = self.get_model().method_family
        if "SERENITY" not in program.upper():
            sys.stderr.write(
                "Warning: Writing an embedding Hamiltonian to the database requires serenity as the QM/QM program.")
            return
        if "HF/" not in method_family.upper():
            sys.stderr.write(
                "Warning: Writing an embedding Hamiltonian to the database requires a QM/QM model using Hartree-Fock"
                " for at least one quantum core.")
            return

        calc = self.get_calc(keys[0], systems)
        base_calculators = self.get_underlying_calculators(calc)
        print("N base calculators:", len(base_calculators))
        for i, calculator in enumerate(base_calculators):
            print("Calculator name:", calculator.name())
            if calculator.name() != "SerenityHFCalculator":
                continue
            results = calculator.get_results()
            if (results.one_electron_matrix is None or results.coefficient_matrix is None
                    or results.partial_energies is None or results.orbital_fragment_populations is None):
                raise RuntimeError("QM/QM calculation did not provide the results necessary to write the embedded"
                                   " Hamiltonian to the database.\n"
                                   + "One electron matrix:" + str(results.one_electron_matrix is not None) + "\n"
                                   + "Coefficient matrix:" + str(results.coefficient_matrix is not None) + "\n"
                                   + "Partial energies matrix:" + str(results.partial_energies is not None) + "\n"
                                   + "Orbital fragment populations:"
                                   + str(results.orbital_fragment_populations is not None) + "\n")

            is_open_shell = self.get_model().spin_mode != "restricted"
            coefficients = results.coefficient_matrix.alpha_matrix\
                if is_open_shell else results.coefficient_matrix.restricted_matrix
            self.store_property(
                self._properties,
                "core_coefficients_" + str(i),
                "DenseMatrixProperty",
                coefficients,
                self.get_model(),
                self._calculation,
                structure,
            )

            h_core = results.one_electron_matrix
            self.store_property(
                self._properties,
                "h_core_" + str(i),
                "DenseMatrixProperty",
                h_core,
                self.get_model(),
                self._calculation,
                structure,
            )

            n_valence_orbitals = np.shape(coefficients)[1]
            orbital_populations = results.orbital_fragment_populations.alpha_matrix[:, :n_valence_orbitals]
            self.store_property(
                self._properties,
                "fragment_charges_" + str(i),
                "DenseMatrixProperty",
                orbital_populations,
                self.get_model(),
                self._calculation,
                structure,
            )

            for partial_energy_name, value in results.partial_energies.items():
                self.store_property(
                    self._properties,
                    partial_energy_name + "_frag_" + str(i),
                    "NumberProperty",
                    value,
                    self._calculation.get_model(),
                    self._calculation,
                    structure
                )
