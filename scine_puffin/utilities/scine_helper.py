# -*- coding: utf-8 -*-
from __future__ import annotations
"""scine_helper.py: Collection of common procedures to be carried out in scine jobs"""
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Tuple, Union, Dict, TYPE_CHECKING, Any, Optional
import copy
import sys

from scine_puffin.config import Configuration
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class SettingsManager:
    """
    An interface for managing calculator and task specific settings in all puffin jobs.
    """

    def __init__(
        self,
        method_family: str,
        program: str,
        calculator_settings: Union[utils.Settings, None] = None,
        task_settings: Union[Dict[str, Any], None] = None,
    ) -> None:
        """
        Constructor of the SettingsManager.
        Initializes class with method_family and program to get the available settings for the calculator.

        Parameters
        ----------
        method_family : db.Model.method_family
            The method_family to be used in the calculator.
        program : db.Model.program
            The program to be used in the calculator.
        calculator_settings : utils.Settings
            Settings specific to the calculator. 'None' by default. This should be the full settings of the calculator.
        task_settings : dict
            Dictionary of settings specific to the task. 'None' by default.
        """

        # Get defaults
        self.default_calculator_settings = utils.core.get_available_settings(method_family, program)
        # Set the calculator settings to defaults if the parameter is 'None'; otherwise set the given settings
        if calculator_settings is None:
            self.calculator_settings = utils.Settings(
                "calc_settings",
                {"program": program, **self.default_calculator_settings},  # type: ignore
            )
        else:
            self.calculator_settings = utils.Settings("calc_settings",
                                                      {"program": program, **calculator_settings})  # type: ignore
        # Set the task settings to an empty dict if the parameter is 'None'; otherwise set the given dict
        self.task_settings = {} if task_settings is None else copy.deepcopy(task_settings)
        self.cores_were_set = False
        self.memory_was_set = False

    def setting_is_available(self, setting_key: str) -> bool:
        """
        Check, if given setting_key is a setting in the calculator

        Parameters
        ----------
        setting_key : list
            The settings_key to be checked.
        """

        if setting_key in self.default_calculator_settings.keys():
            return True
        else:
            sys.stderr.write(
                "WARNING: '"
                + setting_key
                + "' is not an available setting in the chosen calculator. "
                + "This setting will not be set in the calculator.\n"
            )
            return False

    def separate_settings(self, calculation_settings: utils.ValueCollection) -> None:
        """
        Extract calculator and task settings from the given calculation settings.
        Uses the information of the settings which are available for a calculator.

        Parameters
        ----------
        calculation_settings : dict
            The dictionary of the calculation settings (read from database).
        """
        self.calculator_settings.update(calculation_settings)
        for key, value in dict(calculation_settings).items():
            if key not in self.calculator_settings.keys():
                self.task_settings[key] = value
        self.cores_were_set = "external_program_nprocs" in calculation_settings
        self.memory_was_set = "external_program_memory" in calculation_settings

    def update_calculator_settings(self, structure: Union[None, db.Structure], model: db.Model, resources: dict) \
            -> None:
        """
        Update calculator settings with information about the structure, the model, the resources and
        the available_calculator settings.

        Parameters
        ----------
        structure : db.Structure
            The database structure object to be analysed.
        model : db.Model
            The database model object to be used.
        resources : dict
            Resources of the calculator (config['resources']).
        """
        # Update structure related settings
        if structure is not None:
            if self.setting_is_available(utils.settings_names.molecular_charge):
                self.calculator_settings[utils.settings_names.molecular_charge] = structure.get_charge()
            if self.setting_is_available(utils.settings_names.spin_multiplicity):
                self.calculator_settings[utils.settings_names.spin_multiplicity] = structure.get_multiplicity()

        model.complete_settings(self.calculator_settings)

        # Set external resources according to the resources available for the puffin.
        # Check, if the resources requested by the calculator settings do not exceed the available resources.
        # If they do, warn the user and reset the calculator settings to the maximum resources.
        if "external_program_nprocs" in self.calculator_settings.keys():
            set_cores = self.calculator_settings["external_program_nprocs"]
            # Check, if settings are requesting more than available
            if set_cores > int(resources["cores"]):  # type: ignore
                sys.stderr.write(
                    "WARNING: Do not request more than you can chew.\n 'external_program_nprocs' is "
                    "reset to " + str(resources["cores"]) + ".\n"
                )
                self.calculator_settings["external_program_nprocs"] = int(resources["cores"])
            # set puffin resources if the resources were not specified in the calculation settings in the database
            elif not self.cores_were_set:
                self.calculator_settings["external_program_nprocs"] = int(resources["cores"])

        if "external_program_memory" in self.calculator_settings.keys():
            set_memory = int(self.calculator_settings["external_program_memory"])  # type: ignore
            # Check, if settings are requesting more than available
            if set_memory > int(resources["memory"] * 1024):
                sys.stderr.write(
                    "WARNING: Do not request more than you can chew.\n 'external_program_memory' is "
                    "reset to " + str(int(resources["memory"] * 1024)) + " MB.\n"
                )
                self.calculator_settings["external_program_memory"] = int(resources["memory"] * 1024)
            # set puffin resources if the resources were not specified in the calculation settings in the database
            elif not self.memory_was_set:
                self.calculator_settings["external_program_memory"] = int(resources["memory"] * 1024)

    def prepare_readuct_task(
        self,
        structure: db.Structure,
        calculation: db.Calculation,
        settings: utils.ValueCollection,
        resources: dict,
    ) -> Tuple[Dict[str, Optional[utils.core.Calculator]], List[str]]:
        """
        Constructs a dictionary with a Scine Calculator based on the calculation settings and resources to have an
        input for a ReaDuct task

        Parameters
        ----------
        structure : db.Structure
            The database structure object to be analysed.
        calculation : db.Calculation
            The database calculation that shall be calculated with ReaDuct.
        settings : utils.ValueCollection
            The settings of the database calculation.
        resources : dict
            Resources of the calculator (config['resources']).

        Returns
        -------
        Tuple[dict, List[str]]]
            A tuple containing the dictionary containing the calculator and a list containing the corresponding key
        """
        # Separate the calculation settings from the database into the task and calculator settings
        # This overwrites any default settings by user settings
        self.separate_settings(settings)
        # Update the calculator settings
        # Warnings concerning external program resources are written into the stderr
        self.update_calculator_settings(structure, calculation.get_model(), resources)
        self.correct_non_applicable_settings()

        utils.io.write("system.xyz", structure.get_atoms())
        system = utils.core.load_system_into_calculator(
            "system.xyz", calculation.get_model().method_family, **self.calculator_settings
        )
        return {"system": system}, ["system"]

    def correct_non_applicable_settings(self):
        """
        Changes calculator settings that are not applicable in a Puffin execution
        """
        # base working directory might have received a wrong default value depending on the init of the settings manager
        # removing it ensures that Readuct receives the correct default inside the job directory
        if "base_working_directory" in self.calculator_settings:
            del self.calculator_settings["base_working_directory"]


def update_model(
    calculator: utils.core.Calculator,
    calculation: db.Calculation,
    config: Configuration,
) -> db.Model:
    """
    Updates the model of the given calculation both based on its results and the settings of the actual calculator

    Parameters
    ----------
    calculator : utils.core.Calculator
        The calculator that performed the calculation.
    calculation : db.Calculation
        The database calculation that is currently run.
    config : scine_puffin.config.Configuration
       The current configuration of the Puffin necessary for the program's version.
    """
    model = calculation.get_model()
    # Update with program
    results = calculator.get_results()
    if results.program_name:
        model.program = results.program_name.lower()
        model.version = config.programs()[model.program]["version"]
    # update with calculator settings
    model.complete_model(calculator.settings)
    calculation.set_model(model)
    return model
