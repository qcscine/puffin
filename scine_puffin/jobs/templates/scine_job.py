# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Type

from .job import Job, is_configured
from scine_puffin.utilities.scine_helper import SettingsManager, update_model
from scine_puffin.utilities.program_helper import ProgramHelper
from scine_puffin.utilities.transfer_helper import TransferHelper
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class CalculatorNotPresentException(Exception):
    """
    Exception to be raised when a calculator is not present in a system dictionary.
    """


class ScineJob(Job, ABC):
    """
    A common interface for all jobs in Puffin that use the
    Scine::Core::Calculator interface.
    This interface holds basic logic for interacting with Scine classes, which should be used in every ScineJob
    This interface also holds subclasses that include elementary workflows, which can be combined to build complex jobs
    """

    own_expected_results: List[str] = []  # to be overwritten by child class

    def __init__(self) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        # to be added by child class:
        self.properties_to_transfer = [
            "surface_atom_indices",
            "slab_dict",
        ]
        self._fallback_error = "Error: " + self.name + " failed with an unspecified error."

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "readuct", "utils"]

    @classmethod
    def expected_results(cls) -> List[str]:
        """
        Gives a list of str specifying the results expected to be present for a system within a job based on the
        class member and all its parents.

        Returns
        -------
        expected_results : List[str]
            The results to be expected as str corresponding to the members of the Scine::Utils::Results class.
        """
        parent_expects = []
        for Parent in cls.__bases__:
            if hasattr(Parent, "expected_results"):
                parent_expects += Parent.expected_results()  # pylint: disable=no-member
        return list(set(parent_expects + cls.own_expected_results))

    @is_configured
    def create_helpers(self, structure: db.Structure) -> Tuple[SettingsManager, Optional[ProgramHelper]]:
        """
        Creates a Scine SettingsManager and ProgramHelper based on the configured job and the given structure.
        The ProgramHelper is None if no ProgramHelper is specified for the specified program or no program was
        specified in the model

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        structure : db.Structure (Scine::Database::Structure)
            The structure on which a Calculation is performed.

        Returns
        -------
        helper_tuple : Tuple[SettingsManager, Optional[ProgramHelper]
            A tuple of the SettingsManager for Scine Calculators and ProgramHelper if available.
        """
        model = self._calculation.get_model()
        program = model.program if model.program.lower() != "any" else ""
        settings_manager = SettingsManager(model.method_family, program)
        program_helper = ProgramHelper.get_correct_helper(program, self._manager, structure, self._calculation)
        return settings_manager, program_helper

    def raise_named_exception(self, error_message: str, exception_type: Type[BaseException] = BaseException) -> None:
        """
        Raises an error including the name/description of the current job.
        """
        error_begin = "Error: " + self.name + " failed with message:\n"
        raise exception_type(error_begin + error_message)

    @is_configured
    def throw_if_not_successful(
        self,
        success: bool,
        systems: Dict[str, Optional[utils.core.Calculator]],
        keys: List[str],
        expected_results: Optional[List[str]] = None,
        sub_task_error_line: str = "",
    ) -> None:
        """
        Throw an exception if some calculations results are unexpected.

        Notes
        -----
        * Requires run configuration
        * Will throw Exception or do nothing

        Parameters
        ----------
        success : bool
            Whether the calculation was successful (either forwarded from readuct task or specified if not relevant).
        systems : Dict[str, Optional[utils.core.Calculator]] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys : List[str]
            The list of keys of the systems dict to be checked.
        expected_results : Optional[List[str]]
            The results to be required to be present in systems to qualify as successful calculations. If None is
            given, this defaults to the expected results of the class, see expected_results().
        sub_task_error_line : str
            An additional line for the error message to specify in which subtask the calculation crashed.
        """
        self.verify_connection()
        error_begin = "Error: " + self.name + " failed with message:\n"
        if sub_task_error_line:
            error_begin += sub_task_error_line
        if not success:
            error_message = self.expected_results_check(
                systems, keys, expected_results
            )[1]
            error_message = error_begin + error_message
            raise BaseException(error_message)

    @is_configured
    def calculation_postprocessing(
        self,
        success: bool,
        systems: Dict[str, Optional[utils.core.Calculator]],
        keys: List[str],
        expected_results: Optional[List[str]] = None,
    ) -> db.Results:
        """
        Performs a verification protocol that a Scine Calculation was successful. If not throws an exception,
        if yes, the model is updated and the cleared db.Results object of the configured calculation is returned to
        be added with actual results.

        Notes
        -----
        * Requires run configuration
        * May throw Exception

        Parameters
        ----------
        success : bool
            Whether the calculation was successful (either forwarded from readuct task or specified if not relevant).
        systems : Dict[str, utils.core.Calculator] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys : List[str]
            The list of keys of the systems dict to be checked.
        expected_results : Optional[List[str]]
            The results to be required to be present in systems to qualify as successful calculations. If None is
            given, this defaults to the expected results of the class, see expected_results().
        """
        self.verify_connection()  # check valid connection
        results_check, results_error = self.expected_results_check(
            systems, keys, expected_results
        )  # check results
        if not results_check:
            self.raise_named_exception(results_error)
        if not success:
            self.raise_named_exception(self._fallback_error)
        update_model(
            self.get_calc(keys[0], systems), self._calculation, self.config
        )  # calculation is safe -> update model
        db_results = self._calculation.get_results()
        db_results.clear()
        self._calculation.set_results(db_results)
        return db_results

    @is_configured
    def prepend_to_comment(self, message: str) -> None:
        """
        Prepends given message to the comment field of the currently configured calculation.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        message : str
            The message to be prepended.
        """
        comment = self._calculation.get_comment()
        self._calculation.set_comment(message + comment)

    @is_configured
    def expected_results_check(
        self,
        systems: Dict[str, Optional[utils.core.Calculator]],
        keys: List[str],
        expected_results: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Checks the results of the given systems based on the expected results. If the expected results are not given,
        they default to the expected results of the class, see expected_results().
        Throws exception if expected result is not present.

        Notes
        -----
        * Requires run configuration
        * May throw Exception

        Parameters
        ----------
        systems : Dict[str, Optional[utils.core.Calculator]] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys : List[str]
            The list of keys of the systems dict to be checked.
        expected_results : Optional[List[str]]
            The results to be required to be present in systems to qualify as successful calculations. If None is
            given, this defaults to the expected results of the class, see expected_results().

        Returns
        -------
        Tuple[bool, str]
            Whether the results are correct and an error message, describing failure in expected results.
        """
        if expected_results is None:
            expected_results = self.expected_results()
        # check all specified systems
        for key in keys:
            if key not in systems:
                return False, (key + " is missing in systems!")
            if systems[key] is None:
                return False, ""
            # check if desired results are present
            calc = self.get_calc(key, systems)
            if not calc.has_results():
                return False, ("System '" + key + "' is missing results!")
            results = calc.get_results()
            for expected in expected_results:
                if getattr(results, expected) is None:
                    return False, (expected + " is missing in results!")
        return True, ""

    @is_configured
    def store_energy(self, system: utils.core.Calculator, structure: db.Structure) -> None:
        """
        Stores an 'electronic_energy' property for the given structure based on the energy in the results of the given
        system. Does not perform checks.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        system : utils.core.Calculator (Scine::Core::Calculator)
            A Scine calculator holding a results object with the energy property.
        structure : db.Structure (Scine::Database::Structure)
            A structure for which the property is saved.
        """
        self.store_property(
            self._properties,
            "electronic_energy",
            "NumberProperty",
            system.get_results().energy,
            self._calculation.get_model(),
            self._calculation,
            structure,
        )

    @is_configured
    def store_bond_orders(self, bond_orders: utils.BondOrderCollection, structure: db.Structure) -> None:
        self.store_property(
            self._properties,
            "bond_orders",
            "SparseMatrixProperty",
            bond_orders.matrix,
            self._calculation.get_model(),
            self._calculation,
            structure,
        )

    def transfer_properties(self, old_structure: db.Structure, new_structure: db.Structure,
                            transfer_helper: Optional[TransferHelper] = None) -> None:
        """
        Copies property IDs from one structure to another one based on the specified properties in the class member.

        Parameters
        ----------
        old_structure : db.Structure (Scine::Database::Structure)
            The structure holding the properties. If a specified property is not present for the structure,
            no error is given.
        new_structure : db.Structure (Scine::Database::Structure)
            The structure for which the properties are to be added.
        transfer_helper : Optional[TransferHelper]
            An optional helper for more difficult transfer task. Otherwise, the specified properties are just copied.
        """
        properties_to_transfer = list(set(self.properties_to_transfer))  # make sure no duplicates
        if transfer_helper is None:
            for prop in properties_to_transfer:
                TransferHelper.simple_transfer(old_structure, new_structure, prop)
        else:
            transfer_helper.transfer_properties(old_structure, new_structure, properties_to_transfer)

    @is_configured
    def sp_postprocessing(
        self,
        success: bool,
        systems: Dict[str, Optional[utils.core.Calculator]],
        keys: List[str],
        structure: db.Structure,
        program_helper: Optional[ProgramHelper],
    ) -> None:
        """
        Performs a verification and results saving protocol for a Scine Single Point Calculation.

        Notes
        -----
        * Requires run configuration
        * May throw Exception

        Parameters
        ----------
        success : bool
            Whether the calculation was successful (either forwarded from
            readuct task or specified if not relevant).
        systems : Dict[str, utils.core.Calculator] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys : List[str]
            The list of keys of the systems dict to be checked.
        structure : db.Structure (Scine::Database::Structure)
            The structure on which the calculation was performed.
        program_helper : ProgramHelper
            The possible ProgramHelper that may also want to do some
            postprocessing after a calculation.
        """

        # postprocessing of results with sanity checks
        self.calculation_postprocessing(success, systems, keys)

        # handle properties
        calc = self.get_calc(keys[0], systems)
        self.store_energy(calc, structure)
        results = calc.get_results()
        # Store atomic charges if available
        if results.atomic_charges is not None:
            self.store_property(
                self._properties,
                "atomic_charges",
                "VectorProperty",
                results.atomic_charges,
                self._calculation.get_model(),
                self._calculation,
                structure,
            )
        # Store orbital energies if available
        if results.orbital_energies is not None:
            from scine_puffin.utilities.properties import single_particle_energy_to_matrix

            mat = single_particle_energy_to_matrix(results.orbital_energies)
            self.store_property(
                self._properties,
                "orbital_energies",
                "DenseMatrixProperty",
                mat,
                self._calculation.get_model(),
                self._calculation,
                structure,
            )
        # Store gradients if available (because it was requested)
        if results.gradients is not None:
            self.store_property(
                self._properties,
                "gradients",
                "DenseMatrixProperty",
                results.gradients,
                self._calculation.get_model(),
                self._calculation,
                structure,
            )

        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, structure)

    @is_configured
    def get_calculation(self) -> db.Calculation:
        """
        Getter for the current calculation. Throws if not configured.

        Notes
        -----
        * Requires run configuration
        * May throw Exception

        Returns
        -------
        calculation : db.Calculation (Scine::Database::Calculation)
            The current calculation being carried out.
        """
        if self._calculation is None:
            self.raise_named_exception("Job is not configured and does not hold a calculation right now")
        return self._calculation

    @requires("database")
    def create_new_structure(self, calculator: utils.core.Calculator, label: db.Label) -> db.Structure:
        """
        Add a new structure to the database based on the given calculator and label.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        calculator : utils.core.Calculator
            The calculator holding the optimized structure
        label : db.Label
            The label of the new structure

        Returns
        -------
        db.Structure
            The new structure
        """
        # New structure
        new_structure = db.Structure()
        new_structure.link(self._structures)
        new_structure.create(
            calculator.structure,
            self.get_charge(calculator),
            self.get_multiplicity(calculator),
            self._calculation.get_model(),
            label,
        )
        return new_structure

    def get_calc(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]]) -> utils.core.Calculator:
        """
        Get a calculator from the given map and ensures the system is present

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        name : str
            The name of the system to get
        systems : Dict[str, Optional[utils.core.Calculator]]
            The map of systems

        Returns
        -------
        utils.core.Calculator
            The calculator
        """
        calc = systems.get(name)
        if calc is None:
            self.raise_named_exception(f"Could not find system {name}", CalculatorNotPresentException)
            raise RuntimeError("Unreachable")
        return calc

    @staticmethod
    def get_charge(calculator: utils.core.Calculator) -> int:
        """
        Get the molecular charge of a calculator's settings.

        Parameters
        ----------
        calculator : utils.core.Calculator
            The calculator

        Returns
        -------
        int
            The molecular charge
        """
        charge = calculator.settings[utils.settings_names.molecular_charge]
        assert isinstance(charge, int)
        return charge

    @staticmethod
    def get_multiplicity(calculator: utils.core.Calculator) -> int:
        """
        Get the spin multiplicity of a calculator's settings. Return 0 if the setting is not present.

        Parameters
        ----------
        calculator : utils.core.Calculator
            The calculator

        Returns
        -------
        int
            The molecular charge
        """
        multiplicity = calculator.settings.get(utils.settings_names.spin_multiplicity, 0)
        assert isinstance(multiplicity, int)
        return multiplicity

    def get_energy(self, calculator: utils.core.Calculator) -> float:
        """
        Get the energy of a calculator's results.

        Parameters
        ----------
        calculator : utils.core.Calculator
            The calculator

        Returns
        -------
        float
            The energy
        """
        if not calculator.has_results():
            self.raise_named_exception("Calculator has no results")
            raise RuntimeError("Unreachable")  # just for linters
        energy = calculator.get_results().energy
        if energy is None:
            self.raise_named_exception("Calculator has no energy")
            raise RuntimeError("Unreachable")
        return energy
