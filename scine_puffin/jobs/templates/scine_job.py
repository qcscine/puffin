# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Tuple, Union

from .job import Job, job_configuration_wrapper
from scine_puffin.config import Configuration
from scine_puffin.utilities.scine_helper import SettingsManager, update_model
from scine_puffin.utilities.program_helper import ProgramHelper


class ScineJob(Job):
    """
    A common interface for all jobs in Puffin that use the
    Scine::Core::Calculator interface.
    This interface holds basic logic for interacting with Scine classes, which should be used in every ScineJob
    This interface also holds subclasses that include elementary workflows, which can be combined to build complex jobs
    """

    def __init__(self):
        super().__init__()
        self.name = "ScineJob"  # to be overwritten by child class
        self.own_expected_results = []  # to be overwritten by child class
        # to be added by child class:
        self.properties_to_transfer = [
            "surface_atom_indices",
            "slab_dict",
            "slab_formula",
            "primitive_lattice",
        ]
        self._fallback_error = "Error: " + self.name + " failed with an unspecified error."

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    def expected_results(self) -> List[str]:
        """
        Gives a list of str specifying the results expected to be present for a system within a job based on the
        class member and all its parents.

        Returns
        -------
        expected_results :: List[str]
            The results to be expected as str corresponding to the members of the Scine::Utils::Results class.
        """
        parent_expects = []
        for Parent in self.__class__.__bases__:
            if "expected_results" in dir(Parent):
                parent = Parent()
                parent_expects += parent.expected_results()  # pylint: disable=no-member
        return list(set(parent_expects + self.own_expected_results))

    @staticmethod
    def required_programs() -> List[str]:
        """See Job.required_programs()"""
        raise NotImplementedError

    def create_helpers(self, structure) -> Tuple[SettingsManager, Union[ProgramHelper, None]]:
        """
        Creates a Scine SettingsManager and ProgramHelper based on the configured job and the given structure.
        The ProgramHelper is None if no ProgramHelper is specified for the specified program or no program was
        specified in the model

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        structure :: db.Structure (Scine::Database::Structure)
            The structure on which a Calculation is performed.

        Returns
        -------
        helper_tuple :: Tuple[SettingsManager, Union[ProgramHelper, None]
            A tuple of the SettingsManager for Scine Calculators and ProgramHelper if available.
        """
        model = self._calculation.get_model()
        program = model.program if model.program != "any" else ""
        settings_manager = SettingsManager(model.method_family, program)
        program_helper = ProgramHelper.get_correct_helper(program, self._manager, structure, self._calculation)
        return settings_manager, program_helper

    def raise_named_exception(self, error_message: str) -> None:
        """
        Raises an error including the name/description of the current job.
        """
        error_begin = "Error: " + self.name + " failed with message:\n"
        raise BaseException(error_begin + error_message)

    def throw_if_not_successful(
        self,
        success: bool,
        systems: dict,
        keys: List[str],
        expected_results: Union[List[str], None] = None,
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
        success :: bool
            Whether the calculation was successful (either forwarded from readuct task or specified if not relevant).
        systems :: Dict[str, core.calculator] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys :: List[str]
            The list of keys of the systems dict to be checked.
        expected_results :: Union[List[str], None]
            The results to be required to be present in systems to qualify as successful calculations. If None is
            given, this defaults to the expected results of the class, see expected_results().
        sub_task_error_line :: str
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

    def calculation_postprocessing(
        self,
        success: bool,
        systems: dict,
        keys: List[str],
        expected_results: Union[List[str], None] = None,
    ):
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
        success :: bool
            Whether the calculation was successful (either forwarded from readuct task or specified if not relevant).
        systems :: Dict[str, core.calculator] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys :: List[str]
            The list of keys of the systems dict to be checked.
        expected_results :: Union[List[str], None]
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
            systems[keys[0]], self._calculation, self.config
        )  # calculation is safe -> update model
        db_results = self._calculation.get_results()
        db_results.clear()
        self._calculation.set_results(db_results)
        return db_results

    def prepend_to_comment(self, message: str):
        """
        Prepends given message to the comment field of the currently configured calculation.

        Notes
        -----
        * Requires run configuration
        Parameters
        ----------
        message :: str
            The message to be prepended.
        """
        comment = self._calculation.get_comment()
        self._calculation.set_comment(message + comment)

    def expected_results_check(
        self,
        systems: dict,
        keys: List[str],
        expected_results: Union[List[str], None] = None,
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
        systems :: Dict[str, core.calculator] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys :: List[str]
            The list of keys of the systems dict to be checked.
        expected_results :: Union[List[str], None]
            The results to be required to be present in systems to qualify as successful calculations. If None is
            given, this defaults to the expected results of the class, see expected_results().

        Returns
        -------
        error_message :: str
            A string containing the error message, describing failure in expected results.
        """
        if expected_results is None:
            expected_results = self.expected_results()
        # check all specified systems
        for key in keys:
            if key not in systems:
                return False, (key + " is missing in systems!")
            # check if desired results are present
            if not systems[key].has_results():
                return False, ("System '" + key + "' is missing results!")
            results = systems[key].get_results()
            for expected in expected_results:
                if getattr(results, expected) is None:
                    return False, (expected + " is missing in results!")
        return True, ""

    def store_energy(self, system, structure):
        """
        Stores an 'electronic_energy' property for the given structure based on the energy in the results of the given
        system. Does not perform checks.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        system :: core.calculator (Scine::Core::Calculator)
            A Scine calculator holding a results object with the energy property.
        structure :: db.Structure (Scine::Database::Structure)
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

    def transfer_properties(self, old_structure, new_structure):
        """
        Copies property IDs from one structure to another one based on the specified properties in the class member.

        Parameters
        ----------
        old_structure :: db.Structure (Scine::Database::Structure)
            The structure holding the properties. If a specified property is not present for the structure,
            no error is given.
        new_structure :: db.Structure (Scine::Database::Structure)
            The structure for which the property is to be added.
        """
        properties_to_transfer = list(set(self.properties_to_transfer))  # make sure no duplicates
        for prop in properties_to_transfer:
            if old_structure.has_property(prop):
                prop_id = old_structure.get_property(prop)
                new_structure.set_property(prop, prop_id)

    def sp_postprocessing(
        self,
        success: bool,
        systems: dict,
        keys: List[str],
        structure,
        program_helper: Union[ProgramHelper, None],
    ):
        """
        Performs a verification and results saving protocol for a Scine Single Point Calculation.

        Notes
        -----
        * Requires run configuration
        * May throw Exception

        Parameters
        ----------
        success :: bool
            Whether the calculation was successful (either forwarded from
            readuct task or specified if not relevant).
        systems :: Dict[str, core.calculator] (Scine::Core::Calculator)
            The dictionary holding calculators.
        keys :: List[str]
            The list of keys of the systems dict to be checked.
        structure :: db.Structure (Scine::Database::Structure)
            The structure on which the calculation was performed.
        program_helper :: ProgramHelper
            The possible ProgramHelper that may also want to do some
            postprocessing after a calculation.
        """

        # postprocessing of results with sanity checks
        _ = self.calculation_postprocessing(success, systems, keys)

        # handle properties
        self.store_energy(systems[keys[0]], structure)
        results = systems[keys[0]].get_results()
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
