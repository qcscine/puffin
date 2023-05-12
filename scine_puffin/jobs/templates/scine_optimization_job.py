# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Union

from .job import job_configuration_wrapper
from .scine_job import ScineJob
from scine_puffin.config import Configuration
from scine_puffin.utilities.program_helper import ProgramHelper


class OptimizationJob(ScineJob):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface to optimize
    a structure in the Scine database.
    """

    def __init__(self):
        super().__init__()
        self.name = "OptimizationJob"
        self.own_expected_results = ["energy"]

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs():
        return ["database", "readuct", "utils"]

    def determine_new_label(self, structure, ignore_user_label: bool = False):
        """
        Derive the label of the optimized structure based on the given structure.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        structure :: db.Structure
            The structure to be optimized
        ignore_user_label :: bool
            Whether the user label of the given structure shall be ignored.
            If True, an input structure 'user_guess' will get an optimized structure with 'minimum_optimized'

        Returns
        -------
        new_label :: db.Label
            The label of the optimized structure
        """
        import scine_database as db

        label = structure.get_label()
        if label == db.Label.MINIMUM_GUESS or label == db.Label.MINIMUM_OPTIMIZED:
            new_label = db.Label.MINIMUM_OPTIMIZED
        elif label == db.Label.USER_GUESS or label == db.Label.USER_OPTIMIZED:
            new_label = db.Label.USER_OPTIMIZED
        elif label == db.Label.SURFACE_GUESS or label == db.Label.SURFACE_OPTIMIZED:
            new_label = db.Label.SURFACE_OPTIMIZED
        elif label == db.Label.COMPLEX_GUESS or label == db.Label.COMPLEX_OPTIMIZED:
            new_label = db.Label.COMPLEX_OPTIMIZED
        else:
            error = (
                "Unknown label '"
                + str(label)
                + "' of input structure: '"
                + str(structure.id())
                + "'\n"
            )
            self.raise_named_exception(error)
        if ignore_user_label and new_label == db.Label.USER_OPTIMIZED:
            new_label = db.Label.MINIMUM_OPTIMIZED
        return new_label

    def create_new_structure(self, calculator, label):
        """
        Add a new structure to the database based on the given calculator and label.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        calculator :: utils.core.Calculator
            The calculator holding the optimized structure
        label :: db.Label
            The label of the new structure
        """
        import scine_database as db
        import scine_utilities as utils

        # New structure
        new_structure = db.Structure()
        new_structure.link(self._structures)
        new_structure.create(
            calculator.structure,
            calculator.settings[utils.settings_names.molecular_charge],
            calculator.settings[utils.settings_names.spin_multiplicity],
            self._calculation.get_model(),
            label,
        )
        return new_structure

    def optimization_postprocessing(
        self,
        success: bool,
        systems: dict,
        keys: List[str],
        old_structure,
        new_label,
        program_helper: Union[ProgramHelper, None],
        expected_results: Union[List[str], None] = None,
    ):
        """
        Checks after an optimization whether everything went well and saves information to database.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        success :: bool
            The boolean signalling whether the task was successful (forwarded from ReaDuct)
        systems :: dict
            Dictionary holding the calculators (forwarded from ReaDuct)
        keys :: List[str]
            The keys specifying the checked calculator
        old_structure :: db.Structure
            The structure which was optimized
        new_label :: db.Label
            The label of the new structure
        program_helper :: Union[ProgramHelper, None]
            The optional helper of the employed program for postprocessing
        program_helper :: Union[List[str], None]
            The expected results for the calculators, if not given, assumed from invoking Job class
        """

        # postprocessing of results with sanity checks
        db_results = self.calculation_postprocessing(success, systems, keys, expected_results)

        # New structure
        new_structure = self.create_new_structure(systems[keys[0]], new_label)
        db_results.add_structure(new_structure.id())
        self._calculation.set_results(db_results)

        # handle properties
        self.store_energy(systems[keys[0]], new_structure)
        self.transfer_properties(old_structure, new_structure)

        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, old_structure, new_structure)

        return new_structure
