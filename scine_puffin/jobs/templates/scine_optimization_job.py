# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from typing import List, Union, Dict, TYPE_CHECKING, Optional

from .scine_job import ScineJob
from .job import is_configured
from scine_puffin.utilities.program_helper import ProgramHelper
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class OptimizationJob(ScineJob, ABC):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface to optimize
    a structure in the Scine database.
    """

    own_expected_results = ["energy"]

    def __init__(self) -> None:
        super().__init__()
        self.name = "OptimizationJob"

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "readuct", "utils"]

    @requires("database")
    def determine_new_label(
        self,
        old_label: db.Label,
        graph: str,
        is_surface: bool,
        ignore_user_label: bool = False
    ) -> db.Label:
        """
        Derive the label of an optimized structure based on the previous label and the Molassembler graph.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        old_label : db.Label
            The label which the structure had previous to optimization
        graph : str
            The graph of the structure
        is_surface : bool
            Whether the structure is a surface
        ignore_user_label : bool
            Whether the user label of the given structure shall be ignored.
            If True, an input structure 'user_guess' will get an optimized structure with 'minimum_optimized'

        Returns
        -------
        new_label : db.Label
            The label of the optimized structure
        """
        graph_is_split = ";" in graph
        if old_label == db.Label.MINIMUM_GUESS or old_label == db.Label.MINIMUM_OPTIMIZED:
            if graph_is_split:
                new_label = db.Label.COMPLEX_OPTIMIZED
            else:
                new_label = db.Label.MINIMUM_OPTIMIZED
        elif old_label == db.Label.SURFACE_GUESS or old_label == db.Label.SURFACE_OPTIMIZED:
            if graph_is_split:
                new_label = db.Label.SURFACE_COMPLEX_OPTIMIZED
            else:
                new_label = db.Label.SURFACE_OPTIMIZED
        elif old_label == db.Label.COMPLEX_GUESS or old_label == db.Label.COMPLEX_OPTIMIZED:
            if graph_is_split:
                new_label = db.Label.COMPLEX_OPTIMIZED
            else:
                new_label = db.Label.MINIMUM_OPTIMIZED
        elif old_label == db.Label.USER_OPTIMIZED:
            if graph_is_split:
                new_label = db.Label.USER_COMPLEX_OPTIMIZED
            else:
                new_label = db.Label.USER_OPTIMIZED
        elif old_label == db.Label.USER_GUESS:
            if graph_is_split:
                if is_surface:
                    new_label = db.Label.USER_SURFACE_COMPLEX_OPTIMIZED
                else:
                    new_label = db.Label.USER_COMPLEX_OPTIMIZED
            else:
                if is_surface:
                    new_label = db.Label.USER_SURFACE_OPTIMIZED
                else:
                    new_label = db.Label.USER_OPTIMIZED
        else:
            error = f"Unknown label '{str(old_label)}'\n"
            self.raise_named_exception(error)
            raise RuntimeError("Unreachable")  # for type checking
        if ignore_user_label and new_label == db.Label.USER_OPTIMIZED:
            if graph_is_split:
                new_label = db.Label.COMPLEX_OPTIMIZED
            else:
                new_label = db.Label.MINIMUM_OPTIMIZED
        return new_label

    @is_configured
    def optimization_postprocessing(
        self,
        success: bool,
        systems: Dict[str, Optional[utils.core.Calculator]],
        keys: List[str],
        old_structure: db.Structure,
        new_label: db.Label,
        program_helper: Union[ProgramHelper, None],
        expected_results: Union[List[str], None] = None,
    ) -> db.Structure:
        """
        Checks after an optimization whether everything went well and saves information to database.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        success : bool
            The boolean signalling whether the task was successful (forwarded from ReaDuct)
        systems : dict
            Dictionary holding the calculators (forwarded from ReaDuct)
        keys : List[str]
            The keys specifying the checked calculator
        old_structure : db.Structure
            The structure which was optimized
        new_label : db.Label
            The label of the new structure
        program_helper : Union[ProgramHelper, None]
            The optional helper of the employed program for postprocessing
        program_helper : Union[List[str], None]
            The expected results for the calculators, if not given, assumed from invoking Job class
        expected_results : Union[List[str], None]
            The expected results for the calculators, if not given, assumed from invoking Job class

        Returns
        -------
        db.Structure
            The new structure
        """

        # postprocessing of results with sanity checks
        db_results = self.calculation_postprocessing(success, systems, keys, expected_results)

        # New structure
        calc = self.get_calc(keys[0], systems)
        new_structure = self.create_new_structure(calc, new_label)
        db_results.add_structure(new_structure.id())
        self._calculation.set_results(db_results)

        # handle properties
        self.store_energy(calc, new_structure)
        self.transfer_properties(old_structure, new_structure)

        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, old_structure, new_structure)

        return new_structure
