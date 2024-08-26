# -*- coding: utf-8 -*-
from __future__ import annotations
"""program_helper.py: Collection of common procedures to be carried out depending on underlying calculators"""
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, List, Optional, TYPE_CHECKING

from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class ProgramHelper(ABC):
    """
    A common interface for all helper classes for specific Scine calculators
    """

    def __init__(self, *arg, **kwargs) -> None:
        self.helper_settings: List[str] = []

    @staticmethod
    @abstractmethod
    def program() -> str:
        """
        Must be implemented by every subclass.
        Returns the name of the program the helper is written for.
        """
        raise NotImplementedError

    @classmethod
    def get_correct_helper(
        cls,
        program: str,
        manager: db.Manager,
        structure: db.Structure,
        calculation: db.Calculation,
    ) -> Optional[ProgramHelper]:
        """
        Returns the correct ProgramHelper Child class for the given program name or None if no suitable helper exists.

        Parameters
        ----------
        program : str
            The name of the program, not case-sensitive.
        manager : db.Manager (Scine::Database::Manager)
            The manager of the database.
        structure : db.Structure (Scine::Database::Structure)
            The structure to be calculated, necessary for init of child.
        calculation : db.Calculation (Scine::Database::Calculation)
            The calculation that is carried out in the current job.
        """
        for Child in cls.__subclasses__()[1:]:  # parent is first element
            if Child.program().lower() == program.lower():
                return Child(manager, structure, calculation)
        return None

    @abstractmethod
    def calculation_preprocessing(
        self,
        calculator: utils.core.Calculator,
        calculation_settings: utils.ValueCollection,
    ) -> None:
        """
        Makes the necessary preparations before the calculation in a job.
        This function has to be implemented by every subclass.

        Parameters
        ----------
        calculator : utils.core.calculator Scine::Core::Calculator)
            The calculator to work on/with.
        calculation_settings : utils.ValueCollection (Scine::Utils::ValueCollection)
            The settings of the calculation in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def calculation_postprocessing(
        self,
        calculation: db.Calculation,
        old_structure: db.Structure,
        new_structure: Union[db.Structure, None] = None,
    ) -> None:
        """
        Write additional information into the database after the calculation in a job.
        This function has to be implemented by every subclass.
        This function must not throw exceptions.

        Parameters
        ----------
        calculation : db.Calculation (Scine::Database::Calculation)
            The calculation that triggered the execution of the job.
        old_structure : db.Structure (Scine::Database::Structure)
            The structure which was calculated
        new_structure : db.Structure (Scine::Database::Structure)
            An optional resulting structure from the Calculation (e.g. optimized structure)
        """
        raise NotImplementedError


class Cp2kHelper(ProgramHelper):
    def __init__(self, manager: db.Manager, structure: db.Structure, calculation: db.Calculation) -> None:
        super().__init__()
        self.helper_settings = [
            "energy_accuracy",
            "distribution_factor_accuracy",
            "start_cutoff",
            "start_rel_cutoff",
        ]
        self.cutoff_name = "plane_wave_cutoff"
        self.rel_cutoff_name = "relative_multi_grid_cutoff"
        self.structures = manager.get_collection("structures")
        self.compounds = manager.get_collection("compounds")
        self.properties = manager.get_collection("properties")
        self.cutoff_optimization_necessary = False
        self.cutoff_optimization_settings: Dict[str, Any] = {}
        settings = calculation.get_settings()
        self.cutoff_handling(structure, settings)
        self.extract_cutoff_optimization_settings(settings)
        calculation.set_settings(settings)

    @staticmethod
    def program() -> str:
        return "cp2k"

    def calculation_preprocessing(
        self,
        calculator: utils.core.Calculator,
        calculation_settings: utils.ValueCollection,
    ) -> None:
        if self.cutoff_optimization_necessary:
            self.optimize_cutoffs(calculator, calculation_settings)

    def calculation_postprocessing(
        self,
        calculation: db.Calculation,
        old_structure: db.Structure,
        new_structure: Union[db.Structure, None] = None,
    ) -> None:
        if new_structure is None:
            return
        settings = calculation.get_settings()
        cutoff: float = float(settings[self.cutoff_name])  # type: ignore
        rel_cutoff: float = float(settings[self.rel_cutoff_name])  # type: ignore
        if self.structure_has_cutoff_properties(old_structure):
            if not self._compare_and_save_cutoff_properties(old_structure, new_structure, cutoff, rel_cutoff):
                self.make_new_cutoff_properties(calculation, new_structure, cutoff, rel_cutoff)
        else:
            self.make_new_cutoff_properties(calculation, new_structure, cutoff, rel_cutoff)

    def cutoff_handling(self, structure: db.Structure, settings: utils.ValueCollection):
        # error if only one of two cutoffs set
        if (
            self.cutoff_name not in settings
            and self.rel_cutoff_name in settings
            or self.cutoff_name in settings
            and self.rel_cutoff_name not in settings
        ):
            raise RuntimeError(
                "Only specified one of the two cutoff settings in the settings. Either set both or "
                "none to select them from previous structures if available or perform grid evaluation."
            )
        # if none specified, try to find properties
        if self.cutoff_name not in settings and self.rel_cutoff_name not in settings:
            cutoff, rel_cutoff = self.cutoffs_from_properties(structure)
            if cutoff is not None and rel_cutoff is not None:
                settings[self.cutoff_name] = cutoff
                settings[self.rel_cutoff_name] = rel_cutoff
            else:
                self.cutoff_optimization_necessary = True
        # do nothing if cutoffs present in settings

    def cutoffs_from_properties(self, structure: db.Structure) -> Union[Tuple[float, float], Tuple[None, None]]:
        property_id1 = None
        property_id2 = None
        possible_ref = None
        # check structure for existing cutoffs
        if structure.has_property(self.cutoff_name) and structure.has_property(self.rel_cutoff_name):
            property_id1 = structure.get_property(self.cutoff_name)
            property_id2 = structure.get_property(self.rel_cutoff_name)
        # check all structures of compound for existing cutoffs
        elif structure.has_compound():
            compound = db.Compound(structure.get_compound())
            compound.link(self.compounds)
            for struc_id in compound.get_structures():
                struc = db.Structure(struc_id)
                struc.link(self.structures)
                if struc.has_property(self.cutoff_name) and struc.has_property(self.rel_cutoff_name):
                    if possible_ref is not None and possible_ref.olderThan(struc, modification=True):
                        property_id1 = struc.get_property(self.cutoff_name)
                        property_id2 = struc.get_property(self.rel_cutoff_name)
                        possible_ref = struc
                    else:
                        property_id1 = struc.get_property(self.cutoff_name)
                        property_id2 = struc.get_property(self.rel_cutoff_name)
                        possible_ref = struc
            # set properties for the currently calculated structure to avoid lookup next time
            if property_id1 is not None and property_id2 is not None:
                structure.set_property(self.cutoff_name, property_id1)
                structure.set_property(self.rel_cutoff_name, property_id2)

        if property_id1 is not None and property_id2 is not None:
            prop = db.NumberProperty(property_id1, self.properties)
            cutoff = prop.get_data()
            prop = db.NumberProperty(property_id2, self.properties)
            rel_cutoff = prop.get_data()
            return cutoff, rel_cutoff
        return None, None

    def _compare_and_save_cutoff_properties(
        self,
        old_structure: db.Structure,
        new_structure: db.Structure,
        cutoff: float,
        rel_cutoff: float,
        eps: float = 1e-12,
    ) -> bool:
        prop1 = db.NumberProperty(old_structure.get_property(self.cutoff_name))
        prop2 = db.NumberProperty(old_structure.get_property(self.rel_cutoff_name))
        prop1.link(self.properties)
        prop2.link(self.properties)
        if abs(prop1.get_data() - cutoff) < eps and abs(prop2.get_data() - rel_cutoff) < eps:
            new_structure.set_property(self.cutoff_name, prop1.id())
            new_structure.set_property(self.rel_cutoff_name, prop2.id())
            return True
        return False

    def make_new_cutoff_properties(
        self,
        calculation: db.Calculation,
        structure: db.Structure,
        cutoff: float,
        rel_cutoff: float,
    ) -> None:
        p1 = db.NumberProperty.make(
            self.cutoff_name,
            calculation.get_model(),
            cutoff,
            structure.id(),
            calculation.id(),
            self.properties,
        )
        p2 = db.NumberProperty.make(
            self.rel_cutoff_name,
            calculation.get_model(),
            rel_cutoff,
            structure.id(),
            calculation.id(),
            self.properties,
        )
        structure.set_property(self.cutoff_name, p1.id())
        structure.set_property(self.rel_cutoff_name, p2.id())

    def structure_has_cutoff_properties(self, structure: db.Structure) -> bool:
        return structure.has_property(self.cutoff_name) and structure.has_property(self.rel_cutoff_name)

    def extract_cutoff_optimization_settings(self, settings: utils.ValueCollection) -> None:
        if "optimize_cutoffs" in settings.keys():
            self.cutoff_optimization_necessary = settings["optimize_cutoffs"]  # type: ignore
            del settings["optimize_cutoffs"]
        for key in self.helper_settings:
            if key in settings.keys():
                self.cutoff_optimization_settings[key] = settings[key]
                del settings[key]

    def optimize_cutoffs(self, system: utils.core.Calculator, settings: utils.ValueCollection) -> None:
        optimizer = utils.Cp2kCutoffOptimizer(system)
        optimizer.determine_optimal_grid_cutoffs(**self.cutoff_optimization_settings)
        # update calculation settings for later property saving
        settings.update({self.cutoff_name: system.settings[self.cutoff_name]})
        settings.update({self.rel_cutoff_name: system.settings[self.rel_cutoff_name]})
