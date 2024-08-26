# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Optional

from .job import is_configured
from .scine_observers import StoreEverythingObserver, StoreWithFrequencyObserver, StoreWithFractionObserver
from .sub_settings_job import SubSettingsJob
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency
from scine_puffin.utilities.task_to_readuct_call import SubTaskToReaductCall

if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")
if module_exists("scine_readuct") or TYPE_CHECKING:
    import scine_readuct as readuct
else:
    readuct = MissingDependency("scine_readuct")


class ScineJobWithObservers(SubSettingsJob, ABC):
    """
    A common interface for all jobs in Puffin that want to have an observer for individual calculations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "ScineJobWithObservers"  # to be overwritten by child
        # to be extended by child:
        self.settings: Dict[str, Dict[str, Any]] = {
            **self.settings,
            self.job_key: {
                **self.settings[self.job_key],
                "store_all_structures": False,
                "store_structures_with_frequency": {
                    task: 0 for task in SubTaskToReaductCall.__members__
                },
                "store_structures_with_fraction": {
                    task: 0.0 for task in SubTaskToReaductCall.__members__
                },
            }
        }

    @classmethod
    def optional_settings_doc(cls) -> str:
        return super().optional_settings_doc() + """

        These settings are recognized related to storing individual structures encountered during the job:

        store_all_structures : bool
            If all structures encountered during the elementary step search should be stored in the database.
        store_structures_with_frequency : Dict[str, int]
            Determine for each subtask, such as 'opt' or 'tsopt', a frequency, such as every third, to store the
            structures encountered in the subtask in the database.
        store_structures_with_fraction : Dict[str, float]
            Determine for each subtask, such as 'opt' or 'tsopt', a fraction of structures, such as a third,
            to store the structures encountered in the subtask in the database. The saved structure will not be spaced
            evenly but the fraction determines the random chance for each structure to be stored.
        """

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "molassembler", "readuct", "utils"]

    @is_configured
    @requires("readuct")
    def observed_readuct_call(self, task: SubTaskToReaductCall, systems: Dict[str, Optional[utils.core.Calculator]],
                              input_names: List[str], **kwargs) \
            -> Tuple[Dict[str, Optional[utils.core.Calculator]], bool]:
        from scine_utilities.settings_names import molecular_charge, spin_multiplicity
        for name in input_names:
            if systems.get(name) is None:
                self.raise_named_exception(f"System {name} is not in systems: {systems}")
                raise RuntimeError("Unreachable")  # just for linters
        observers = []
        observer_functions = []
        model = self._calculation.get_model()
        model.complete_model(self.get_calc(input_names[0], systems).settings)
        if self.settings[self.job_key]["store_all_structures"]:
            observers.append(StoreEverythingObserver(self._calculation.get_id(), model))
            observer_functions = [observers[-1].gather]
        elif (self.settings[self.job_key]["store_structures_with_frequency"][task.name] and
              self.settings[self.job_key]["store_structures_with_fraction"][task.name]):
            raise NotImplementedError("Non-zero values in a single task for both store_structures_with_frequency ",
                                      "and store_structures_with_fraction are not allowed.")
        elif self.settings[self.job_key]["store_structures_with_frequency"].get(task.name):
            observers.append(StoreWithFrequencyObserver(
                self._calculation.get_id(), model,
                self.settings[self.job_key]["store_structures_with_frequency"][task.name])
            )
            observer_functions = [observers[-1].gather]
        elif self.settings[self.job_key]["store_structures_with_fraction"].get(task.name):
            observers.append(StoreWithFractionObserver(
                self._calculation.get_id(), model,
                self.settings[self.job_key]["store_structures_with_fraction"][task.name])
            )
            observer_functions = [observers[-1].gather]
        # carry out ReaDuct task
        result = getattr(readuct, task.value)(systems, input_names, observers=observer_functions, **kwargs)
        # TODO this may need to be redone for multi input calls
        # i.e. tasks with more than one input structure.
        # But it would only be a problem if they have a different charge or multiplicity
        # currently no such task exists in ReaDuct
        calc = self.get_calc(input_names[0], systems)
        charge = calc.settings[molecular_charge]
        multiplicity = calc.settings[spin_multiplicity]
        for observer in observers:
            observer.finalize(self._manager, charge, multiplicity)
        return result

    @is_configured
    def observed_readuct_call_with_throw(self, subtask: SubTaskToReaductCall,
                                         systems: Dict[str, Optional[utils.core.Calculator]], input_names: List[str],
                                         expected_results: List[str], error_msg: str, **kwargs) \
            -> Dict[str, Optional[utils.core.Calculator]]:
        systems, success = self.observed_readuct_call(subtask, systems, input_names, **kwargs)
        names_to_check = [kwargs['output'] if 'output' in kwargs else input_names]
        self.throw_if_not_successful(success, systems, names_to_check, expected_results, error_msg)
        return systems
