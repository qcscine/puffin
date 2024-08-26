# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from typing import Any, Dict

from .job import is_configured
from .scine_connectivity_job import ConnectivityJob


class SubSettingsJob(ConnectivityJob, ABC):
    """
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "SubSettingsJob"  # to be overwritten by child
        self.job_key = "job"
        # to be extended by child:
        self.settings: Dict[str, Dict[str, Any]] = {
            self.job_key: {}
        }

    @is_configured
    def sort_settings(self, task_settings: Dict[str, Any]) -> None:
        """
        Take settings of configured calculation and save them in class member. Throw exception for unknown settings.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        task_settings : dict
            A dictionary from which the settings are taken
        """
        self.extract_connectivity_settings_from_dict(task_settings)
        # Dissect settings into individual user task_settings
        for key, value in task_settings.items():
            for task in self.settings.keys():
                if task == self.job_key:
                    if key in self.settings[task].keys():
                        self.settings[task][key] = value
                        break  # found right task, leave inner loop
                else:
                    indicator_length = len(task) + 1  # underscore to avoid ambiguities
                    if key[:indicator_length] == task + "_":
                        self.settings[task][key[indicator_length:]] = value
                        break  # found right task, leave inner loop
            else:
                self.raise_named_exception(
                    f"The key '{key}' was not recognized."
                )

        if "ircopt" in self.settings.keys() and "output" in self.settings["ircopt"]:
            self.raise_named_exception(
                "Cannot specify a separate output system for the optimization of the IRC end points"
            )
