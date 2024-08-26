# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from time import sleep
from typing import TYPE_CHECKING, List

from scine_puffin.config import Configuration
from .templates.job import Job, job_configuration_wrapper
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class Sleep(Job):
    """
    A dummy job used for debug purposes. The job sleeps for a given amount of
    time.

    **Order Name**
      ``sleep``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      time : int
         The time to sleep for in seconds. Default: 300.

    **Required Packages**
      - SCINE: Database (present by default)

    **Generated Data**
      This dummy job does not generate new data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Sleep Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        # Get the requested sleep time
        sleeptime = int(calculation.get_settings().get("time", 300))  # type: ignore

        sleep(sleeptime)

        calculation.set_executor(config["daemon"]["uuid"])
        calculation.set_status(db.Status.COMPLETE)
        return True

    @staticmethod
    def required_programs() -> List[str]:
        return ["database"]
