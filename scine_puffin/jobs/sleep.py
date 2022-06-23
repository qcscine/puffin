# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from time import sleep
from scine_puffin.config import Configuration
from .templates.job import Job, job_configuration_wrapper


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

      time :: int
         The time to sleep for in seconds. Default: 300.

    **Required Packages**
      - SCINE: Database (present by default)

    **Generated Data**
      This dummy job does not generate new data.
    """

    def __init__(self):
        super().__init__()
        self.name = "Sleep Job"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_database as db

        # Get the requested sleep time
        sleeptime = int(calculation.get_settings().get("time", 300))

        sleep(sleeptime)

        calculation.set_executor(config["daemon"]["uuid"])
        calculation.set_status(db.Status.COMPLETE)
        return True

    @staticmethod
    def required_programs():
        return ["database"]
