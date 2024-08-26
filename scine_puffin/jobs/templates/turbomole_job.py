# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import subprocess
from typing import List, TYPE_CHECKING

from scine_puffin.config import Configuration
from scine_puffin.jobs.templates.job import Job
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class TurbomoleJob(Job):
    """
    A common interface for all jobs in Puffin that use Turbomole.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_structure = "system.xyz"

        env = os.environ.copy()

        self.turboexe = ""
        self.turboscripts = ""
        self.smp_turboexe = ""

        if "TURBODIR" in env.keys():
            if env["TURBODIR"]:
                if os.environ.get("PARA_ARCH") is not None:
                    del os.environ["PARA_ARCH"]
                if os.path.exists(os.path.join(env["TURBODIR"], "scripts", "sysname")):
                    self.sysname = (
                        subprocess.check_output(os.path.join(env["TURBODIR"], "scripts", "sysname"))
                        .decode("utf-8", errors='replace')
                        .rstrip()
                    )
                    self.sysname_parallel = self.sysname + "_smp"
                    self.turboexe = os.path.join(env["TURBODIR"], "bin", self.sysname)
                    self.smp_turboexe = os.path.join(env["TURBODIR"], "bin", self.sysname_parallel)
                    self.turboscripts = os.path.join(env["TURBODIR"], "scripts")
                else:
                    raise RuntimeError("TURBODIR not assigned correctly. Check spelling or empty the env variable.")

    @requires("utilities")
    def prepare_calculation(self, structure: db.Structure, calculation_settings: utils.ValueCollection,
                            model: db.Model, job: db.Job) -> None:
        from scine_puffin.utilities.turbomole_helper import TurbomoleHelper

        tm_helper = TurbomoleHelper()
        # Write xyz file
        utils.io.write(self.input_structure, structure.get_atoms())
        # Write coord file
        tm_helper.write_coord_file(calculation_settings)
        # Check if settings are available
        tm_helper.check_settings_availability(job, calculation_settings)
        # Generate input file for preprocessing tool 'define'
        tm_helper.prepare_define_session(structure, model, calculation_settings, job)
        # Initialize via define
        tm_helper.initialize(model, calculation_settings)

    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs() -> List[str]:
        """See Job.required_programs()"""
        raise NotImplementedError
