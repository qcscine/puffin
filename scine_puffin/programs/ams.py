# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
from typing import List

from .program import Program
from scine_puffin.config import Configuration


class Ams(Program):
    """
    Setup of the AMS program via the Scine AMS_wrapper
    """

    def install(self, repo_dir: str, install_dir: str, ncores: int):
        if self.root:
            pass
        if self.source:
            self.scine_module_install(repo_dir, install_dir, ncores)
        else:
            raise RuntimeError

    def check_install(self):
        raise NotImplementedError

    def setup_environment(self, config: Configuration, env_paths: dict, env_vars: dict):
        if self.root:
            env_vars["SCMLICENSE"] = os.getenv("SCMLICENSE")
            if all(os.getenv(var) is None for var in ["AMSHOME", "AMSBIN", "AMS_BINARY_PATH"]):
                if os.path.exists(os.path.join(self.root, "bin", "ams")):
                    env_vars["AMSBIN"] = os.path.join(self.root, "bin")
                    env_vars["AMSHOME"] = os.path.join(self.root)
                    env_vars["AMS_BINARY_PATH"] = os.path.join(self.root, "bin")
        elif self.source:
            pass
        else:
            raise RuntimeError

    def available_models(self) -> List[str]:
        return ["DFT", "DFTB3", "DFTB2", "DFTB0", "GFN1", "GFN0", "REAXFF", "MLPOTENTIAL"]
