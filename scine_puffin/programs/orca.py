# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
from typing import List

from .program import Program
from scine_puffin.config import Configuration


class Orca(Program):
    """
    Setup of the Orca program
    """

    def install(self, repo_dir: str, install_dir: str, ncores: int):
        if self.root:
            pass
        elif self.source:
            raise NotImplementedError
        else:
            raise RuntimeError

    def check_install(self):
        raise NotImplementedError

    def setup_environment(self, config: Configuration, env_paths: dict, env_vars: dict):
        if self.root:
            env_vars["ORCA_BINARY_PATH"] = os.path.join(self.root, "bin", "orca")
        elif self.source:
            pass
        else:
            raise RuntimeError

    def available_models(self) -> List[str]:
        return ["DFT", "HF", "PM3"]
