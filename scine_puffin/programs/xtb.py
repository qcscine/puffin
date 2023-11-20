# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List
import os

from .program import Program
from scine_puffin.config import Configuration


class Xtb(Program):
    """
    Scine: Xtb wrapper -- installation and verification class
    """

    def install(self, repo_dir: str, install_dir: str, ncores: int):
        if self.root:
            raise NotImplementedError
        elif self.source:
            self.scine_module_install(repo_dir, install_dir, ncores, add_lib=True, add_bin=True)
        else:
            raise RuntimeError

    def check_install(self):
        raise NotImplementedError

    def setup_environment(self, config: Configuration, env_paths: dict, env_vars: dict):
        if self.root:
            raise NotImplementedError
        elif self.source:
            env_vars["XTBPATH"] = os.path.join(config.daemon()["software_dir"], "install", "share", "xtb")
        else:
            raise RuntimeError

    def available_models(self) -> List[str]:
        return ["GFN0", "GFN1", "GFN2", "GFNFF"]
