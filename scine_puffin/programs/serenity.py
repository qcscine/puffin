# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
from typing import List

from .program import Program
from scine_puffin.config import Configuration


class Serenity(Program):
    """
    Serenity -- installation and verification class
    """

    def install(self, repo_dir: str, install_dir: str, ncores: int):
        if self.root:
            pass
        elif self.source:
            self.scine_module_install(repo_dir, install_dir, ncores, add_lib=True, add_bin=True)
        else:
            raise RuntimeError

    def check_install(self):
        raise NotImplementedError

    def setup_environment(self, config: Configuration, env_paths: dict, env_vars: dict):
        if self.root:
            env_paths["PATH"] = env_paths["PATH"] + ":" + os.path.join(self.root, "bin")
            env_paths["LD_LIBRARY_PATH"] = env_paths["LD_LIBRARY_PATH"] + ":" + os.path.join(self.root, "lib")
            env_vars["SERENITY_RESOURCES"] = os.path.join(self.root, "share", "serenity", "data", "")
            env_vars["SERENITY_MEMORY"] = str(float(config.resources()["memory"]) * 1024)
            env_vars["OMP_NUM_THREADS"] = str(config.resources()["cores"])
        elif self.source:
            env_vars["SERENITY_RESOURCES"] = os.path.join(
                config.daemon()["software_dir"],
                "install",
                "share",
                "serenity",
                "data",
                "",
            )
            env_vars["SERENITY_MEMORY"] = str(float(config.resources()["memory"]) * 1024)
            env_vars["OMP_NUM_THREADS"] = str(config.resources()["cores"])
        else:
            raise RuntimeError

    def available_models(self) -> List[str]:
        return ["DFT", "HF", "CC"]
