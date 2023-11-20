# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List

from .program import Program
from scine_puffin.config import Configuration


class Parrot(Program):
    """
    Parrot -- Machine Learning Potentials for SCINE
    """

    def install(self, repo_dir: str, install_dir: str, _: int):
        if self.root:
            raise RuntimeError
        elif self.source:
            self.pip_module_source_install(repo_dir, install_dir)
        else:
            self.pip_package_install('scine_parrot', install_dir)

    def check_install(self):
        raise NotImplementedError

    def setup_environment(self, _: Configuration, __: dict, ___: dict):
        pass

    def available_models(self) -> List[str]:
        return ['lmlp', 'ani', 'm3gnet']

    @staticmethod
    def initialize():
        import scine_parrot  # noqa: F401 , pylint: disable=unused-import
