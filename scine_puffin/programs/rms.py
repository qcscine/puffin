# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
from typing import List, Optional

from .program import Program
from scine_puffin.config import Configuration


class Rms(Program):
    """
    The reaction mechanism simulator. See https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl

    See the install scripts in the "scripts" directory for details. Note that the installation requires Conda.
    """

    def install(self, repo_dir: str, install_dir: str, ncores: int):
        if self.root:
            pass
        elif self.source:
            raise NotImplementedError("RMS must be installed manually. See rms.py for details.")
        else:
            raise NotImplementedError("RMS must be installed manually. See rms.py for details.")

    def check_install(self):
        self.assert_install()

    @staticmethod
    def assert_install():
        if not Rms.is_installed():
            raise ModuleNotFoundError('RMS was not installed correctly. It must be preinstalled in a conda'
                                      ' environment. An installation script is provided in scripts/rms/build_rms.sh.'
                                      ' More information on the RMS installation process is provided on'
                                      ' http://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/'
                                      'anacondaDeveloper.html')

    @staticmethod
    def is_installed():
        try:
            # pylint: disable=unused-import
            import julia  # noqa: F401
            from julia import ReactionMechanismSimulator  # noqa: F401
            import diffeqpy  # noqa: F401
            # pylint: enable=unused-import
        except ImportError as e:
            print("Julia, pyrms or diffeqpy could not be imported. The error message was:\n" + str(e))
            return False
        return True

    def setup_environment(self, config: Configuration, env_paths: dict, env_vars: dict):
        if self.root:
            raise NotImplementedError
        elif self.source:
            raise NotImplementedError
        else:
            raise RuntimeError

    def available_models(self) -> List[str]:
        return []


class JuliaPrecompiler(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(JuliaPrecompiler, cls).__new__(cls)
            cls.instance.julia_is_precompiled = False
            cls.instance.root: Optional[str] = None
        return cls.instance

    def set_root(self, root: str):
        # pylint: disable=attribute-defined-outside-init
        self.root = root
        # pylint: enable=attribute-defined-outside-init

    def compile_julia(self):
        # Try to load the system image if the file already exists.
        if self.root:
            if ".so" not in self.root or not os.path.exists(self.root):
                raise RuntimeError("The shared library file for RMS was not found. Install RMS through the installation"
                                   "scripts in the scripts directory and activate the conda environment after"
                                   " installation.")
            # pylint: disable=import-error
            from julia import Julia  # noqa: F401
            _ = Julia(sysimage=self.root)
            # pylint: enable=import-error
        else:
            print("Compiling Julia on the fly. This may take a while!")
            # If the system image is not available we resort to compiling it on the fly. This is potentially very slow.
            # pylint: disable=import-error
            from julia.api import Julia
            _ = Julia(compiled_modules=False)
            # pylint: enable=import-error

        # pylint: disable=attribute-defined-outside-init
        self.julia_is_precompiled = True
        # pylint: enable=attribute-defined-outside-init

    def ensure_is_compiled(self):
        if not self.julia_is_precompiled:
            self.compile_julia()
