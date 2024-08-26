# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from pkgutil import iter_modules
from typing import List
from warnings import warn
import git
import os
import subprocess
import sys

from scine_puffin.config import Configuration


class Program:
    """
    A common interface for all programs and their setups

    Parameters
    ----------
    settings : dict
        The settings for the particular program. This dictionary should be the
        given program's block in the ``Configuration``.
    """

    def __init__(self, settings: dict) -> None:
        self.version = settings["version"]
        self.root = settings["root"]
        self.source = settings["source"]
        self.settings = settings

    def install(self, repo_dir: str, install_dir: str, ncores: int):
        """
        Installs or loads the given program. After the install, the
        ``check_install`` function should run through with out exceptions.
        The choice of installation/compilation or loading of the program is
        based on the settings given in the constructor.

        Parameters
        ----------
        repo_dir : str
            The folder for all repositories, if a clone or download is required
            for the installation, this folder will be used.
        install_dir : str
            If the program is actually installed and not just loaded, this
            folder will be used as target directory for the install process.
        ncores : int
            The number of cores/threads to be used when compiling/installing
            the program.
        """
        raise NotImplementedError

    @staticmethod
    def initialize():
        """
        Executed at Puffin start, run once for each available program
        """

    def check_install(self):
        """
        A small function checking if the program was installed/located correctly
        and does provide the expected features.
        """
        raise NotImplementedError

    def setup_environment(self, config: Configuration, env_paths: dict, env_vars: dict):
        """
        Appends the program specific environment variables to the given
        dictionaries.

        Parameters
        ----------
        config : scine_puffin.config.Configuration
            The current global configuration.
        env_paths : dict
            A dictionary for all the environment paths, such as ``PATH`` and
            ``LD_LIBRARY_PATH``. The added settings will be appended to the
            existing paths, using ``export PATH=$PATH:...``.
        env_vars : dict
            A dictionary for all fixed environment variables. All settings
            will replace existing variables such as ``export OMP_NUM_THREADS=1``
        """
        raise NotImplementedError

    def available_models(self) -> List[str]:
        """
        A small function returning the single point models available now with
        the given program loaded/installed.

        Returns
        -------
        models : List[str]
            A list of names of models that are available if the program is
            available.
        """
        raise NotImplementedError

    def scine_module_install(self, repo_dir: str, install_dir: str, ncores: int,
                             add_lib: bool = False, add_bin: bool = False):
        initial_dir = os.getcwd()

        # Handle repository
        if os.path.exists(repo_dir):
            repository = git.Repo(repo_dir)
            try:
                repository.remotes.origin.pull()
                repository.git.submodule("update", "--init")
            except BaseException:
                try:
                    repository.git.checkout("master")
                except git.exc.GitCommandError:  # type: ignore[misc]
                    repository.git.checkout("main")
                repository.git.submodule("update", "--init")
                repository.remotes.origin.pull()
                repository.git.submodule("update", "--init")
            finally:
                repository.git.checkout(self.version)
                repository.remotes.origin.pull()
                repository.git.submodule("update", "--init")
        else:
            repository = git.Repo.clone_from(self.source, repo_dir)
            repository.git.checkout(self.version)
            repository.git.submodule("update", "--init")

        # Build from sources into <repo>/build and install
        build_dir = os.path.join(repo_dir, "build")
        if build_dir and not os.path.exists(build_dir):
            os.makedirs(build_dir)
        os.chdir(build_dir)
        env = os.environ.copy()
        if add_bin:
            env["PATH"] = env["PATH"] + ":" + os.path.join(install_dir, "bin")
        if add_lib:
            if "LD_LIBRARY_PATH" in env.keys():
                env["LD_LIBRARY_PATH"] = (
                    env["LD_LIBRARY_PATH"] + ":" + os.path.join(install_dir, "lib")
                )
            else:
                env["LD_LIBRARY_PATH"] = os.path.join(install_dir, "lib")
            env["LD_LIBRARY_PATH"] = (
                env["LD_LIBRARY_PATH"] + ":" + os.path.join(install_dir, "lib64")
            )
        args = ["cmake"]
        args.append("-DCMAKE_BUILD_TYPE=Release")
        args.append("-DSCINE_BUILD_TESTS=OFF")
        args.append("-DSCINE_BUILD_PYTHON_BINDINGS=ON")
        args.append("-DSCINE_MARCH=" + self.settings["march"])
        if self.settings["cxx_compiler_flags"]:
            args.append("-DCMAKE_CXX_FLAGS=" + self.settings["cxx_compiler_flags"])
        args.append("-DCMAKE_INSTALL_PREFIX=" + install_dir)
        args.append("-DPYTHON_EXECUTABLE=" + sys.executable)
        if self.settings["cmake_flags"]:
            args += self.settings["cmake_flags"].split(" ")
        if "sphinx" not in (name for loader, name, ispkg in iter_modules()):
            warn("Sphinx is not installed, skipping Scine documentation build")
            args.append("-DSCINE_BUILD_DOCS=OFF")
        args.append("..")
        subprocess.run(args, env=env, check=True)
        subprocess.run(["make", "-j" + str(ncores), "install"], env=env, check=True)
        os.chdir(initial_dir)

    def pip_module_source_install(self, repo_dir: str, install_dir: str):
        initial_dir = os.getcwd()

        # Handle repository
        if os.path.exists(repo_dir):
            repository = git.Repo(repo_dir)
            try:
                repository.remotes.origin.pull()
                repository.git.submodule("update", "--init")
            except BaseException:
                try:
                    repository.git.checkout("master")
                except git.exc.GitCommandError:  # type: ignore[misc]
                    repository.git.checkout("main")
                repository.git.submodule("update", "--init")
                repository.remotes.origin.pull()
                repository.git.submodule("update", "--init")
            finally:
                repository.git.checkout(self.version)
                repository.remotes.origin.pull()
                repository.git.submodule("update", "--init")
        else:
            repository = git.Repo.clone_from(self.source, repo_dir)
            repository.git.checkout(self.version)
            repository.git.submodule("update", "--init")

        build_dir = os.path.join(repo_dir, "build")
        if build_dir and not os.path.exists(build_dir):
            os.makedirs(build_dir)
        os.chdir(build_dir)
        self.pip_package_install('../.', install_dir)
        os.chdir(initial_dir)

    @staticmethod
    def pip_package_install(package: str, install_dir: str):
        env = os.environ.copy()
        suffix = (
            'python' + str(sys.version_info.major) + '.' +
            str(sys.version_info.minor) + '/site-packages'
        )
        if "PYTHONPATH" in env.keys():
            env["PYTHONPATH"] = (
                os.path.join(install_dir, 'lib', suffix) +
                ":" + os.path.join(install_dir, 'lib64', suffix) +
                ":" + env["PYTHONPATH"]
            )
        else:
            env["PYTHONPATH"] = (
                os.path.join(install_dir, 'lib', suffix) +
                ":" + os.path.join(install_dir, 'lib64', suffix)
            )
        subprocess.run(
            [
                sys.executable, '-m' + 'pip', 'install', package,
                '--no-cache',
                '--prefix', install_dir
            ],
            env=env,
            check=True
        )
