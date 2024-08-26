# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from functools import lru_cache, wraps
from pkgutil import iter_modules
from types import ModuleType
from typing import Callable, List
import os


class MissingDependencyError(Exception):
    pass


@lru_cache
def module_exists(module_name: str) -> bool:
    """
    Allows checking if a Python module is installed with the Python module given as a string.
    Additionally, also checks for environment variables for specific programs that can be available to the Puffin
    such as different quantum chemistry programs.

    Parameters
    ----------
    module_name : str
        The name of the module to check for.

    Examples
    --------
    This function alleviates the challenge that no Scine module is a hard dependency for Puffin in order to allow
    the Puffin to bootstrap itself by building the Scine modules, but also relies on the Scine modules in almost all
    jobs.
    The usage should be to add
    >>> from __future__ import annotations
    >>> from typing import TYPE_CHECKING

    to the top of the file and then use the function straight after the imports as follows:
    >>> if module_exists("scine_database") or TYPE_CHECKING:
    >>>     import scine_database as db

    this allows typehinting all functions with database objects and still makes the file importable without the
    database module.

    Returns
    -------
    bool
        True if the module is installed, False otherwise.
    """
    if module_name.lower() == "cp2k":
        return os.getenv("CP2K_BINARY_PATH") is not None
    elif module_name.lower() == "gaussian":
        return os.getenv("GAUSSIAN_BINARY_PATH") is not None
    elif module_name.lower() == "orca":
        return os.getenv("ORCA_BINARY_PATH") is not None
    elif module_name.lower() == "turbomole":
        return os.getenv("TURBODIR") is not None
    elif module_name.lower() == "ams":
        possibles = ['AMSHOME', 'AMSBIN', 'AMS_BINARY_PATH']
        return any(os.getenv(p) is not None for p in possibles)
    elif module_name.lower() == "mrcc":
        return os.getenv("MRCC_BINARY_PATH") is not None
    else:
        return module_name in (name for loader, name, ispkg in iter_modules())


def calculator_import_resolve(dependency_list: List[str]) -> None:
    """
    Automatically loads the calculators if they are in the dependency list.

    Parameters
    ----------
    dependency_list : List[str]
        The list of dependencies.
    """
    for d in dependency_list:
        if d == "scine_sparrow":
            import scine_sparrow  # noqa # pylint: disable=(unused-import,import-error)
        elif d == "scine_ams_wrapper":
            import scine_ams_wrapper  # noqa # pylint: disable=(unused-import,import-error)
        elif d == "scine_dftbplus_wrapper":
            import scine_dftbplus_wrapper  # noqa # pylint: disable=(unused-import,import-error)
        elif d == "scine_serenity_wrapper":
            import scine_serenity_wrapper  # noqa # pylint: disable=(unused-import,import-error)
        elif d == "scine_swoose":
            import scine_swoose  # noqa # pylint: disable=(unused-import,import-error)
        elif d == "scine_xtb_wrapper":
            import scine_xtb_wrapper  # noqa # pylint: disable=(unused-import,import-error)
        elif d == "scine_parrot":
            import scine_parrot  # noqa # pylint: disable=(unused-import,import-error)


def dependency_addition(dependencies: List[str]) -> List[str]:
    """
    A utility function that adds the "scine" prefix to specific Scine dependencies and adds additional dependencies
    based on the given dependencies, such as the utilities package or the Sparrow calculator to the ReaDuct dependency.

    Parameters
    ----------
    dependencies : List[str]
        The list of dependencies.

    Returns
    -------
    List[str]
        The updated list of dependencies with proper "scine" prefixes and additional dependencies.
    """

    # allow to give scine packages without 'scine_' prefix
    short_terms = ['readuct', 'swoose', 'sparrow', 'molassembler', 'database', 'utilities', 'kinetx', 'xtb_wrapper',
                   'ams_wrapper', 'serenity_wrapper', 'dftbplus_wrapper', 'parrot']
    dependencies = ['scine_' + d if d in short_terms else d for d in dependencies]
    # dependencies of key as value list, only utilities must not be included
    dependency_data = {
        'scine_readuct': ['scine_sparrow'],
    }
    for package, dependency in dependency_data.items():
        if package in dependencies:
            dependencies += dependency
    # add utilities
    if any('scine' in d for d in dependencies):
        dependencies.append('scine_utilities')
    return list(set(dependencies))  # give back unique list


def requires(*dependencies) -> Callable:
    """
    This function is meant to be used as a decorator for functions that require specific dependencies to be installed.
    It checks if the given dependencies are installed and raises an ImportError if not.
    Then it automatically loads the calculators if they are in the dependency list.

    Examples
    --------
    Add this function as a decorator with the required modules as arguments

    >>> @requires("readuct", "database")
    >>> def function():
    >>>     ...

    Returns
    -------
    Callable
        The wrapped function.

    Raises
    ------
    MissingDependencyError
        If the given dependencies are not installed, but the function is called.
    """
    dependency_list: List[str] = list(dependencies)
    dependency_list = dependency_addition(dependency_list)

    def wrap(f: Callable):

        @wraps(f)
        def wrapped_f(*args, **kwargs):
            if not all(module_exists(d) for d in dependency_list):
                raise MissingDependencyError(f"Execution of function requires {dependency_list} to be installed")
            calculator_import_resolve(dependency_list)
            return f(*args, **kwargs)
        return wrapped_f

    return wrap


class MissingDependency(ModuleType):
    """
    This class is meant to be used as a placeholder for missing dependencies.
    It allows access to arbitrarily named attributes and methods, but raises a MissingDependencyError when accessed
    or called.
    """

    def __getattribute__(self, key):
        if key.startswith("__"):
            return super().__getattribute__(key)
        raise NameError(f"Attribute '{key}' access not allowed")

    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise MissingDependencyError(f"Attribute '{name}' access not allowed")
        return method

    def __call__(self, *args, **kwargs):
        raise MissingDependencyError("Method execution not allowed")
