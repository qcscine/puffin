# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from ast import literal_eval
from typing import TYPE_CHECKING
from platform import python_version

from pymatgen.core import Lattice
from pymatgen.core.surface import Slab
import pymatgen

from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


def get_slab_dict(structure: db.Structure, properties: db.Collection) -> dict:
    """
    Generate the dictionary defining a pymatgen Slab object from a database Structure,
    which must hold the required information as a property.

    Notes
    -----
    This code is taken from scine.chemoton.utilities.surfaces.pymatgen_interface


    Parameters
    ----------
    structure : db.Structure
        The periodic Structure
    properties : db.Collection
        The properties collection to link the structure's properties

    Returns
    -------
    dict
        A dictionary that can be used as a constructor for the pymatgen.core.surface.Slab

    Raises
    ------
    RuntimeError
        The property 'slab_dict' not present
    """
    if not structure.has_property("slab_dict"):
        raise RuntimeError(f"Slab information is missing for structure '{str(structure)}'")
    dict_info = db.StringProperty(structure.get_property('slab_dict'), properties)
    dict_info_string = dict_info.get_data()
    # remove some specific extra strings from representation to be able to make dict out of string
    dict_info_string = dict_info_string.replace("]])", "]]")
    dict_info_string = dict_info_string.replace("array(", "")
    # transform into dict
    return literal_eval(dict_info_string)


def update_slab_dict(structure: db.Structure, properties: db.Collection, replace_property: bool = False) -> None:
    """
    Update the slab dict property of the given structure with its current
    positions and periodic boundary conditions

    Notes
    -----
    This code is taken from scine.chemoton.utilities.surfaces.pymatgen_interface

    Parameters
    ----------
    structure : db.Structure
        The structure holding the property
    properties : db.Collection
        The properties collection to link the structure's properties
    replace_property : bool, optional
        If the old property should be replaced with the new one

    Raises
    ------
    RuntimeError
        The structure is not periodic
    """
    slab_dict_name = "slab_dict"
    slab = Slab.from_dict(get_slab_dict(structure, properties))
    pbc_string = structure.get_model().periodic_boundaries
    if not pbc_string or pbc_string.lower() == "none":
        raise RuntimeError("Structure is missing periodic boundary conditions")
    pbc = utils.PeriodicBoundaries(pbc_string)
    atoms = structure.get_atoms()
    ele = [utils.ElementInfo.symbol(e) for e in atoms.elements]
    coords = pbc.transform(atoms.positions, False)
    lattice = _construct_pmg_lattice(pbc)
    new_slab = Slab(lattice, ele, coords, slab.miller_index, slab.oriented_unit_cell, slab.shift,
                    slab.scale_factor, coords_are_cartesian=False)
    if not replace_property:
        dict_property = db.StringProperty(structure.get_property(slab_dict_name), properties)
        dict_property.set_data(str(new_slab.as_dict()))
        return
    structure.clear_properties(slab_dict_name)
    new_property = db.StringProperty.make(slab_dict_name, structure.get_model(),
                                          str(new_slab.as_dict()), properties)
    structure.set_property(slab_dict_name, new_property.id())


def _construct_pmg_lattice(pbc: utils.PeriodicBoundaries) -> Lattice:
    matrix = pbc.matrix * utils.ANGSTROM_PER_BOHR
    if python_version() >= "3.8":
        from importlib.metadata import version
        pymatgen_version = version(pymatgen.__name__)
    elif hasattr(pymatgen, "__version__"):
        pymatgen_version = getattr(pymatgen, "__version__")
    else:
        pymatgen_version = "0"
    if pymatgen_version >= "2022":
        return Lattice(matrix, pbc.periodicity)  # type: ignore  # pylint: disable=too-many-function-args
    return Lattice(matrix)
