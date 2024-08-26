# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union, TYPE_CHECKING

from scine_puffin.utilities.imports import module_exists, requires, MissingDependency
if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


@requires("database")
def get_compound_or_flask(object_id: db.ID, object_type: db.CompoundOrFlask, compounds: db.Collection,
                          flasks: db.Collection) -> Union[db.Compound, db.Flask]:
    """
    Construct the compound or flask object depending on the type. Flask and Compound have a large set of functions
    in common. Through this function, we do not have to differentiate between them at every point.

    Parameters
    ----------
    object_id : db.ID
        The ID of the object to construct.
    object_type : db.CompoundOrFlask
        The label for Compound or Flaks.
    compounds : db.Collection
        The compounds collection.
    flasks : db.Collection
        The flasks collection.

    Returns
    -------
    Either the flask or compound object.

    Raises
    ------
    RuntimeError
        If the object_type is unknown.
    """
    if object_type == db.CompoundOrFlask.COMPOUND:
        return db.Compound(object_id, compounds)
    if object_type == db.CompoundOrFlask.FLASK:
        return db.Flask(object_id, flasks)
    raise RuntimeError("Requested aggregate type is not supported.")
