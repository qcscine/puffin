#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
from typing import Optional


def get_test_db_credentials(name: str = "puffin_unittests"):
    """
    Generate a set of credentials pointing to a database and server.
    The server IP and port are assumed to be `127.0.0.1` and `27017`
    unless specified otherwise with the environment variables
    ``TEST_MONGO_DB_IP`` and ``TEST_MONGO_DB_PORT``.

    Parameters
    ----------
    name :: str
        The name of the database to connect to.

    Returns
    -------
    result :: db.Credentials
        The credentials to access the test database.
    """
    import scine_database as db
    ip = os.environ.get('TEST_MONGO_DB_IP', "127.0.0.1")
    port = os.environ.get('TEST_MONGO_DB_PORT', "27017")
    return db.Credentials(ip, int(port), name)


def get_clean_db(name: str = "puffin_unittests"):
    """
    Generate a clean database using ``get_test_db_credentials`` to determine
    the database IP.

    Parameters
    ----------
    name :: str
        The name of the database to connect to.

    Returns
    -------
    result :: db.Manager
        The database manager, connected to the requested server and named
        database.
    """
    import scine_database as db
    credentials = get_test_db_credentials(name)
    manager = db.Manager()
    manager.set_credentials(credentials)
    try:
        manager.connect()
        manager.wipe()
    except BaseException:
        manager.wipe(True)
        manager.connect()
    manager.init()
    return manager


def add_structure(manager, xyz_path, label, charge: int = 0, multiplicity: int = 1):
    """
    Generates a Structure in the database according to the
    specifications given as arguments.

    Parameters
    ----------
    manager :: db.Manager
        The manager of the database to create data in.
    xyz_path :: str
        Path to the xyz file containing the structures coordinates.
    label :: db.Label
        The label of the structure to be generated.
    charge :: int
        The charge of the structure
    multiplicity :: int
        The multiplicity of the structure

    Returns
    -------
    structure :: db.Structure
        The generated Structure linked to its collection
    """
    import scine_database as db
    import scine_utilities as utils
    structures = manager.get_collection("structures")
    atoms, _ = utils.io.read(xyz_path)
    structure = db.Structure.make(atoms, charge, multiplicity, structures)
    structure.set_label(label)
    return structure


def add_calculation(manager, model, job, structures, settings: Optional[dict] = None):
    """
    Generates a Calculation in the database according to the
    specifications given as arguments.

    Parameters
    ----------
    manager :: db.Manager
        The manager of the database to create data in.
    model :: db.Model
        The Model of the calculation.
    job :: db.Job
        The Job of the calculation.
    structures :: List[db.ID]
        List of structure IDs set as structures used as
        input for the calculation
    settings :: dict
        Settings to be set in the Calculation.

    Returns
    -------
    calculation :: db.Calculation
        The generated Calculation linked to its collection
    """
    import scine_database as db
    import scine_utilities as utils

    if settings is None:
        settings = {}
    calculations = manager.get_collection("calculations")
    calculation = db.Calculation.make(model, job, structures, calculations)
    calculation.set_settings(utils.ValueCollection(settings))
    calculation.set_status(db.Status.NEW)
    return calculation


def add_compound_and_structure(manager, xyz_file: str = "proline_acid.xyz"):
    """
    Generates a Compound with one structure according to the given xyz_file.

    Parameters
    ----------
    manager :: db.Manager
        The manager of the database to create data in.
    xyz_file :: str
        The xyz file name for the structure that is added
    Returns
    -------
    compound :: db.Compound
        The Compound.
    """
    import scine_database as db
    from .resources import resource_path
    compounds = manager.get_collection("compounds")
    path = os.path.join(resource_path(), xyz_file)
    structure = add_structure(manager, path, db.Label.MINIMUM_OPTIMIZED)
    new_compound = db.Compound.make([structure.id()], compounds)
    structure.set_compound(new_compound.id())
    return new_compound


def add_flask_and_structure(manager, xyz_file: str = "proline_acid.xyz"):
    """
    Generates a Flask with one structure according to the given xyz_file.

    Parameters
    ----------
    manager :: db.Manager
        The manager of the database to create data in.
    xyz_file :: str
        The xyz file name for the structure that is added
    Returns
    -------
    flask :: db.Flask
        The Flask.
    """
    import scine_database as db
    from .resources import resource_path
    flasks = manager.get_collection("flasks")
    path = os.path.join(resource_path(), xyz_file)
    structure = add_structure(manager, path, db.Label.COMPLEX_OPTIMIZED)
    new_flask = db.Flask.make([structure.id()], [], flasks)
    structure.set_compound(new_flask.id())
    return new_flask


def add_reaction(manager, lhs_compound_ids, rhs_compound_ids):
    """
    Generates a Reaction without any elementary steps.

    Parameters
    ----------
    manager :: db.Manager
        The manager of the database to create data in.
    lhs_compound_ids :: List[db.ID]
        The left-hand side of the reaction.
    rhs_compound_ids :: List[db.ID]
        The right-hand side of the reaction.
    Returns
    -------
    compound :: db.Compound
        The Reaction.
    """
    import scine_database as db
    reactions = manager.get_collection("reactions")
    compounds = manager.get_collection("compounds")
    new_reaction = db.Reaction.make(lhs_compound_ids, rhs_compound_ids, reactions)
    c_ids = lhs_compound_ids + rhs_compound_ids
    for c_id in c_ids:
        compound = db.Compound(c_id, compounds)
        compound.add_reaction(new_reaction.id())
    return new_reaction
