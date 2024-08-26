# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class TransferHelper(ABC):
    """
    An abstract base class to transfer properties from one or more structures
    to one or more other structures.
    """

    @abstractmethod
    def transfer_properties(self, old_structure: db.Structure, new_structure: db.Structure,
                            properties_to_transfer: List[str]) \
            -> None:
        """
        Transfer properties between individual structures based on given property names

        Parameters
        ----------
        old_structure : db.Structure
            The structure holding the properties
        new_structure : db.Structure
            The structure receiving the properties
        properties_to_transfer : List[str]
            The names of the properties to transfer
        """
        raise NotImplementedError

    @abstractmethod
    def transfer_properties_between_multiple(self,
                                             old_structures: List[db.Structure],
                                             new_structures: List[db.Structure],
                                             properties_to_transfer: List[str]) -> None:
        """
        Transfer properties between multiple structures based on given property names

        Parameters
        ----------
        old_structures : List[db.Structure]
            The structures holding the properties
        new_structures : List[db.Structure]
            The structures receiving the properties
        properties_to_transfer: List[str]
            The names of the properties to transfer
        """
        raise NotImplementedError

    @staticmethod
    def simple_transfer_all(old_structure: db.Structure, new_structure: db.Structure, properties: List[str]) \
            -> None:
        """
        Simply set the id of the given properties from one structure to another one

        Parameters
        ----------
        old_structure : db.Structure
            The structure holding the properties
        new_structure : db.Structure
            The structure receiving the properties
        properties : List[str]
            The names of the properties to transfer
        """
        for prop in properties:
            TransferHelper.simple_transfer(old_structure, new_structure, prop)

    @staticmethod
    def simple_transfer(old_structure: db.Structure, new_structure: db.Structure, property_to_transfer: str) \
            -> None:
        """
        Transfer a single property from one structure to another.

        Parameters
        ----------
        old_structure : db.Structure
            The structure holding the properties
        new_structure : db.Structure
            The structure receiving the properties
        property_to_transfer : str
            The name of the property to transfer
        """
        if old_structure.has_property(property_to_transfer):
            prop_id = old_structure.get_property(property_to_transfer)
            new_structure.set_property(property_to_transfer, prop_id)
