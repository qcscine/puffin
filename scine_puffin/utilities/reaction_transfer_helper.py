# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Set, Union, Optional, TYPE_CHECKING
import sys

import numpy as np

from .transfer_helper import TransferHelper
from .surface_helper import update_slab_dict
from scine_puffin.jobs.templates.scine_react_job import ReactJob
from scine_puffin.utilities.imports import module_exists, MissingDependency
if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ReactionTransferHelper(TransferHelper):
    """
    A class that can transfer some properties from reactants to products
    after we found a reaction.
    """

    surface_indices_name = "surface_atom_indices"
    slab_dict_name = "slab_dict"

    def __init__(self, react_job: ReactJob, properties: db.Collection,
                 alternative_component_map: Optional[List[int]] = None) -> None:
        """
        Constructor based on a puffin ReactJob

        Parameters
        ----------
        react_job : ReactJob
            The job currently executing that found an elementary step
        properties : db.Collection
            The properties collection of the database
        alternative_component_map : Optional[List[int]]
            An alternative component map to use instead of the product component map in the react job
        """
        self.react_job = react_job
        self.properties = properties
        self.alternative_component_map = alternative_component_map
        self._special_properties = [
            self.surface_indices_name,
            self.slab_dict_name
        ]

    def transfer_properties(self, old_structure: db.Structure, new_structure: db.Structure,
                            properties_to_transfer: List[str]) \
            -> None:
        """
        Transfer the given properties from one structure to another.

        Parameters
        ----------
        old_structure : db.Structure
            Structure holding property
        new_structure : db.Structure
            Structure getting property
        properties_to_transfer : List[str]
            A list of names of the properties to transfer
        """
        for prop in properties_to_transfer:
            if old_structure.has_property(prop):
                if prop in self._special_properties:
                    sys.stderr.write(f"{prop} can only be transferred properly by transferring them to all reaction "
                                     f"products at once. This was not done.")
                self.simple_transfer_all(old_structure, new_structure, properties_to_transfer)

    def transfer_properties_between_multiple(self,
                                             old_structures: List[db.Structure],
                                             new_structures: List[db.Structure],
                                             properties_to_transfer: List[str]) -> None:
        """
        Transfer properties between multiple structures to multiple other structures.
        This exists because some property transfers may require the knowledge of all reactants
        and all products.

        Parameters
        ----------
        old_structures : List[db.Structure]
            The structures holding the properties
        new_structures : List[db.Structure]
            The structures receiving the properties
        properties_to_transfer : List[str]
            The names of the properties to transfer

        Raises
        ------
        RuntimeError
            The react job is missing a component map to map reactants to products.
        NotImplementedError
            Some unknown properties are requested for transferring
        """
        if self.react_job.products_component_map is None:
            raise RuntimeError("Could not transfer the properties to the products without the job holding a "
                               "component map")
        for prop in properties_to_transfer:
            if prop not in self._special_properties:
                for old_structure in old_structures:
                    for new_structure in new_structures:
                        self.simple_transfer(old_structure, new_structure, prop)
            elif prop == self.surface_indices_name:
                self._surface_indices_impl(old_structures, new_structures)
            elif prop == self.slab_dict_name:
                self._slab_dict_impl(old_structures, new_structures)
            else:
                raise NotImplementedError(f"Have not implemented a method to transfer "
                                          f"{prop} with {self.__class__.__name__}")

    def _surface_indices_impl(self, old_structures: List[db.Structure], new_structures: List[db.Structure]) -> None:
        """
        The method implementing the transfer of surface indices to track which nuclei
        belong to the solid state surface structure.

        Parameters
        ----------
        old_structures : List[db.Structure]
            The structures holding the properties
        new_structures : List[db.Structure]
            The structures receiving the properties
        """
        new_surface_indices = self._determine_new_indices(old_structures, new_structures)
        calculation = self.react_job.get_calculation()
        thresh = self.react_job.connectivity_settings["n_surface_atom_threshold"]
        for new_indices, new_structure in zip(new_surface_indices, new_structures):
            # do not transfer single surface atom, since we assume that this means we don't have a surface anymore
            if len(new_indices) > thresh:
                self._sanity_checks(new_structure, self.surface_indices_name)
                new_property = db.VectorProperty.make(self.surface_indices_name, calculation.get_model(),
                                                      np.array([float(i) for i in new_indices]), new_structure.id(),
                                                      calculation.id(), self.properties)
                new_structure.set_property(self.surface_indices_name, new_property.id())

    def _slab_dict_impl(self, old_structures: List[db.Structure], new_structures: List[db.Structure]) -> None:
        """
        The implementation to transfer the slab dictionary information.

        Parameters
        ----------
        old_structures : List[db.Structure]
            The structures holding the properties
        new_structures : List[db.Structure]
            The structures receiving the properties
        """
        for old_structure in old_structures:
            if old_structure.has_property(self.slab_dict_name):
                old_structure_with_prop = old_structure
                break
        else:
            # no slab dict in all old_structures
            return
        new_surface_indices = self._determine_new_indices(old_structures, new_structures)
        thresh = self.react_job.connectivity_settings["n_surface_atom_threshold"]
        for new_indices, new_structure in zip(new_surface_indices, new_structures):
            if len(new_indices) > thresh:
                self._sanity_checks(new_structure, self.slab_dict_name)
                self.simple_transfer(old_structure_with_prop, new_structure, self.slab_dict_name)
                update_slab_dict(new_structure, self.properties, replace_property=True)

    def _determine_new_indices(self, old_structures: List[db.Structure], new_structures: List[db.Structure]) \
            -> List[List[int]]:
        """
        Maps the surface indices from the old structures for the new structures as list for each
        new structure specifying its surface indices.

        Parameters
        ----------
        old_structures : List[db.Structure]
            The structures holding the properties
        new_structures : List[db.Structure]
            The structures receiving the properties

        Returns
        -------
        List[List[int]]
            A list for each new structure giving its surface indices starting with 0

        Raises
        ------
        RuntimeError
            The old and new structure do not fit together
        """
        n_new_structures = len(new_structures)
        new_surface_indices: List[List[int]] = [[]] * n_new_structures  # new surface indices for each new structure
        old_indices = self.react_job.surface_indices_all_structures()
        if not old_indices:
            return new_surface_indices
        # sanity checks
        n_atoms_old = sum(len(s.get_atoms()) for s in old_structures)
        n_atoms_new = sum(len(s.get_atoms()) for s in new_structures)
        if n_atoms_old != n_atoms_new:
            raise RuntimeError(f"{self.__class__.__name__} could not transfer {self.surface_indices_name}, because"
                               f"not all old structures and new structures were given")
        if max(old_indices) >= n_atoms_old:
            raise RuntimeError(f"{self.__class__.__name__} could not transfer {self.surface_indices_name}, because"
                               f"the {self.surface_indices_name} do not fit to the size of the given structures")
        component_map = self.alternative_component_map if self.alternative_component_map is not None \
            else self.react_job.products_component_map
        assert component_map is not None
        return self.map_total_indices_to_split_structure_indices(old_indices, component_map)

    @staticmethod
    def map_total_indices_to_split_structure_indices(total_indices: Union[Set[int], List[int]],
                                                     component_map: List[int]) -> List[List[int]]:
        """
        Maps the total indices to the split structure indices.
        This relies on the fact that component_map.apply keeps the order of indices within the new structures
        So we can infer the index in the new structure based on filling up

        Parameters
        ----------
        total_indices : List[int]
            The total indices
        component_map : List[int]
            The component map specifying which total index belongs to which split structure

        Returns
        -------
        List[List[int]]
            The split structure indices
        """
        n_new_structures = max(component_map) + 1
        # prepare output object
        new_surface_indices: List[List[int]] = []
        for _ in range(n_new_structures):  # new surface indices for each new structure
            new_surface_indices.append([])
        if not total_indices:
            return new_surface_indices
        # fill up
        current_indices = [0] * n_new_structures  # containing info of current index in each new structure
        for i, new_structure_index in enumerate(component_map):
            if i in total_indices:
                new_surface_indices[new_structure_index].append(current_indices[new_structure_index])
            current_indices[new_structure_index] += 1
        return new_surface_indices

    @staticmethod
    def _sanity_checks(new_structure: db.Structure, prop_name: str) -> None:
        if "surface" not in str(new_structure.get_label()).lower():
            sys.stderr.write(f"Something went wrong with the index transfer or the labeling of "
                             f"{str(new_structure.id())}, this structure gets {prop_name}, but does not "
                             f"have a surface label")
        if new_structure.has_property(prop_name):
            sys.stderr.write(f"New structure {str(new_structure.id())} already had {prop_name}."
                             f" The property was somehow transferred more than once")
