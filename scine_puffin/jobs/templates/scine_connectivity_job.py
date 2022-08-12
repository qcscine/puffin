# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Dict, Set
import sys

from .job import job_configuration_wrapper
from .scine_job import ScineJob
from scine_puffin.config import Configuration
from scine_puffin.utilities import masm_helper


class ConnectivityJob(ScineJob):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface
    and aim at deriving some form of bonding information within a job.
    This can be simple bond orders and/or a full Molassembler graph.
    """

    def __init__(self):
        super().__init__()
        self.name = "ConnectivityJob"
        self.own_expected_results = ["energy", "bond_orders"]
        self.connectivity_settings = {
            "only_distance_connectivity": False,  # determine connectivity solely based on distances
            "sub_based_on_distance_connectivity": True,  # remove connectivity based on distances
            "add_based_on_distance_connectivity": True,  # add connectivity based on distances
            "enforce_bond_order_model": True,  # Whether bond orders must have an identical model
            "dihedral_retries": 100,  # Number of attempts to generate the dihedral decision during conformer generation
        }

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs():
        return ["database", "molassembler", "readuct", "utils"]

    def connectivity_settings_from_only_connectivity_settings(self):
        """
        Overwrite default connectivity settings based on settings of configured Calculation and expect no other
        settings to be present. Throws if there are other settings present.

        Notes
        -----
        * Requires run configuration
        * May throw exception
        """
        custom_settings = self._calculation.get_settings().as_dict()
        self.extract_connectivity_settings_from_dict(custom_settings)
        if custom_settings:  # unknown keys left
            self.raise_named_exception(
                "Error: The key(s) "
                + str(list(custom_settings.keys()))
                + " was/were not recognized."
            )

    def extract_connectivity_settings_from_dict(self, dictionary: Dict[str, bool]):
        """
        Overwrite default connectivity settings based on given dictionary and removes those from the
        dictionary.
        """
        for key, value in self.connectivity_settings.items():
            self.connectivity_settings[key] = dictionary.pop(key, value)

    def make_bond_orders_from_calc(self, systems: dict, key: str):
        """
        Gives bond orders for the specified system based on the connectivity settings of this class.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        systems :: Dict[str, utils.core.Calculator]
            Dictionary of system names to calculators representing them
        key :: str
            Index into systems dictionary to get bond orders for

        Returns
        -------
        bond_orders :: utils.BondOrderCollection (Scine::Utilties::BondOrderCollection)
            The bond orders of the system
        """
        import scine_readuct as readuct

        # Distance based bond orders
        if self.connectivity_settings["only_distance_connectivity"]:
            bond_orders = self.distance_bond_orders(systems[key].structure)
        # Bond order calculation with readuct
        else:
            if not self.expected_results_check(systems, [key], ["energy", "bond_orders", "atomic_charges"])[0]:
                # we don't have results yet, so we calculate
                systems, success = readuct.run_single_point_task(
                    systems, [key], require_bond_orders=True, require_charges=True
                )
                self.throw_if_not_successful(
                    success, systems, [key], ["energy", "bond_orders", "atomic_charges"]
                )
            bond_orders = systems[key].get_results().bond_orders

        return bond_orders

    def make_graph_from_calc(self, systems: dict, key: str) -> str:
        """
        Runs bond orders for the specified name in the dictionary of systems if not present already and
        return cbor graph for based on them.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        systems :: Dict[str, utils.core.Calculator]
            Dictionary of system names to calculators representing them
        key :: str
            Index into systems dictionary to get bond orders for

        Returns
        -------
        graph_cbor :: str
            Serialized representation of interpreted molassembler molecule.
        """
        import scine_utilities as utils

        if self.connectivity_settings["only_distance_connectivity"]:
            bond_orders = self.distance_bond_orders(systems[key].structure)
        else:
            bond_orders = systems[key].get_results().bond_orders
            if bond_orders is None:
                bond_orders = self.make_bond_orders_from_calc(systems, key)

        pbc_string = systems[key].settings.get(utils.settings_names.periodic_boundaries, "")
        return masm_helper.get_cbor_graph(systems[key].structure, bond_orders, self.connectivity_settings, pbc_string)

    def make_decision_lists_from_calc(self, systems: dict, key: str):
        """
        Calculates bond orders for the specified name in the dictionary of systems
        if not present already.
        Then generates and returns the decision lists.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        systems :: Dict[str, utils.core.Calculator]
            Dictionary of system names to calculators representing them
        key :: str
            Index into systems dictionary to get bond orders for

        Returns
        -------
        decision_lists :: List[str]
            Decision lists per molecule in structure.
        """
        import scine_utilities as utils

        if self.connectivity_settings["only_distance_connectivity"]:
            bond_orders = self.distance_bond_orders(systems[key].structure)
        else:
            bond_orders = systems[key].get_results().bond_orders
            if bond_orders is None:
                bond_orders = self.make_bond_orders_from_calc(systems, key)

        pbc_string = systems[key].settings.get(utils.settings_names.periodic_boundaries, "")
        return masm_helper.get_decision_lists(
            systems[key].structure,
            bond_orders,
            self.connectivity_settings,
            pbc_string)

    def surface_indices_all_structures(self):
        """
        Get the combined surface indices of all structures of the configured calculation. Throws if a structure is
        specified to be a surface but does not have surface_indices property.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Returns
        -------
        surface_indices :: set
            A set of all surface indices over all structures combined assuming an atom ordering identical to the
            addition of all structures in their order within the calculation.
        """
        import scine_database as db

        surface_indices = []
        n_atoms = 0
        for sid in self._calculation.get_structures():
            structure = db.Structure(sid, self._structures)
            indices = self.surface_indices(structure)
            surface_indices += [index + n_atoms for index in indices]
            n_atoms += len(structure.get_atoms())
        return set(surface_indices)

    def surface_indices(self, structure) -> Set[int]:
        import scine_database as db

        if "surface" in str(structure.get_label()).lower():
            if not structure.has_property("surface_atom_indices"):
                self.raise_named_exception(
                    "The structure is a surface, but has no property indicating "
                    "the surface atom indices."
                )
            surface_atoms_prop = db.VectorProperty(structure.get_property("surface_atom_indices"))
            surface_atoms_prop.link(self._properties)
            surface_indices = set(surface_atoms_prop.get_data())
        else:
            surface_indices = set()
        return surface_indices

    def distance_bond_orders(self, structure):
        """
        Construct bond order solely based on distance for either an AtomCollection or a Database Structure.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        structure :: Union[utils.AtomCollection, db.Structure]
            Either an AtomCollection or a structure for which distance based bond orders are constructed.

        Returns
        -------
        bond_orders :: utils.BondOrderCollection (Scine::Utilties::BondOrderCollection)
            The bond orders of the structure.
        """
        import scine_database as db
        import scine_utilities as utils

        if isinstance(structure, db.Structure):
            atoms = structure.get_atoms()
            surface_indices = self.surface_indices(structure)
        elif isinstance(structure, utils.AtomCollection):
            atoms = structure
            surface_indices = self.surface_indices_all_structures()
        else:
            self.raise_named_exception(
                "Unknown type of provided structure for distance bond orders."
            )

        model = self._calculation.get_model()
        # generate bond orders depending on model and surface atoms
        if model.periodic_boundaries and model.periodic_boundaries != "none":
            # PeriodicSystem handles everything
            pbc = utils.PeriodicBoundaries(model.periodic_boundaries)
            ps = utils.PeriodicSystem(pbc, atoms, surface_indices)
            bond_orders = ps.construct_bond_orders()
        elif surface_indices:
            # Use mixture of Nearest Neighbors and BondDetector
            bond_orders = utils.SolidStateBondDetector.detect_bonds(atoms, surface_indices)
        else:
            bond_orders = utils.BondDetector.detect_bonds(atoms)
        return bond_orders

    def add_graph(self, structure, bond_orders):
        """
        Add molassembler graph information to a Database structure based on the given bond orders.

        Parameters
        ----------
        structure :: Union[utils.AtomCollection, db.Structure]
            Either an AtomCollection or a structure for which distance based bond orders are constructed.
        bond_orders :: utils.BondOrderCollection (Scine::Utilties::BondOrderCollection)
            The bond orders of the structure.
        """
        print("\nGenerating Molassembler graphs")
        if structure.has_graph("masm_cbor_graph"):
            sys.stderr.write("Warning: The structure had a graph already. This graph will be replaced.")

        masm_helper.add_masm_info(
            structure,
            bond_orders,
            self.connectivity_settings,
            list(self.surface_indices(structure)),
        )
        # Print the graph representations to the output
        print("Generated graph:")
        print("masm_cbor_graph: " + structure.get_graph("masm_cbor_graph"))
        print("masm_decision_list: " + structure.get_graph("masm_decision_list"))

    def query_bond_orders(self, structure):
        """
        Query the given Database structure for bond orders based on the model of the configured calculation

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        structure :: db.Structure (Scine::Database::Structure)
            A database structure to query.
        Returns
        -------
        db_bond_orders :: db.SpareMatrixProperty (Scine::Database::SparseMatrixProperty)
            A database property holding bond orders.
        """
        import scine_database as db

        # db bond orders
        bos = structure.query_properties("bond_orders", self._calculation.get_model(), self._properties)
        if len(bos) == 0 and not self.connectivity_settings["enforce_bond_order_model"]:
            bos = structure.get_properties("bond_orders")
        if len(bos) == 0:
            self.raise_named_exception("Error: Missing bond orders.")

        print("Interpreting from bond orders with property ID '{}'.".format(bos[0].string()))
        db_bond_orders = db.SparseMatrixProperty(bos[0])
        db_bond_orders.link(self._properties)

        return db_bond_orders

    @staticmethod
    def bond_orders_from_db_bond_orders(structure, db_bond_orders):
        """
        A shortcut to construct a BondOrderCollection from a Database Property holding bond orders.
        Returns
        -------
        bond_orders :: utils.BondOrderCollection (Scine::Utilties::BondOrderCollection)
            The bond orders of the structure.
        """
        import scine_utilities as utils

        atoms = structure.get_atoms()
        bond_orders = utils.BondOrderCollection(len(atoms))
        bond_orders.matrix = db_bond_orders.get_data()
        return bond_orders
