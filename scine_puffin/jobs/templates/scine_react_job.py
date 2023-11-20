# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from math import ceil
from typing import Any, Dict, List, Tuple, Union, Optional, Iterator, Set
import numpy as np
import sys
import os
from copy import deepcopy

import scine_database as db
import scine_utilities as utils

from .job import job_configuration_wrapper, breakable
from .scine_connectivity_job import ConnectivityJob
from .scine_hessian_job import HessianJob
from .scine_optimization_job import OptimizationJob
from .scine_observers import StoreEverythingObserver
from scine_puffin.config import Configuration
from scine_puffin.utilities.scine_helper import SettingsManager, update_model
from scine_puffin.utilities.program_helper import ProgramHelper
from scine_puffin.utilities import masm_helper


class ReactJob(OptimizationJob, HessianJob, ConnectivityJob):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface to find new reactions.
    """

    def __init__(self):
        super().__init__()
        self.name = "ReactJob"  # to be overwritten by child
        self.exploration_key = ""  # to be overwritten by child
        self.own_expected_results = []
        self.rc_key = "rc"
        self.job_key = "job"
        self.rc_opt_system_name = "rcopt"
        self.single_point_key = "sp"
        self.no_irc_structure_matches_start = False
        # to be extended by child:
        self.settings: Dict[str, Dict[str, Any]] = {
            self.job_key: {
                "imaginary_wavenumber_threshold": 0.0,
                "spin_propensity_check_for_unimolecular_reaction": True,
                "spin_propensity_energy_range_to_save": 200.0,
                "spin_propensity_optimize_all": True,
                "spin_propensity_energy_range_to_optimize": 500.0,
                "spin_propensity_check": 2,
                "store_full_mep": False,
                "store_all_structures": False,
                "n_surface_atom_threshold": 1,
            },
            self.rc_key: {
                "minimal_spin_multiplicity": False,
                "x_spread": 2.0,
                "x_rotation": 0.0,
                "x_alignment_0": [],
                "x_alignment_1": [],
                "displacement": 0.0,
            },
            self.rc_opt_system_name: {
                "output": [self.rc_opt_system_name],
                "stop_on_error": False,
                "convergence_max_iterations": 500,
                "geoopt_coordinate_system": "cartesianWithoutRotTrans",
            },
            self.single_point_key: {
                "expect_charge_separation": False,
                "charge_separation_threshold": 0.4
            }
        }
        """
        expect_charge_separation : If true, fragment charges are no longer determined by rounding, i.e, if a product
        consists of multiple molecules (according to its graph), the charges are determined initially by rounding.
        However, then the residual (the difference of the integrated charge to the rounded one) is checked against
        <charge_separation_threshold>. If this residual exceeds the charge separation threshold, the charge is
        increased/lowered by one according to its sign. This is especially useful if a clear charge separation only
        occurs upon separation of the molecules which is often the case for DFT-based descriptions of the electronic
        structure.
        charge_separation_threshold : The threshold for the charge separation (vide supra).
        """
        self.start_graph = ""
        self.end_graph = ""
        self.start_charges = []
        self.start_multiplicities = []
        self.start_decision_lists = []
        self.ref_structure = None
        self.step_direction = None
        self.lhs_barrierless_reaction = False
        self.lhs_complexation = False
        self.rhs_complexation = False
        self.complexation_criterion = -12.0 / 2625.5  # kj/mol
        self.check_charges = True
        self.systems = {}
        self._component_maps: Dict[str, List[int]] = {}
        self.products_component_map: Optional[List[int]] = None

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs():
        return ["database", "molassembler", "readuct", "utils"]

    def clear(self) -> None:
        self.systems = {}
        super().clear()

    def observed_readuct_call(self, call_str: str, systems: dict, input_names: List[str], **kwargs) \
            -> Tuple[dict, bool]:
        import scine_readuct as readuct
        observers = []
        observer_functions = []
        model = self._calculation.get_model()
        model.complete_model(systems[input_names[0]].settings)
        if self.settings[self.job_key]["store_all_structures"]:
            observers.append(StoreEverythingObserver(self._calculation.get_id(), model))
            observer_functions = [observers[-1].gather]
        ret = getattr(readuct, call_str)(systems, input_names, observers=observer_functions, **kwargs)
        # TODO this may need to be redone for multi input calls
        charge = systems[input_names[0]].settings["molecular_charge"]
        multiplicity = systems[input_names[0]].settings["spin_multiplicity"]
        for observer in observers:
            observer.finalize(self._manager, charge, multiplicity)
        return ret

    def observed_readuct_call_with_throw(self, call_str: str, systems: dict, input_names: List[str],
                                         expected_results: List[str], error_msg: str, **kwargs) -> dict:
        systems, success = self.observed_readuct_call(call_str, systems, input_names, **kwargs)
        self.throw_if_not_successful(success, systems, input_names, expected_results, error_msg)
        return systems

    def reactive_complex_preparations(self) -> Tuple[SettingsManager, Union[ProgramHelper, None]]:
        """
        Determine settings for this task based on settings of configured calculation, construct a reactive complex
        from the structures of the configured calculation, build a Scine Calculator for it and construct the
        SettingsManager and ProgramHelper that are returned.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Returns
        -------
        settings_manager, program_helper :: Tuple[SettingsManager, Union[ProgramHelper, None]]
            A database property holding bond orders.
        """
        import scine_molassembler as masm
        # preprocessing of structure
        self.ref_structure = self.check_structures()
        settings_manager, program_helper = self.create_helpers(self.ref_structure)

        # Separate the calculation settings from the database into the task and calculator settings
        # This overwrites any default settings by user settings
        settings_manager.separate_settings(self._calculation.get_settings())
        self.sort_settings(settings_manager.task_settings)
        """ Setup calculators for all reactants """
        self.systems = dict()
        for i, structure_id in enumerate(self._calculation.get_structures()):
            structure = db.Structure(structure_id, self._structures)
            if structure.get_label() == db.Label.SURFACE_ADSORPTION_GUESS:
                continue
            name = "reactant_{:02d}".format(i)
            xyz_name = name + ".xyz"
            utils.io.write(xyz_name, structure.get_atoms())
            # correct PES
            structure_calculator_settings = deepcopy(settings_manager.calculator_settings)
            structure_calculator_settings[utils.settings_names.molecular_charge] = structure.get_charge()
            structure_calculator_settings[utils.settings_names.spin_multiplicity] = structure.get_multiplicity()
            reactant = utils.core.load_system_into_calculator(
                xyz_name,
                self._calculation.get_model().method_family,
                **structure_calculator_settings,
            )
            self.systems[name] = reactant
        reactive_complex_atoms = self.build_reactive_complex(settings_manager)
        # If a user explicitly chooses a reactive complex charge that is different from the start structures
        # we cannot expect any of our a posteriori checks for structure charges to work
        self.check_charges = bool(
            sum(self.start_charges) == settings_manager.calculator_settings[utils.settings_names.molecular_charge]
        )
        if not self.check_charges:
            sys.stderr.write(
                "Warning: You specified a reactive complex charge that differs from the sum of the "
                "structure charges."
            )
        """ Setup Reactive Complex """
        # Set up initial calculator
        utils.io.write("reactive_complex.xyz", reactive_complex_atoms)
        reactive_complex = utils.core.load_system_into_calculator(
            "reactive_complex.xyz",
            self._calculation.get_model().method_family,
            **settings_manager.calculator_settings,
        )
        self.systems[self.rc_key] = reactive_complex
        if program_helper is not None:
            program_helper.calculation_preprocessing(
                self.systems[self.rc_key], self._calculation.get_settings())

        # Calculate bond orders and graph of reactive complex and compare to database graph of start structures
        reactive_complex_graph, self.systems = self.make_graph_from_calc(self.systems, self.rc_key)
        if not masm.JsonSerialization.equal_molecules(reactive_complex_graph, self.start_graph):
            print("Reactive complex graph differs from combined start structure graphs.")
            self.start_graph = reactive_complex_graph
        return settings_manager, program_helper

    def check_structures(self, start_structures: Union[List[db.ID], None] = None) -> db.Structure:
        """
        Perform sanity check whether we only have 1 or 2 structures in the configured calculation. Return a possible
        reference structure (the largest one) for the construction of a ProgramHelper.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        start_structures :: List[db.ID]
            If given, this structure id list is used instead of the list given in self._calculation.get_structures().

        Returns
        -------
        ref_structure :: db.Structure (Scine::Database::Structure)
            The largest structure of the calculation.
        """
        if start_structures is None:
            start_structures = self._calculation.get_structures()
        if len(start_structures) == 0:
            self.raise_named_exception("Not enough structures in input")
        if len(start_structures) == 1:
            ref_id = start_structures[0]
        elif len(start_structures) == 2:
            s1 = db.Structure(start_structures[0], self._structures)
            s2 = db.Structure(start_structures[1], self._structures)
            # choose larger structure as reference
            ref_id = start_structures[0] if len(s1.get_atoms()) >= len(s2.get_atoms()) else start_structures[1]
        elif len(start_structures) == 3 and self._includes_label(start_structures, [db.Label.SURFACE_ADSORPTION_GUESS]):
            # the given reaction has 3 structure, with one already representing the reactive complex
            # the reactive complex is therefore a good ref_Id
            for s in start_structures:
                structure = db.Structure(s, self._structures)
                if structure.get_label() == db.Label.SURFACE_ADSORPTION_GUESS:
                    ref_id = s
        else:
            self.raise_named_exception(
                "Reactive complexes built from more than 2 structures are not supported."
            )
        return db.Structure(ref_id, self._structures)

    def sort_settings(self, task_settings: dict) -> None:
        """
        Take settings of configured calculation and save them in class member. Throw exception for unknown settings.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        task_settings :: dict
            A dictionary from which the settings are taken
        """
        self.extract_connectivity_settings_from_dict(task_settings)
        # Dissect settings into individual user task_settings
        for key, value in task_settings.items():
            for task in self.settings.keys():
                if task == self.job_key:
                    if key in self.settings[task].keys():
                        self.settings[task][key] = value
                        break  # found right task, leave inner loop
                else:
                    indicator_length = len(task) + 1  # underscore to avoid ambiguities
                    if key[:indicator_length] == task + "_":
                        self.settings[task][key[indicator_length:]] = value
                        break  # found right task, leave inner loop
            else:
                self.raise_named_exception(
                    f"The key '{key}' was not recognized."
                )

        if "ircopt" in self.settings.keys() and "output" in self.settings["ircopt"]:
            self.raise_named_exception(
                "Cannot specify a separate output system for the optimization of the IRC end points"
            )

    def save_initial_graphs_and_charges(self, settings_manager: SettingsManager, structures: List[db.Structure]) \
            -> None:
        """
        Save the graphs and charges of the reactants.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        settings_manager :: SettingsManager
            The settings manager for the calculation.
        structures :: List[scine_database.Structure]
            The reactant structures.
        """
        graphs = []
        if len(structures) < 3 or any(s.get_label() == db.Label.SURFACE_ADSORPTION_GUESS for s in structures):
            for i, s in enumerate(structures):
                if s.get_label() == db.Label.SURFACE_ADSORPTION_GUESS:
                    # the given reaction has 3 structure, with one already representing the reactive complex
                    # the reactive complex should be skipped in this sanity check
                    continue
                decision_list = self._decision_list_from_structure(s)
                graph = self._cbor_graph_from_structure(s)
                if decision_list is None:
                    decision_list = ""
                if ";" not in graph:
                    graphs.append(graph)
                    self.start_charges.append(s.get_charge())
                    self.start_multiplicities.append(s.get_multiplicity())
                    self.start_decision_lists.append(decision_list)
                else:
                    graphs += graph.split(';')
                    name = "reactant_{:02d}".format(i)
                    (
                        _,
                        split_graph,
                        split_charges,
                        split_multiplicities,
                        split_decision_lists,
                    ) = self.get_graph_charges_multiplicities(name, s.get_charge())
                    graphs += split_graph
                    self.start_decision_lists += split_decision_lists
                    self.start_charges += split_charges
                    self.start_multiplicities += split_multiplicities

            graphs, self.start_charges, self.start_multiplicities, self.start_decision_lists = (
                list(start_val) for start_val in zip(*sorted(zip(
                    graphs,
                    self.start_charges,
                    self.start_multiplicities,
                    self.start_decision_lists)))
            )
            self.start_graph = ";".join(graphs)
            self.determine_pes_of_rc(settings_manager, *[s for s in structures
                                                         if s.get_label() != db.Label.SURFACE_ADSORPTION_GUESS])
        else:
            # should not be reachable
            self.raise_named_exception(
                "Reactive complexes built from more than 2 structures are not supported."
            )

    def _cbor_graph_from_structure(self, structure: db.Structure) -> str:
        """
        Retrieve masm_cbor_graph from a database structure and throws error if none present.

        Parameters
        ----------
        structure :: db.Structure

        Returns
        -------
        masm_cbor_graph :: str
        """
        if not structure.has_graph("masm_cbor_graph"):
            self.raise_named_exception(f"Missing graph in structure {str(structure.id())}.")
        return structure.get_graph("masm_cbor_graph")

    @staticmethod
    def _decision_list_from_structure(structure: db.Structure) -> Optional[str]:
        """
        Retrieve masm_decision_list from a database structure.
        Returns ``None`` if none present.

        Parameters
        ----------
        structure :: db.Structure

        Returns
        -------
        masm_decision_list :: Optional[str]
        """
        if not structure.has_graph("masm_decision_list"):
            return None
        return structure.get_graph("masm_decision_list")

    def build_reactive_complex(self, settings_manager: SettingsManager) -> utils.AtomCollection:
        """
        Aligns the structure(s) to form a reactive complex and returns the AtomCollection. In case of multiple
        structures, the active site settings are modified to reflect the correct index in the supermolecule.
        Some 'start' information is saved in class members as well.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        settings_manager :: SettingsManager
            The settings_manager in which the charge and multiplicity of the new atoms are set.

        Returns
        -------
        reactive_complex :: utils.AtomCollection (Scine::Utilities::AtomCollection)
            The atoms of the reactive complex
        """
        start_structure_ids = self._calculation.get_structures()
        start_structures = [db.Structure(sid, self._structures) for sid in start_structure_ids]
        self.save_initial_graphs_and_charges(settings_manager, start_structures)
        if len(start_structures) == 1:
            # For an intramolecular structure it is sufficient to provide one
            # structure that is both, start structure and reactive complex
            structure = start_structures[0]
            atoms = structure.get_atoms()
            self.random_displace_atoms(atoms, self.settings[self.rc_key]["displacement"])  # breaks symmetry
            return atoms

        if len(start_structures) == 2:
            # Intermolecular reactions require in situ generation of the reactive complex
            s0 = start_structures[0]
            s1 = start_structures[1]

            # Get coordinates
            atoms1 = s0.get_atoms()
            atoms2 = s1.get_atoms()
            elements1 = atoms1.elements
            elements2 = atoms2.elements
            coordinates1 = atoms1.positions
            coordinates2 = atoms2.positions
            # Calculate reactive center mean position
            if self.exploration_key + "_lhs_list" in self.settings[self.exploration_key]:
                sites1 = self.settings[self.exploration_key][self.exploration_key + "_lhs_list"]
                sites2 = self.settings[self.exploration_key][self.exploration_key + "_rhs_list"]
                self.settings[self.exploration_key][self.exploration_key + "_rhs_list"] = list(
                    idx + len(elements1) for idx in sites2
                )
            elif "nt_associations" in self.settings[self.exploration_key]:
                sites1 = []
                sites2 = []
                nAtoms1 = len(atoms1.elements)
                for i in range(0, len(self.settings[self.exploration_key]["nt_associations"]), 2):
                    at1 = self.settings[self.exploration_key]["nt_associations"][i]
                    at2 = self.settings[self.exploration_key]["nt_associations"][i + 1]
                    if at1 >= nAtoms1 > at2:
                        sites1.append(at2)
                        sites2.append(at1 - nAtoms1)
                    if at2 >= nAtoms1 > at1:
                        sites1.append(at1)
                        sites2.append(at2 - nAtoms1)
            else:
                self.raise_named_exception(
                    "Reactive complex can not be build: missing reactive atoms list(s)."
                )
            reactive_center1 = np.mean(coordinates1[sites1], axis=0)
            reactive_center2 = np.mean(coordinates2[sites2], axis=0)
            # Place reactive center mean position into origin
            coord1 = coordinates1 - reactive_center1
            coord2 = coordinates2 - reactive_center2
            positions = self._orient_coordinates(coord1, coord2)
            atoms = utils.AtomCollection(elements1 + elements2, positions)
            self.random_displace_atoms(atoms, self.settings[self.rc_key]["displacement"])  # breaks symmetry
            return atoms

        if len(start_structures) == 3:
            # the given reaction has 3 structure, with one already representing the reactive complex
            for s in start_structures:
                if s.get_label() == db.Label.SURFACE_ADSORPTION_GUESS:
                    return s.get_atoms()

        # should not be reachable
        self.raise_named_exception(
            "Reactive complexes built from more than 2 structures are not supported."
        )

    def determine_pes_of_rc(self, settings_manager: SettingsManager, s0: db.Structure,
                            s1: Optional[db.Structure] = None) -> None:
        """
        Set charge and spin multiplicity within the settings_manager based on the reaction type (uni- vs. bimolecular)
        and the given settings for the reactive complex.

        Notes
        -----
        * Requires run configuration

        Parameters
        -------
        settings_manager :: SettingsManager
            The settings_manager in which the charge and multiplicity of the new atoms are set.
        s0 :: db.Structure (Scine::Database::Structure)
            A structure of the configured calculation
        s1 :: Union[db.Structure, None]
            A potential second structure for bimolecular reactions
        """
        from scine_utilities.settings_names import molecular_charge, spin_multiplicity

        if s1 is None:
            settings_manager.update_calculator_settings(
                s0, self._calculation.get_model(), self.config["resources"])
            if molecular_charge in self.settings[self.rc_key]:
                settings_manager.calculator_settings[molecular_charge] = self.settings[self.rc_key][molecular_charge]
            if spin_multiplicity in self.settings[self.rc_key]:
                settings_manager.calculator_settings[spin_multiplicity] = self.settings[self.rc_key][spin_multiplicity]
        else:
            settings_manager.update_calculator_settings(
                None, self._calculation.get_model(), self.config["resources"])

            # set defaults
            default_charge = s0.get_charge() + s1.get_charge()
            min_mult = abs(s0.get_multiplicity() - s1.get_multiplicity()) + 1  # max spin recombination
            max_mult = s0.get_multiplicity() + s1.get_multiplicity() - 1  # no spin recombination between molecules
            default_mult = min_mult if self.settings[self.rc_key]["minimal_spin_multiplicity"] else max_mult

            # pick values from settings, otherwise defaults
            charge = self.settings[self.rc_key].get(molecular_charge, default_charge)
            multiplicity = self.settings[self.rc_key].get(spin_multiplicity, default_mult)

            # set values in manager
            settings_manager.calculator_settings[molecular_charge] = charge
            settings_manager.calculator_settings[spin_multiplicity] = multiplicity

    def _orient_coordinates(self, coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
        """
        A mathematical helper function to align the respective coordinates based on the given settings.

        Parameters
        -------
        coord1 :: np.ndarray of shape (n,3)
            The coordinates of the first molecule
        coord2 :: np.ndarray of shape (m,3)
            The coordinates of the second molecule

        Returns
        -------
        coord :: np.ndarray of shape (n+m, 3)
            The combined and aligned coordinates of both molecules
        """
        rc_settings = self.settings[self.rc_key]
        # Rotate directions towards each other
        r = np.array(rc_settings["x_alignment_0"]).reshape((3, 3))
        coord1 = (r.T.dot(coord1.T)).T
        r = np.array(rc_settings["x_alignment_1"]).reshape((3, 3))
        coord2 = (r.T.dot(coord2.T)).T
        # Rotate around x-axis
        angle = rc_settings["x_rotation"]
        x_rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        coord2 = x_rot.dot(coord2.T).T
        # Displace coordinates of the molecules along the x-axis
        coord2 += np.array([rc_settings["x_spread"], 0.0, 0.0])
        coord1 -= np.array([rc_settings["x_spread"], 0.0, 0.0])
        return np.concatenate((coord1, coord2), axis=0)

    @staticmethod
    def random_displace_atoms(atoms: utils.AtomCollection, displacement: float = 0.05) -> None:
        """
        Apply small seeded random displacement based on setting
        """
        np.random.seed(42)
        coords = np.array([atoms.get_position(i) for i in range(len(atoms))])
        coords += displacement * (np.random.rand(*coords.shape) - 0.5) * 2.0 / np.sqrt(3.0)
        atoms.positions = coords

    def setup_automatic_mode_selection(self, name: str) -> None:
        """
        A settings sanity check, which adds the settings for automatic mode selection or doesn't based on the given
        user settings.

        Parameters
        -------
        name :: str
            The name of the subtask for which the automatic mode selection is added.
        """
        if "automatic_mode_selection" not in self.settings[name] and all(
                "_follow_mode" not in key for key in self.settings[name]
        ):
            if self.exploration_key + "_lhs_list" in self.settings[self.exploration_key]:
                self.settings[name]["automatic_mode_selection"] = (
                    self.settings[self.exploration_key][self.exploration_key + "_lhs_list"]
                    + self.settings[self.exploration_key][self.exploration_key + "_rhs_list"]
                )
            else:
                self.settings[name]["automatic_mode_selection"] = (
                    self.settings[self.exploration_key]["nt_associations"]
                    + self.settings[self.exploration_key]["nt_dissociations"]
                )

    def n_imag_frequencies(self, name: str) -> int:
        """
        A helper function to count the number of imaginary frequencies based on the threshold in the settings.
        Does not carry out safety checks.

        Parameters
        -------
        name :: str
            The name of the system which holds Hessian results.
        """
        atoms = self.systems[name].structure
        modes_container = utils.normal_modes.calculate(self.systems[name].get_results().hessian, atoms)
        wavenumbers = modes_container.get_wave_numbers()

        return np.count_nonzero(np.array(wavenumbers) < self.settings[self.job_key]["imaginary_wavenumber_threshold"])

    def get_graph_charges_multiplicities(self, name: str, total_charge: int, total_system_name: Optional[str] = None,
                                         split_index: Optional[int] = None) \
            -> Tuple[List[utils.AtomCollection], str, List[int], List[int], List[str]]:
        """
        Runs bond orders for the specified name in the dictionary of
        systems, constructs the resulting graphs and splits the system into
        the corresponding molecules. Computes the charges of each molecule
        from partial charges and the corresponding minimal spin.
        All resulting lists are sorted according to the graphs and, if these
        are equal, according to the charges.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        name :: str
            Index into systems dictionary to calculate bond orders for
        total_charge :: str
            The charge of the system
        total_system_name :: str
            The name of the total system which can be specified in case this method is called for a partial system.
            This can enable to assign the indices of the total system to the indices of the partial system.
        split_index :: int
            The index of the system in the total system which is split. This is used to assign the indices of the total
            system to the indices of the partial system. Both total_system_name and split_index must be specified or
            neither must be specified.

        Returns
        -------
        ordered_structures :: List[utils.AtomCollection]
            List of atom collections corresponding to the split molecules.
        graph_string :: str
            Sorted molassembler cbor graphs separated by semicolons.
        charges :: List[int]
            Charges of the molecules.
        multiplicities :: List[int]
            Multiplicities of the molecules, total multiplicity before split influences these returned values based
            on a buff spread over all split structures, these values have to be checked with spin propensity checks
        decision_lists :: List[str]
            Molassembler decision lists for free dihedrals
        """
        import scine_readuct as readuct
        from scine_puffin.utilities.reaction_transfer_helper import ReactionTransferHelper

        all_surface_indices = self.surface_indices_all_structures()
        if total_system_name is None:
            surface_indices: Union[Set[int], List[int]] = all_surface_indices
        elif total_system_name not in self._component_maps:
            self.raise_named_exception(f"Total system name '{total_system_name}' not found in component maps")
            return utils.AtomCollection(), "", [], [], []  # For type checking
        elif split_index is None:
            self.raise_named_exception(f"Split index must be given, "
                                       f"if total system name '{total_system_name}' is specified")
            return utils.AtomCollection(), "", [], [], []  # For type checking
        else:
            split_surfaces_indices = \
                ReactionTransferHelper.map_total_indices_to_split_structure_indices(
                    all_surface_indices, self._component_maps[total_system_name])
            surface_indices = split_surfaces_indices[split_index]

        masm_results, self.systems = self.make_masm_result_from_calc(self.systems, name, surface_indices)

        split_structures = masm_results.component_map.apply(self.systems[name].structure)
        decision_lists = [masm_helper.get_decision_list_from_molecule(m, a)
                          for m, a in zip(masm_results.molecules, split_structures)]

        # Get cbor graphs
        graphs = []
        for molecule in masm_results.molecules:
            graphs.append(masm_helper.get_cbor_graph_from_molecule(molecule))

        # Determine partial charges, charges per molecules and number of electrons per molecule
        bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, name, surface_indices)
        partial_charges = self.systems[name].get_results().atomic_charges
        if partial_charges is None:
            self.systems, success = readuct.run_single_point_task(
                self.systems, [name], require_charges=True
            )
            self.throw_if_not_successful(
                success, self.systems, [name], ["energy", "atomic_charges"]
            )
            partial_charges = self.systems[name].get_results().atomic_charges
            # TODO replace with propert setter if we have on in utils, this does not work
            self.systems[name].get_results().bond_orders = bond_orders

        charges, n_electrons, _ = self._integrate_charges(masm_results.component_map, partial_charges,
                                                          split_structures, total_charge)

        # Assign multiplicities where we try to spread the buff
        # (i.e. multiplicity difference before to singlet / duplet multiplicity)
        # --> if before 3 -> give one structure (largest) triplet, before 5 --> give each a triplet
        # this ensures that the spin propensity checks later can cover as much as possible
        # this should work with any multiplicity and any number of split structures
        multiplicity_before = self.systems[name].settings[utils.settings_names.spin_multiplicity]
        total_electrons_were_even = multiplicity_before % 2 != 0
        min_multiplicity = 1 if total_electrons_were_even else 2
        buff = (multiplicity_before - min_multiplicity) / 2.0
        n_structures = len(split_structures)
        if n_structures == 1:
            multiplicities = [multiplicity_before]
        elif not buff:
            multiplicities = [nel % 2 + 1 for nel in n_electrons]
        else:
            buff_per_structure = int(ceil(buff / n_structures))
            sorted_structures = sorted(split_structures, reverse=True)  # sort by size, the largest first
            # sort electrons just like structures
            sorted_n_electrons = [n for _, n in sorted(zip(sorted_structures, n_electrons), reverse=True)]
            # determine real index of the sorted electrons
            sorting_indices = [n for _, n in sorted(zip(sorted_structures, list(range(n_structures))), reverse=True)]
            multiplicities_array = np.zeros(n_structures, dtype=int)
            for index, nel in zip(sorting_indices, sorted_n_electrons):
                is_even = nel % 2 == 0
                multiplicity = 1 if is_even else 2
                if buff:
                    multiplicity += 2 * buff_per_structure
                    buff -= 1
                multiplicities_array[index] = multiplicity
            multiplicities = list(multiplicities_array)

        # Sort everything according to graphs and if these are equal according to charges and then multiplicities
        graphs, charges, multiplicities, decision_lists, structure_order = (
            list(start_val)
            for start_val in zip(*sorted(zip(
                graphs,
                charges,
                multiplicities,
                decision_lists,
                range(len(split_structures)))))
        )
        graph_string = ";".join(graphs)

        ordered_structures = [split_structures[i] for i in structure_order]
        new_component_map = [structure_order.index(i) for i in list(masm_results.component_map)]
        self._component_maps[name] = new_component_map

        return ordered_structures, graph_string, charges, multiplicities, decision_lists

    @staticmethod
    def _custom_round(number: float, threshold: float = 0.5) -> float:
        """
        Rounding number up or down depending on the threshold.
        To round down, delta must be smaller than the threshold.

        Parameters
        ----------
        number : float
            Number which should be rounded.
        threshold : float, optional
            Threshold when to round up, by default 0.5

        Returns
        -------
        float
            Number rounded according to threshold.
        """
        sign = np.copysign(1.0, number)
        number = abs(number)
        delta = number - np.trunc(number)
        if delta < threshold:
            return np.trunc(number) * sign
        else:
            return (np.trunc(number) + 1) * sign

    @staticmethod
    def _calculate_residual(original_values: List[Any], new_values: List[Any]) -> List[float]:
        """
        Calculate the residual where one subtracts new from old values.

        Parameters
        ----------
        original_values : List[float]
            A list of old values.
        new_values : List[float]
            A list of new values.

        Returns
        -------
        residual : List[float]
            The list of differences between old and new.
        """
        residual = []
        for i in range(len(original_values)):
            residual.append(original_values[i] - new_values[i])
        return residual

    def _distribute_charge(
            self,
            total_charge: float,
            charge_guess,
            summed_partial_charges: List[float]) -> List[int]:
        """
        Check if the sum of the charges of the non-bonded molecules equals the total charge of the supersystem.
        If this should not be the case, add or remove one charge, depending on the difference between the total charge
        and the sum of the charge guess.
        A charge is added where the residual between the partial charges and the charge guess (partial - guess)
        is maximal and subtracted where it is minimal.
        The re-evaluated after the charge guess was modified.

        Parameters
        ----------
        total_charge : float
            Total charge of the supersystem.
        charge_guess : List[int]
            List of guessed charges for each molecule in the supersystem.
        summed_partial_charges : List[float]
            List of the sum over the partial charges of the non-bonded molecules in the supersystem.

        Returns
        -------
            charge_guess : List[float]
                The updated list of guessed charges where the sum equals the total charge of the supersystem.
        """
        residual = self._calculate_residual(summed_partial_charges, charge_guess)
        while sum(charge_guess) != total_charge:
            charge_diff = sum(charge_guess) - total_charge
            # too many electrons, add a charge
            if charge_diff < 0.0:
                # Add one charge to selection
                charge_guess[np.argmax(residual)] += 1
            # too little electrons, remove a charge
            else:
                # Substract one charge from selection
                charge_guess[np.argmin(residual)] -= 1
            # Update residual
            residual = self._calculate_residual(summed_partial_charges, charge_guess)
        # return updated charge guess
        return charge_guess

    def _integrate_charges(self, component_map: List[int], partial_charges: List[float],
                           split_structures, total_charge: float) -> Tuple[List[int], List[int], List[float]]:
        """
        Determine the charges, the number of electrons and the residual to the partial charges per molecule of
        the non-bonded molecules in the supersystem.

        Parameters
        ----------
        component_map : List[int]
            List of indices to map atoms to the molecule it belongs according to molassembler.
        partial_charges : List[float]
            The partial charges of the atoms.
        split_structures : List[utils.AtomCollection]
            The non-bonded molecules in the supersystem.
        total_charge : List[utils.AtomCollection]
            The total charge of the supersystem.

        Returns
        -------
        charges : List[int]
            The charges for each non-bonded molecule in the supersystem.
        n_electrons : List[int]
            The number of electrons for each non-bonded molecule in the supersystem.
        residual : List[float]
            The difference between the original sum of partial charges and
            the determined charges per non-bonded molecule in the supersystem.

        """
        charges = []
        n_electrons = []
        for i in range(len(split_structures)):
            charges.append(0.0)
        for i, c in zip(component_map, partial_charges):
            charges[i] += c
        summed_partial_charges = deepcopy(charges)
        print("Charge separation check " + str(self.settings[self.single_point_key]["expect_charge_separation"]))
        # Update charges to charge guess, only containing ints
        for i in range(len(split_structures)):
            if not self.settings[self.single_point_key]["expect_charge_separation"]:
                charges[i] = int(self._custom_round(charges[i], 0.5))
            else:
                charges[i] = int(self._custom_round(charges[i],
                                 self.settings[self.single_point_key]["charge_separation_threshold"]))

        # Check and re-distribute if necessary
        updated_charges = self._distribute_charge(total_charge, charges, summed_partial_charges)
        # Update number of electrons
        for i in range(len(split_structures)):
            electrons = 0
            for elem in split_structures[i].elements:
                electrons += utils.ElementInfo.Z(elem)
            electrons -= updated_charges[i]
            n_electrons.append(int(round(electrons)))
        residual = self._calculate_residual(summed_partial_charges, updated_charges)
        return updated_charges, n_electrons, residual

    def check_for_barrierless_reaction(self) -> Union[Tuple[str, List[str]], Tuple[None, None]]:
        """
        Optimizes the reactive complex, comparing the result to the start
        structures determining if a barrierless reaction occurred.

        Returns
        -------
        rc_opt_graph :: Optional[str]
            Sorted molassembler cbor graphs separated by semicolons of the
            reaction product if there was any.
        rc_opt_decision_lists :: Optional[List[str]]
            Molassembler decision lists for free dihedrals of the reaction
            product if there was any.
        """
        import scine_molassembler as masm
        # Check for barrierless reaction leading to new graphs
        if self.rc_opt_system_name not in self.systems:  # Skip if already done
            print("Running Reactive Complex Optimization")
            print("Settings:")
            print(self.settings[self.rc_opt_system_name], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_opt_task', self.systems, [self.rc_key], **self.settings[self.rc_opt_system_name]
            )
            self.throw_if_not_successful(
                success,
                self.systems,
                [self.rc_opt_system_name],
                [],
                "Reactive complex optimization failed.\n",
            )
        _, rc_opt_graph, _, _, rc_opt_decision_lists = \
            self.get_graph_charges_multiplicities(self.rc_opt_system_name, sum(self.start_charges))

        print("Optimized Reactive Complex Graph:")
        print(rc_opt_graph)

        if not masm.JsonSerialization.equal_molecules(self.start_graph, rc_opt_graph):
            return rc_opt_graph, rc_opt_decision_lists
        return None, None

    def output(self, name: str) -> List[str]:
        """
        A helper function fetching the output entry from the system specified with the given name to ease task
        chaining and ensure correct chaining even if user changes the output setting.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        name :: str
            Index into systems dictionary to retrieve output for

        Returns
        -------
        outputs :: List[str]
            A list of output system names
        """
        if name not in self.settings:
            self.raise_named_exception(
                "The system "
                + name
                + " is not present in the settings of the job "
                + self.name
            )
        if "output" not in self.settings[name]:
            self.raise_named_exception(
                "The settings for "
                + name
                + " in the Job "
                + self.name
                + "do not include an output specification"
            )
        return self.settings[name]["output"]

    def irc_sanity_checks_and_analyze_sides(
            self,
            initial_charge: int,
            check_charges: bool,
            inputs: List[str],
            calculator_settings: dict) -> Union[Tuple[List[str], Optional[List[str]]], Tuple[None, None]]:
        """
        Check whether we found a new structure, whether our IRC matches the start
        (and end in case of double ended). This decision is made based on optimized
        structures of the separated molecules taken from IRC endpoints.
        Both starting structures and products are assigned, starting structure names
        are returned only if they do not match the conformers of the initial input.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        initial_charge :: int
            The charge of the reactive complex
        check_charges :: bool
            Whether the charges must be checked
        inputs :: List[str]
            The name of the IRC outputs to use as inputs
        calculator_settings :: dict
            The general settings for the Scine calculator. Charge and spin multiplicity will be overwritten.

        Returns
        -------
        product_names :: Optional[List[str]]
            A list of the access keys to the products in the system map.
        start_names :: Optional[List[str]]
            A list of the access keys to the starting materials in the system map.
        """
        import scine_molassembler as masm
        if len(inputs) != 2:
            self.raise_named_exception(
                "Requires to pass 2 systems to the IRC sanity check"
            )
        # All lists ordered according to graph - charges - multiplicities with decreasing priority
        # Get graphs, charges and minimal multiplicities of split forward and backward structures
        print("Forward Bond Orders")
        (
            forward_structures,
            forward_graph,
            forward_charges,
            forward_multiplicities,
            forward_decision_lists,
        ) = self.get_graph_charges_multiplicities(inputs[0], initial_charge)
        print("Backward Bond Orders")
        (
            backward_structures,
            backward_graph,
            backward_charges,
            backward_multiplicities,
            backward_decision_lists,
        ) = self.get_graph_charges_multiplicities(inputs[1], initial_charge)

        print("Forward charges: " + str(forward_charges))
        print("Backward charges: " + str(backward_charges))

        # Optimize separated forward molecules
        forward_names = self.optimize_structures("forward", forward_structures,
                                                 forward_charges,
                                                 forward_multiplicities,
                                                 calculator_settings)
        # Optimize separated backward molecules
        backward_names = self.optimize_structures("backward", backward_structures,
                                                  backward_charges,
                                                  backward_multiplicities,
                                                  calculator_settings)
        # Analyze separated forward molecules
        forward_graphs = []
        forward_decision_lists = []
        for i, (name, charge) in enumerate(zip(forward_names, forward_charges)):
            s, g, _, _, d = self.get_graph_charges_multiplicities(name, charge,
                                                                  total_system_name=inputs[0], split_index=i)
            if len(s) > 1:
                self._calculation.set_comment(self.name + ": IRC results keep decomposing (more than once).")
                return None, None
            forward_graphs += g.split(';')
            forward_decision_lists += d
        # Sort everything again
        forward_graphs, forward_charges, forward_multiplicities, forward_decision_lists, \
            forward_names = (
                list(start_val) for start_val in zip(*sorted(zip(
                    forward_graphs,
                    forward_charges,
                    forward_multiplicities,
                    forward_decision_lists,
                    forward_names)))
            )
        forward_graph = ';'.join(forward_graphs)
        # Analyze separated backward molecules
        backward_graphs = []
        backward_decision_lists = []
        for i, (name, charge) in enumerate(zip(backward_names, backward_charges)):
            s, g, _, _, d = self.get_graph_charges_multiplicities(name, charge,
                                                                  total_system_name=inputs[1], split_index=i)
            if len(s) > 1:
                self._calculation.set_comment(self.name + ": IRC results keep decomposing (more than once).")
                return None, None
            backward_graphs += g.split(';')
            backward_decision_lists += d
        # Sort everything again
        backward_graphs, backward_charges, backward_multiplicities, backward_decision_lists, \
            backward_names = (
                list(start_val) for start_val in zip(*sorted(zip(
                    backward_graphs,
                    backward_charges,
                    backward_multiplicities,
                    backward_decision_lists,
                    backward_names)))
            )
        backward_graph = ';'.join(backward_graphs)

        # Check for new structures and compare IRC to Start
        print("Start Graph:")
        print(self.start_graph)  # Equals reactive complex graph
        if self.end_graph:
            print("End Graph:")
            print(self.end_graph)  # for double ended method
        print("Forward Graph:")
        print(forward_graph)
        print("Backward Graph:")
        print(backward_graph)

        # TODO Check whether multiplicities of split result structures fit with settings
        found_new_structures = bool(not masm.JsonSerialization.equal_molecules(forward_graph, backward_graph)
                                    or forward_charges != backward_charges)
        if not found_new_structures:
            self._calculation.set_comment(self.name + ": IRC forward and backward have identical structures.")
            self._save_ts_for_restart(db.Label.TS_OPTIMIZED)
            return None, None

        compare_decision_lists = True
        # Do not expect matching charges if reactive complex charge differs from sum of start structure charges
        if masm.JsonSerialization.equal_molecules(forward_graph, self.start_graph) \
                and (not check_charges or forward_charges == self.start_charges):
            product_names = backward_names
            self.step_direction = "backward"
            self.products_component_map = self._component_maps[inputs[1]]
            compare_decision_lists = False
        elif masm.JsonSerialization.equal_molecules(backward_graph, self.start_graph) and (
                not check_charges or backward_charges == self.start_charges
        ):
            product_names = forward_names
            self.step_direction = "forward"
            self.products_component_map = self._component_maps[inputs[0]]
        elif ';' in self.start_graph:
            rc_opt_graph, _ = self.check_for_barrierless_reaction()
            print("Barrierless Check Graph:")
            print(rc_opt_graph)
            if rc_opt_graph is None:
                print(self.name + ": No IRC structure matches starting structure.")
                product_names = forward_names
                # Step direction must be forward to guarantee working logic downstream
                self.step_direction = "forward"
                self.products_component_map = self._component_maps[inputs[0]]
                # Trigger to set 'start_names' as 'backward_names'
                compare_decision_lists = False
                self.no_irc_structure_matches_start = True
            elif masm.JsonSerialization.equal_molecules(forward_graph, rc_opt_graph):
                self.step_direction = "backward"
                product_names = backward_names
                self.products_component_map = self._component_maps[inputs[1]]
                self.lhs_barrierless_reaction = True
            elif masm.JsonSerialization.equal_molecules(backward_graph, rc_opt_graph):
                self.step_direction = "forward"
                product_names = forward_names
                self.products_component_map = self._component_maps[inputs[0]]
                self.lhs_barrierless_reaction = True
            else:
                print(self.name + ": No IRC structure matches starting structure.")
                product_names = forward_names
                # Step direction must be forward to guarantee working logic downstream
                self.step_direction = "forward"
                self.products_component_map = self._component_maps[inputs[0]]
                # Trigger to set 'start_names' as 'backward_names'
                compare_decision_lists = False
                self.no_irc_structure_matches_start = True
        else:
            print(self.name + ": No IRC structure matches starting structure.")
            product_names = forward_names
            # Step direction must be forward to guarantee working logic downstream
            self.step_direction = "forward"
            self.products_component_map = self._component_maps[inputs[0]]
            # Trigger to set 'start_names' as 'backward_names'
            compare_decision_lists = False
            self.no_irc_structure_matches_start = True

        if not compare_decision_lists:
            # ensures that we save the start structures
            decision_lists_match = False
        else:
            # Compare decision lists of start structures:
            original_decision_lists = self.start_decision_lists
            if self.step_direction == "backward":
                new_decision_lists = forward_decision_lists
            else:
                new_decision_lists = backward_decision_lists
            decision_lists_match = True
            for new, orig in zip(new_decision_lists, original_decision_lists):
                if not masm.JsonSerialization.equal_decision_lists(new, orig):
                    decision_lists_match = False
                    break
        if not decision_lists_match:
            if self.step_direction == "backward":
                start_names = forward_names
            else:
                # Important, if no_irc_structure_matches_start!
                start_names = backward_names
        else:
            start_names = None
        # additional check for double ended methods
        if self.end_graph:
            if (
                    masm.JsonSerialization.equal_molecules(forward_graph, self.start_graph)
                    and masm.JsonSerialization.equal_molecules(backward_graph, self.end_graph)
                    and (not check_charges or forward_charges == self.start_charges)
            ):
                product_names = backward_names
            elif (
                    masm.JsonSerialization.equal_molecules(forward_graph, self.end_graph)
                    and masm.JsonSerialization.equal_molecules(backward_graph, self.start_graph)
                    and (not check_charges or backward_charges == self.start_charges)
            ):
                product_names = forward_names
            else:
                print(self.name + ": IRC does not match double-ended method")
                # IRC points do not match end points of double ended method,
                # hence the IRC points are forwarded for post-processing.
                product_names = backward_names
                start_names = forward_names
                self.step_direction = "backward"
        # Check if complexations need to be tracked
        forward_complexation_energy = 0.0
        for name in forward_names:
            forward_complexation_energy -= self.systems[name].get_results().energy
        forward_complexation_energy += self.systems[inputs[0]].get_results().energy
        if forward_complexation_energy < self.complexation_criterion:
            if self.step_direction == "backward":
                self.lhs_complexation = True
            else:
                self.rhs_complexation = True
        backward_complexation_energy = 0.0
        for name in backward_names:
            backward_complexation_energy -= self.systems[name].get_results().energy
        backward_complexation_energy += self.systems[inputs[1]].get_results().energy
        if backward_complexation_energy < self.complexation_criterion:
            if self.step_direction == "backward":
                self.rhs_complexation = True
            else:
                self.lhs_complexation = True

        return product_names, start_names

    def optimize_structures(
            self,
            name_stub: str,
            structures: List[utils.AtomCollection],
            structure_charges: List[int],
            structure_multiplicities: List[int],
            calculator_settings: dict,
            stop_on_error: bool = True
    ) -> List[str]:
        """
        For each given product AtomCollection:
        First, construct a Scine Calculator and save in class member map.
        Second, perform a Single Point with the given charge and spin multiplicity including spin propensity check
        Last, optimize the product if more than one atom and perform spin propensity check again to be sure.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        name_stub :: str
            The stub for naming of the structures, example: `start` will generate
            systems `start_00`, `start_01`, and so on.
        structures :: List[utils.AtomCollection]
            The atoms of the structures in a list.
        structure_charges :: List[int]
            The charges of the structures.
        structure_multiplicities :: List[int]
            The spin multiplicities of the structures.
        calculator_settings :: dict
            The general settings for the Scine calculator. Charge and spin multiplicity will be overwritten.
        stop_on_error :: bool
            If set to False, skip unsuccessful calculations and replace calculator with None

        Returns
        -------
        product_names :: List[str]
            A list of the access keys to the structures in the system map.
        """
        import scine_readuct as readuct
        structure_names = []
        method_family = self._calculation.get_model().method_family
        # Generate structure systems
        for i, structure in enumerate(structures):
            name = f"{name_stub}_{i:02d}"
            structure_names.append(name)
            utils.io.write(name + ".xyz", structure)
            try:
                # correct PES
                structure_calculator_settings = deepcopy(calculator_settings)
                structure_calculator_settings[utils.settings_names.molecular_charge] = structure_charges[i]
                structure_calculator_settings[utils.settings_names.spin_multiplicity] = structure_multiplicities[i]
                # generate calculator
                new = utils.core.load_system_into_calculator(
                    name + ".xyz",
                    method_family,
                    **structure_calculator_settings,
                )
                self.systems[name] = new
                self._add_propensity_systems(name)
            except RuntimeError as e:
                if stop_on_error:
                    raise e
                sys.stderr.write(f"{name} cannot be calculated because: {str(e)}")
                self.systems[name] = None

        print("Product Opt Settings:")
        print(self.settings["opt"], "\n")
        required_properties = ["energy"]
        if not self.connectivity_settings['only_distance_connectivity']:
            required_properties.append("bond_orders")
        # Optimize structures, if they have more than one atom; otherwise just run a single point calculation
        for structure in structure_names:
            if self.systems[structure] is None:
                continue
            try:
                if not self.settings[self.job_key]["spin_propensity_check"]:
                    self.systems, success = readuct.run_single_point_task(
                        self.systems,
                        [structure],
                        require_bond_orders=not self.connectivity_settings['only_distance_connectivity'],
                    )
                    self.throw_if_not_successful(success, self.systems, [structure], required_properties,
                                                 f"{name_stub.capitalize()} single point failed:\n")
                else:
                    self._spin_propensity_single_points(structure, f"{name_stub.capitalize()} single point failed:\n")
                if len(self.systems[structure].structure) > 1:
                    if len(structure_names) == 1 and len(self._calculation.get_structures()) == 1 and \
                            not self.settings[self.job_key]["spin_propensity_check_for_unimolecular_reaction"]:
                        # optimize only base multiplicity
                        self.systems = self.observed_readuct_call_with_throw(
                            'run_opt_task', self.systems, [structure], required_properties,
                            f"{name_stub.capitalize()} optimization failed:\n", **self.settings["opt"]
                        )
                        # still do propensity SP to store close energy multiplicities in DB
                        self._spin_propensity_single_points(structure,
                                                            f"{name_stub.capitalize()} optimization failed:\n")
                    elif not self.settings[self.job_key]["spin_propensity_optimize_all"]:
                        prev_lowest = None
                        lowest_name, _ = self._get_propensity_names_within_range(
                            structure, self.settings[self.job_key]["spin_propensity_energy_range_to_optimize"]
                        )
                        while lowest_name != prev_lowest:
                            print("Optimizing " + lowest_name + ":\n")
                            self.systems = self.observed_readuct_call_with_throw(
                                'run_opt_task', self.systems, [lowest_name], required_properties,
                                f"{name_stub.capitalize()} optimization failed:\n", **self.settings["opt"]
                            )
                            self._spin_propensity_single_points(structure,
                                                                f"{name_stub.capitalize()} optimization failed:\n")
                            lowest_name, _ = self._get_propensity_names_within_range(
                                structure, self.settings[self.job_key]["spin_propensity_energy_range_to_optimize"]
                            )
                    else:
                        self._spin_propensity_optimizations(structure,
                                                            f"{name_stub.capitalize()} optimization failed:\n")
            except RuntimeError as e:
                if stop_on_error:
                    raise e
                sys.stderr.write(f"{structure} cannot be calculated because: {str(e)}")
                self.systems[structure] = None
        return structure_names

    def _add_propensity_systems(self, name: str) -> None:
        for shift_name, multiplicity in self._propensity_iterator(name):
            if shift_name == name:
                continue
            self.systems[shift_name] = self.systems[name].clone()
            self.systems[shift_name].delete_results()  # make sure results of clone are empty
            if utils.settings_names.spin_mode in self.systems[shift_name].settings:
                dc = self.systems[shift_name].settings.descriptor_collection
                if isinstance(dc[utils.settings_names.spin_mode],
                              utils.OptionListDescriptor):
                    for suitable in ["unrestricted", "restricted_open_shell", "any"]:
                        if suitable in dc[utils.settings_names.spin_mode].options:
                            self.systems[shift_name].settings[utils.settings_names.spin_mode] = suitable
                            break
                else:
                    self.systems[shift_name].settings[utils.settings_names.spin_mode] = "any"
            self.systems[shift_name].settings[utils.settings_names.spin_multiplicity] = multiplicity

    def _propensity_iterator(self, name: str) -> Iterator[Tuple[str, int]]:
        from scine_utilities import settings_names

        propensity_limit = self.settings[self.job_key]["spin_propensity_check"]
        for shift in range(-propensity_limit, propensity_limit + 1):
            multiplicity = self.systems[name].settings[settings_names.spin_multiplicity] + shift * 2
            if multiplicity > 0:
                shift_name = f"{name}_multiplicity_shift_{shift}" if shift else name
                yield shift_name, multiplicity

    def _spin_propensity_single_points(self, name: str, error_msg: str) -> None:
        import scine_readuct as readuct
        info = f"Single point calculations of {name}"
        if self.settings[self.job_key]["spin_propensity_check"]:
            info += " with potential spin propensities"
        info += ":\n"
        print(info)
        total_success = 0
        for shift_name, _ in self._propensity_iterator(name):
            if self.systems.get(shift_name) is None:
                continue
            if self.systems[shift_name].get_results().energy is not None:
                # we already have an energy for this system
                total_success += 1
                continue
            self.systems, success = readuct.run_single_point_task(
                self.systems,
                [shift_name],
                require_bond_orders=not self.connectivity_settings['only_distance_connectivity'],
                stop_on_error=False
            )
            if success:
                total_success += 1
            else:
                self.systems[shift_name] = None
        if not total_success:
            self.throw_if_not_successful(False, self.systems, [name], ["energy"], error_msg)

    def _spin_propensity_optimizations(self, name: str, error_msg: str) -> None:
        info = f"Optimizing {name}"
        if self.settings[self.job_key]["spin_propensity_check"]:
            info += " with potential spin propensities"
        info += ":\n"
        print(info)
        total_success = 0
        lowest_name, allowed_names = self._get_propensity_names_within_range(
            name,
            self.settings[self.job_key]["spin_propensity_energy_range_to_optimize"]
        )
        all_names = [lowest_name] + allowed_names
        for shift_name, _ in self._propensity_iterator(name):
            if self.systems.get(shift_name) is None or shift_name not in all_names:
                continue
            self.systems, success = self.observed_readuct_call(
                'run_opt_task', self.systems, [shift_name], stop_on_error=False, **self.settings["opt"]
            )
            if success:
                total_success += 1
            else:
                self.systems[shift_name] = None
        if not total_success:
            self.throw_if_not_successful(False, self.systems, [name], ["energy"], error_msg)

    def _save_ts_for_restart(self, ts_label: db.Label) -> None:
        """
        Saves the output system of 'tsopt' (hence must already be finished)
        as a restart information after some additional single points.

        Notes
        -----
        * Requires run configuration
        """
        ts_name = self.output("tsopt")[0]
        # do propensity single_points for TS and save data
        _, ts = self._store_ts_with_propensity_info(ts_name, None, ts_label)
        self._calculation.set_restart_information("TS", ts.id())

    def generate_spline(
            self, tsopt_task_name: str, n_fit_points: int = 23, degree: int = 3
    ):
        """
        Using the transition state, IRC and IRC optimization outputs generates
        a spline that describes the trajectory of the elementary step, fitting
        both atom positions and energy.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        tsopt_task_name :: str
            Name of the transition state task.
        n_fit_points :: str
            Number of fit points to use in the spline compression.
        degree :: str
            Fit degree to use in the spline generation.

        Returns
        -------
        spline :: utils.bsplines.TrajectorySpline
            The fitted spline of the elementary step trajectory.
        """
        rpi = utils.bsplines.ReactionProfileInterpolation()

        def read_trj(fname):
            trj = utils.io.read_trajectory(utils.io.TrajectoryFormat.Xyz, fname)
            energies = []
            with open(fname, "r") as f:
                lines = f.readlines()
                nAtoms = int(lines[0].strip())
                i = 0
                while i < len(lines):
                    energies.append(float(lines[i + 1].strip()))
                    i += nAtoms + 2
            return trj, energies

        if self.step_direction == "forward":
            dir = "forward"
            rev_dir = "backward"
        elif self.step_direction == "backward":
            dir = "backward"
            rev_dir = "forward"
        else:
            self.raise_named_exception("Could not determine elementary step direction.")

        ts_calc = self.systems[self.output(tsopt_task_name)[0]]
        ts_energy = ts_calc.get_results().energy

        fpath = os.path.join(
            self.work_dir, f"irc_{rev_dir}", f"irc_{rev_dir}.opt.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(reversed(trj), reversed(energies)):
                if e > ts_energy:
                    continue
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)

        fpath = os.path.join(
            self.work_dir, f"irc_{rev_dir}", f"irc_{rev_dir}.irc.{rev_dir}.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(reversed(trj), reversed(energies)):
                if e > ts_energy:
                    continue
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)
        else:
            raise RuntimeError(
                f"Missing IRC trajectory file: irc_{rev_dir}/irc_{rev_dir}.irc.{rev_dir}.trj.xyz"
            )

        fpath = os.path.join(self.work_dir, "ts", "ts.xyz")
        if os.path.isfile(fpath):
            ts_calc = self.systems[self.output(tsopt_task_name)[0]]
            results = ts_calc.get_results()
            ts_xyz, _ = utils.io.read(fpath)
            rpi.append_structure(ts_xyz, results.energy, True)
        else:
            raise RuntimeError("Missing TS structure file: ts/ts.xyz")

        fpath = os.path.join(
            self.work_dir, f"irc_{dir}", f"irc_{dir}.irc.{dir}.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(trj, energies):
                if e > ts_energy:
                    continue
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)
        else:
            raise RuntimeError(
                f"Missing IRC trajectory file: irc_{dir}/irc_{dir}.irc.{dir}.trj.xyz"
            )

        fpath = os.path.join(self.work_dir, f"irc_{dir}", f"irc_{dir}.opt.trj.xyz")
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(trj, energies):
                if e > ts_energy:
                    continue
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)

        # Get spline
        spline = rpi.spline(n_fit_points, degree)
        return spline

    def store_start_structures(
            self,
            start_structure_names: List[str],
            program_helper: Union[ProgramHelper, None],
            tsopt_task_name: str,
            start_structures: Optional[List[db.ID]] = None
    ):
        """
        Store the new start systems in the database.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        start_structure_names :: List[str]
            The names of the start structure names in the system map.
        program_helper :: Union[ProgramHelper, None]
            The ProgramHelper which might also want to do postprocessing
        tsopt_task_name :: str
            The name of the task where the TS was output
        start_structures :: Optional[List[db.ID]]
            Optional list of the starting structure ids. If no list is given. The input
            structures of the calculation are used.

        Returns
        -------
        start_structure_ids :: List[scine_database.ID]
            A list of the database IDs of the start structures.
        """
        import scine_molassembler as masm
        from scine_puffin.utilities.reaction_transfer_helper import ReactionTransferHelper

        if start_structures is None:
            start_structures = self._calculation.get_structures()
        # get start name
        if self.step_direction == "forward":
            start_name = self.output("irc")[1]
        elif self.step_direction == "backward":
            start_name = self.output("irc")[0]
        else:
            self.raise_named_exception("Could not determine elementary step direction.")
            return  # unreachable, just for linter
        if start_name not in self._component_maps:
            self.raise_named_exception("Could not find component map for start structures.")
            return  # unreachable, just for linter

        # check for surface indices
        all_indices = self.surface_indices_all_structures(start_structures)
        split_surfaces_indices = \
            ReactionTransferHelper.map_total_indices_to_split_structure_indices(
                all_indices, self._component_maps[start_name])
        models = [db.Structure(sid, self._structures).get_model()
                  for sid in start_structures]
        start_model = models[0]
        if not all(model == start_model for model in models):
            self.raise_named_exception("React job with mixed model input structures")

        # Update model to make sure there are no 'any' values left
        update_model(
            self.systems[self.output(tsopt_task_name)[0]],
            self._calculation,
            self.config,
        )

        start_structure_ids = []
        for i, name in enumerate(start_structure_names):
            surface_indices = split_surfaces_indices[i] if split_surfaces_indices is not None else None
            # Check if the new structures are actually duplicates
            duplicate: Optional[db.ID] = None
            dl = ';'.join(self.make_decision_lists_from_calc(self.systems, name, surface_indices)[0])
            graph, self.systems = self.make_graph_from_calc(self.systems, name, surface_indices)
            for initial_id in start_structures:
                initial_structure = db.Structure(initial_id, self._structures)
                if not initial_structure.has_graph('masm_cbor_graph'):
                    continue
                initial_graph = initial_structure.get_graph("masm_cbor_graph")
                if not masm.JsonSerialization.equal_molecules(initial_graph, graph):
                    continue
                aggregate_id = initial_structure.get_aggregate()
                if ';' in initial_graph:
                    aggregate = db.Flask(aggregate_id)
                    aggregate.link(self._flasks)
                else:
                    aggregate = db.Compound(aggregate_id)
                    aggregate.link(self._compounds)
                existing_structures = aggregate.get_structures()
                for existing_structure_id in existing_structures:
                    existing_structure = db.Structure(existing_structure_id, self._structures)
                    if existing_structure.get_label() in \
                            [db.Label.DUPLICATE, db.Label.MINIMUM_GUESS, db.Label.USER_GUESS,
                             db.Label.SURFACE_GUESS, db.Label.SURFACE_ADSORPTION_GUESS]:
                        continue
                    if existing_structure.get_model() != start_model:
                        continue
                    existing_structure_dl = existing_structure.get_graph("masm_decision_list")
                    if masm.JsonSerialization.equal_decision_lists(dl, existing_structure_dl):
                        duplicate = existing_structure_id
                        break
                if duplicate is not None:
                    break
            if duplicate is not None:
                start_structure_ids.append(duplicate)
                continue

            label = self._determine_new_label_based_on_graph_and_surface_indices(graph, surface_indices)
            new_structure = self.create_new_structure(self.systems[name], label)
            for initial_id in start_structures:
                initial_structure = db.Structure(initial_id, self._structures)
                if not initial_structure.has_graph('masm_cbor_graph'):
                    continue
                if initial_structure.get_model() != new_structure.get_model():
                    continue
                initial_graph = initial_structure.get_graph("masm_cbor_graph")
                if masm.JsonSerialization.equal_molecules(initial_graph, graph):
                    self.transfer_properties(initial_structure, new_structure)
                    if program_helper is not None:
                        program_helper.calculation_postprocessing(self._calculation, initial_structure, new_structure)
            bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, name, surface_indices)
            self.store_energy(self.systems[name], new_structure)
            self.store_bond_orders(bond_orders, new_structure)
            self.add_graph(new_structure, bond_orders, surface_indices)
            start_structure_ids.append(new_structure.id())
        return start_structure_ids

    def save_barrierless_reaction_from_rcopt(self, product_graph: str, program_helper: Optional[ProgramHelper]) -> None:
        self.lhs_barrierless_reaction = True
        print("Barrierless product Graph:")
        print(product_graph)
        print("Start Graph:")
        print(self.start_graph)
        print("Barrierless Reaction Found")
        db_results = self._calculation.get_results()
        db_results.clear()
        # Save RHS of barrierless step
        rhs_complex_id = self._save_complex_to_db(self.rc_opt_system_name, program_helper)
        db_results.add_structure(rhs_complex_id)
        # Save step
        new_step = db.ElementaryStep(db.ID(), self._elementary_steps)
        new_step.create(self._calculation.get_structures(), [rhs_complex_id])
        new_step.set_type(db.ElementaryStepType.BARRIERLESS)
        db_results.add_elementary_step(new_step.id())
        self._calculation.set_comment(self.name + ": Barrierless reaction found.")
        self._calculation.set_results(self._calculation.get_results() + db_results)

    def _save_complex_to_db(self, complex_name: str, program_helper: Optional[ProgramHelper]) -> db.ID:
        """
        Saves structure with given name in systems map as a new structure in the database together with
        energy, bond orders, and graph.
        The label is determined based on the generated graph. Both of which rely on the fact that the given complex
        is the supersystem of all start structures.
        See `_determine_new_label_based_on_graph` for more details.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        complex_name :: str
            The name of the complex system in the systems map
        program_helper :: Union[ProgramHelper, None]
            The ProgramHelper which might also want to do postprocessing
        Returns
        -------
        complex_structure_id :: db.ID
            The id of the added structure
        """
        complex_system = self.systems[complex_name]
        complex_graph, self.systems = self.make_graph_from_calc(self.systems, complex_name)
        structure_label = self._determine_new_label_based_on_graph(complex_system, complex_graph)
        complex_structure = self.create_new_structure(complex_system, structure_label)
        bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, complex_name)
        self.transfer_properties(self.ref_structure, complex_structure)
        self.store_energy(self.systems[complex_name], complex_structure)
        self.store_bond_orders(bond_orders, complex_structure)
        self.add_graph(complex_structure, bond_orders)
        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, self.ref_structure, complex_structure)
        return complex_structure.id()

    def react_postprocessing(
        self,
        product_names: List[str],
        program_helper: Union[ProgramHelper, None],
        tsopt_task_name: str,
        reactant_structure_ids: List[db.ID]
    ) -> Tuple[List[db.ID], List[db.ID], db.ElementaryStep]:
        """
        Carries out a verification protocol after the calculation context has been closed, clears database result
        and then fills it with the found products, TS, and elementary step and all properties that can be associated
        with those.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        product_names :: List[str]
            A list of the access keys to the products in the system map.
        program_helper :: Union[ProgramHelper, None]
            The ProgramHelper which might also want to do postprocessing
        tsopt_task_name :: str
            The name of the task where the TS was output
        reactant_structure_ids :: List[scine_database.ID]
            A list of all structure IDs for the reactants.
        """
        from scine_puffin.utilities.reaction_transfer_helper import ReactionTransferHelper

        if not product_names:
            # should not be reachable
            self.raise_named_exception("Uncaught error in product calculation(s)")

        # clear existing results
        db_results = self._calculation.get_results()
        db_results.clear()
        self._calculation.set_results(db_results)

        # calculation is safe to be complete -> update model
        # do this with TS system, because we want a calculator that captures the whole system
        # and is safe to have a successful last calculation
        update_model(
            self.systems[self.output(tsopt_task_name)[0]],
            self._calculation,
            self.config,
        )

        """ Save products """
        new_labels = self._determine_product_labels_of_single_compounds(product_names)
        # check for surface indices
        assert self.products_component_map is not None
        all_indices = self.surface_indices_all_structures(self._calculation.get_structures())
        split_surfaces_indices = \
            ReactionTransferHelper.map_total_indices_to_split_structure_indices(
                all_indices, self.products_component_map)

        end_structures = []
        single_molecule_mode: bool = len(product_names) == 1 and len(self._calculation.get_structures()) == 1 and \
            not self.settings[self.job_key]["spin_propensity_check_for_unimolecular_reaction"]
        for i, (label, product) in enumerate(zip(new_labels, product_names)):
            surface_indices = split_surfaces_indices[i]
            new_structure = self._store_structure_with_propensity_check(product, label,
                                                                        enforce_to_save_base_name=single_molecule_mode,
                                                                        surface_indices=surface_indices)
            if program_helper is not None:
                program_helper.calculation_postprocessing(self._calculation, self.ref_structure, new_structure)
            end_structures.append(new_structure.id())
        """ transfer properties to products which requires to pass all structures"""
        transfer_helper = ReactionTransferHelper(self, self._properties)
        start_structures = [db.Structure(sid, self._structures) for sid in self._calculation.get_structures()
                            if db.Structure(sid, self._structures).get_label() != db.Label.SURFACE_ADSORPTION_GUESS]
        product_structures = [db.Structure(sid, self._structures) for sid in end_structures]
        transfer_helper.transfer_properties_between_multiple(start_structures, product_structures,
                                                             self.properties_to_transfer)
        """ Save TS """
        ts_name = self.output(tsopt_task_name)[0]
        # do propensity single_points for TS and save data
        ts_calc, new_ts = self._store_ts_with_propensity_info(ts_name, program_helper, db.Label.TS_OPTIMIZED)

        """ Save Complexes """
        if self.lhs_barrierless_reaction or self.lhs_complexation:
            if self.lhs_barrierless_reaction:
                lhs_complex_label = self.rc_opt_system_name
            elif self.step_direction == "forward":
                lhs_complex_label = "irc_backward"
            else:
                lhs_complex_label = "irc_forward"
            lhs_complex_id = self._save_complex_to_db(lhs_complex_label, program_helper)
            db_results.add_structure(lhs_complex_id)
        if self.rhs_complexation:
            rhs_complex_label = "irc_forward" if self.step_direction == "forward" else "irc_backward"
            rhs_complex_id = self._save_complex_to_db(rhs_complex_label, program_helper)
            db_results.add_structure(rhs_complex_id)

        """ Save Steps """
        main_step_lhs = [rsid for rsid in reactant_structure_ids
                         if db.Structure(rsid, self._structures).get_label() != db.Label.SURFACE_ADSORPTION_GUESS]
        main_step_rhs = end_structures
        if self.lhs_barrierless_reaction or self.lhs_complexation:
            new_step = db.ElementaryStep()
            new_step.link(self._elementary_steps)
            new_step.create(reactant_structure_ids, [lhs_complex_id])
            new_step.set_type(db.ElementaryStepType.BARRIERLESS)
            db_results.add_elementary_step(new_step.id())
            main_step_lhs = [lhs_complex_id]
        if self.rhs_complexation:
            new_step = db.ElementaryStep()
            new_step.link(self._elementary_steps)
            new_step.create([rhs_complex_id], end_structures)
            new_step.set_type(db.ElementaryStepType.BARRIERLESS)
            db_results.add_elementary_step(new_step.id())
            main_step_rhs = [rhs_complex_id]
        new_step = db.ElementaryStep()
        new_step.link(self._elementary_steps)
        new_step.create(main_step_lhs, main_step_rhs)
        new_step.set_type(db.ElementaryStepType.REGULAR)
        new_step.set_transition_state(new_ts.id())
        db_results.add_elementary_step(new_step.id())

        """ Save Reaction Path as a Spline"""
        spline = self.generate_spline(tsopt_task_name)
        new_step.set_spline(spline)
        """ Save Reaction Path """
        charge = ts_calc.settings[utils.settings_names.molecular_charge]
        multiplicity = ts_calc.settings[utils.settings_names.spin_multiplicity]
        model = self._calculation.get_model()
        if self.settings[self.job_key]["store_full_mep"]:
            _ = self.save_mep_in_db(new_step, charge, multiplicity, model)
        """ Save new starting materials if there are any"""
        original_start_structures = self._calculation.get_structures()
        for rid in reactant_structure_ids:
            if rid not in original_start_structures:
                # TODO should duplicates be removed here?
                db_results.add_structure(rid)
        # intermediate function may have written directly to calculation
        # results, therefore add to already existing
        self._calculation.set_results(self._calculation.get_results() + db_results)
        return main_step_lhs, main_step_rhs, new_step

    def _store_ts_with_propensity_info(self, ts_name: str, program_helper: Optional[ProgramHelper],
                                       ts_label: db.Label) -> Tuple[utils.core.Calculator, db.Structure]:
        # do propensity single_points for TS
        self._add_propensity_systems(ts_name)
        self._spin_propensity_single_points(ts_name, "Failed all spin propensity single points for TS, "
                                                     "which means we could not recalculate the TS system. "
                                                     "This points to a SCINE calculator error.")
        new_ts = self._store_structure_with_propensity_check(ts_name, ts_label,
                                                             enforce_to_save_base_name=True)
        self.transfer_properties(self.ref_structure, new_ts)
        ts_calc = self.systems[ts_name]
        self.store_hessian_data(ts_calc, new_ts)
        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, self.ref_structure, new_ts)
        return ts_calc, new_ts

    def _store_structure_with_propensity_check(self, name: str, label: db.Label, enforce_to_save_base_name: bool,
                                               surface_indices: Optional[Union[List[int], Set[int]]] = None) \
            -> db.Structure:
        from scine_utilities import settings_names as sn
        from scine_utilities import KJPERMOL_PER_HARTREE

        def create_impl(structure_name: str) -> db.Structure:
            bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, structure_name, surface_indices)
            new_structure = self.create_new_structure(self.systems[structure_name], label)
            self.store_energy(self.systems[structure_name], new_structure)
            self.store_bond_orders(bond_orders, new_structure)
            self.add_graph(new_structure, bond_orders, surface_indices)
            # Label can change based on graph after optimization
            if label not in [db.Label.TS_OPTIMIZED, db.Label.TS_GUESS]:
                new_graph = self._cbor_graph_from_structure(new_structure)
                new_label = self._determine_new_label_based_on_graph_and_surface_indices(new_graph, surface_indices)
                if label != new_label:
                    print("Propensity check led to new label of " + structure_name + ". Relabeling it.")
                    new_structure.set_label(new_label)
            results = self._calculation.get_results()
            results.add_structure(new_structure.id())
            self._calculation.set_results(results)
            return new_structure

        lowest_name, names_to_save = self._get_propensity_names_within_range(
            name, self.settings[self.job_key]["spin_propensity_energy_range_to_save"]
        )
        spin_propensity_hit = lowest_name != name
        # Printing information
        if spin_propensity_hit:
            print(f"Noticed spin propensity. Lowest energy spin multiplicity of {name} is "
                  f"{self.systems[lowest_name].settings[sn.spin_multiplicity]}")
        if names_to_save:
            print("Spin states with rel. energies to lowest state in kJ/mol which are also saved to the database:")
            print("name | multiplicity | rel. energy")
            base_energy = self.systems[lowest_name].get_results().energy
            for n in names_to_save:
                multiplicity = self.systems[n].settings[sn.spin_multiplicity]
                energy = self.systems[n].get_results().energy
                rel_energy = (energy - base_energy) * KJPERMOL_PER_HARTREE
                print(f"  {n} | {multiplicity} | {rel_energy}")
        if enforce_to_save_base_name:
            print(f"Still saving the base multiplicity of {self.systems[name].settings[sn.spin_multiplicity]} "
                  f"in the elementary step")
            # overwrite names to simply safe and write as product of elementary step
            names_to_save += [lowest_name]
            if name in names_to_save:
                names_to_save.remove(name)
            lowest_name = name

        # Saving information
        name_to_structure_and_label_map = {}
        for n in names_to_save:
            # Store as Tuple[db.Sturcture, db.Label]
            name_to_structure_and_label_map[n] = [create_impl(n)]
            name_to_structure_and_label_map[n] += [name_to_structure_and_label_map[n][0].get_label()]

        name_to_structure_and_label_map[lowest_name] = [create_impl(lowest_name)]
        name_to_structure_and_label_map[lowest_name] += [name_to_structure_and_label_map[lowest_name][0].get_label()]

        # Decide which structure to return
        # Lowest name if no better spin state was found or if the lower spin state still has the same label as name
        if not spin_propensity_hit or \
           name_to_structure_and_label_map[lowest_name][1] == label or \
           enforce_to_save_base_name:
            return name_to_structure_and_label_map[lowest_name][0]
        else:
            return name_to_structure_and_label_map[name][0]

    def store_bond_orders(self, bond_orders: utils.BondOrderCollection, structure: db.Structure) -> None:
        self.store_property(
            self._properties,
            "bond_orders",
            "SparseMatrixProperty",
            bond_orders.matrix,
            self._calculation.get_model(),
            self._calculation,
            structure,
        )

    def _get_propensity_names_within_range(self, name: str, allowed_energy_range: float) -> Tuple[str, List[str]]:
        energies: Dict[str, Optional[float]] = {}
        for shift_name, _ in self._propensity_iterator(name):
            calc = self.systems[shift_name]
            energy = calc.get_results().energy if calc is not None else None
            energies[shift_name] = energy
        # get name with the lowest energy to save as product
        lowest_name = min({k: v for k, v in energies.items() if v is not None}, key=energies.get)  # type: ignore
        lowest_energy = energies[lowest_name]
        assert lowest_energy is not None
        names_within_range: List[str] = []
        for k, v in energies.items():
            if v is not None and k != lowest_name and \
                    abs(v - lowest_energy) * utils.KJPERMOL_PER_HARTREE < allowed_energy_range:
                names_within_range.append(k)
        return lowest_name, names_within_range

    def save_mep_in_db(self, elementary_step: db.ElementaryStep, charge: int, multiplicity: int, model: db.Model) \
            -> List[db.ID]:
        """
        Store each point on the MEP as a structure in the database.
        Attaches `electronic_energy` properties for each point.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        elementary_step :: scine_database.ElementaryStep
            The elementary step of which to store the MEP.
        charge :: int
            The total charge of the system.
        multiplicity :: int
            The spin multiplicity of the system.
        model :: scine_database.Model
            The model with which all energies in the elementary Step were
            calculated.
        """
        def read_trj(fname):
            trj = utils.io.read_trajectory(utils.io.TrajectoryFormat.Xyz, fname)
            energies = []
            with open(fname, "r") as f:
                lines = f.readlines()
                nAtoms = int(lines[0].strip())
                i = 0
                while i < len(lines):
                    energies.append(float(lines[i + 1].strip()))
                    i += nAtoms + 2
            return trj, energies

        def generate_structure(atoms, charge, multiplicity, model):
            # New structure
            new_structure = db.Structure()
            new_structure.link(self._structures)
            new_structure.create(
                atoms,
                charge,
                multiplicity,
                model,
                db.Label.ELEMENTARY_STEP_OPTIMIZED,
            )
            return new_structure.get_id()

        if self.step_direction == "forward":
            dir = "forward"
            rev_dir = "backward"
        elif self.step_direction == "backward":
            dir = "backward"
            rev_dir = "forward"
        else:
            self.raise_named_exception("Could not determine elementary step direction.")

        structure_ids = []
        fpath = os.path.join(
            self.work_dir, f"irc_{rev_dir}", f"irc_{rev_dir}.opt.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, _ = read_trj(fpath)
            for pos in reversed(trj):
                sid = generate_structure(utils.AtomCollection(trj.elements, pos), charge, multiplicity, model)
                structure_ids.append(sid)

        fpath = os.path.join(
            self.work_dir, f"irc_{rev_dir}", f"irc_{rev_dir}.irc.{rev_dir}.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, _ = read_trj(fpath)
            for pos in reversed(trj):
                sid = generate_structure(utils.AtomCollection(trj.elements, pos), charge, multiplicity, model)
                structure_ids.append(sid)
        else:
            raise RuntimeError(
                f"Missing IRC trajectory file: irc_{rev_dir}/irc_{rev_dir}.irc.{rev_dir}.trj.xyz"
            )

        structure_ids.append(elementary_step.get_transition_state())

        fpath = os.path.join(
            self.work_dir, f"irc_{dir}", f"irc_{dir}.irc.{dir}.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, _ = read_trj(fpath)
            for pos in trj:
                sid = generate_structure(utils.AtomCollection(trj.elements, pos), charge, multiplicity, model)
                structure_ids.append(sid)
        else:
            raise RuntimeError(
                f"Missing IRC trajectory file: irc_{dir}/irc_{dir}.irc.{dir}.trj.xyz"
            )

        fpath = os.path.join(self.work_dir, f"irc_{dir}", f"irc_{dir}.opt.trj.xyz")
        if os.path.isfile(fpath):
            trj, _ = read_trj(fpath)
            for pos in trj:
                sid = generate_structure(utils.AtomCollection(trj.elements, pos), charge, multiplicity, model)
                structure_ids.append(sid)

        elementary_step.set_path(structure_ids)
        return structure_ids

    def _includes_label(self, structure_id_list: List[db.ID], labels: List[db.Label]) -> bool:
        """
        Returns if any structure in the list has any of the given labels.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        structure_id_list :: List[db.ID]
            A list structure ids
        labels :: List[db.Label]
            The required labels
        """
        return self._label_locations(structure_id_list, labels)[0] is not None

    def _label_locations(self, structure_id_list: List[db.ID], labels: List[db.Label]) \
            -> Union[Tuple[int, int], Tuple[None, None]]:
        """
        Returns the first index of the structure in the list that holds any of the given labels
        and the index of the label.
        Returns None if no given structure has none of the given labels.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        structure_id_list :: List[db.ID]
            A list structure ids
        labels :: List[db.Label]
            The required labels
        """
        for i, sid in enumerate(structure_id_list):
            structure = db.Structure(sid, self._structures)
            for j, label in enumerate(labels):
                if structure.get_label() == label:
                    return i, j
        return None, None

    def _determine_new_label_based_on_graph_and_surface_indices(self, graph_str: str,
                                                                surface_indices: Union[List[int], Set[int], None]) \
            -> db.Label:
        graph_is_split = ";" in graph_str
        no_surf_split_decision_label = db.Label.COMPLEX_OPTIMIZED if graph_is_split else db.Label.MINIMUM_OPTIMIZED
        surf_split_decision_label = db.Label.SURFACE_COMPLEX_OPTIMIZED if graph_is_split else db.Label.SURFACE_OPTIMIZED
        thresh = self.settings[self.job_key]["n_surface_atom_threshold"]
        if surface_indices is not None and len(surface_indices) > thresh:
            return surf_split_decision_label
        return no_surf_split_decision_label

    def _determine_new_label_based_on_graph(self, calculator: utils.core.Calculator, graph_str: str) -> db.Label:
        """
        Determines label for a product structure of the given react job based on the given graph and the labels
        of the starting structures.
        Crucially, this method only works if
          - the given structure is a superstructure of all start structures
        For multiple split structures we require a mapping information on the atom level between the start structures
        and the individual products. If this is the case, the labels must be assigned to all products at once.
        See `_determine_product_labels_of_single_compounds` for that, which will however not work on complexes.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        calculator :: Core::Calculator
            The calculator holding the structure
        graph_str :: str
            The cbor graph of one or more molecules (separated by ';')
        Returns
        -------
        label :: db.Label
            The correct label for the new structure corresponding to the given graph
        """
        graph_is_split = ";" in graph_str
        no_surf_split_decision_label = db.Label.COMPLEX_OPTIMIZED if graph_is_split else db.Label.MINIMUM_OPTIMIZED
        surf_split_decision_label = db.Label.SURFACE_COMPLEX_OPTIMIZED if graph_is_split else db.Label.SURFACE_OPTIMIZED
        start_structure_ids = self._calculation.get_structures()
        if not self._includes_label(start_structure_ids, [db.Label.SURFACE_OPTIMIZED,
                                                          db.Label.USER_SURFACE_OPTIMIZED,
                                                          db.Label.SURFACE_COMPLEX_OPTIMIZED,
                                                          db.Label.USER_SURFACE_COMPLEX_OPTIMIZED]):
            # no surface present in inputs
            return no_surf_split_decision_label
        # we had a surface in the inputs
        start_structures = [db.Structure(s, self._structures) for s in start_structure_ids]
        adsorb_guess_index, _ = self._label_locations(start_structure_ids, [db.Label.SURFACE_ADSORPTION_GUESS])
        if adsorb_guess_index is not None:
            # eliminate adsorb guess from start structure considerations
            start_structures = [s for i, s in enumerate(start_structures) if i != adsorb_guess_index]
        n_start_atoms = sum(len(s.get_atoms()) for s in start_structures)
        if len(calculator.structure) == n_start_atoms:
            # we got no split in react job, structure must still be a surface
            return surf_split_decision_label
        raise RuntimeError(f"Could not deduced the label for the new structure {graph_str} "
                           f"based on start structures {[str(s) for s in start_structure_ids]}")

    def _determine_product_labels_of_single_compounds(self, names: List[str],
                                                      component_map: Optional[List[int]] = None) -> List[db.Label]:
        """
        Determines labels of all individual product structures of the given react job based on the labels of the
        starting structures.
        Crucially, this method only works if
          - each specified system in the `names` holds only a compound
          - the `products_component_map` has been evaluated in the IRC check
        For complex structures this method does not work, because we require the graph for that, which requires
        the knowledge about individual surface atoms.
        See `_determine_new_label_based_on_graph` for that, which will however not work on only a partial structure
        of the initial start structure combination for surfaces.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        names :: List[str]
            The list of system names of the products in the systems map
        component_map :: Optional[List[int]]
            The component map of the given systems, take product_component_map if None

        Returns
        -------
        labels :: List[db.Label]
            The correct labels for the new structures
        """
        if self.products_component_map is None and component_map is None:
            self.raise_named_exception(f"Could not deduce the labels for the new structures {names}")
        if component_map is None:
            component_map = self.products_component_map
        assert component_map is not None  # for type check
        surface_indices = self.surface_indices_all_structures()
        if not surface_indices:
            # we don't have a surface --> all compounds and no user input because products
            return [db.Label.MINIMUM_OPTIMIZED] * len(names)
        # sanity checks
        n_product_atoms = sum(len(self.systems[name].structure) for name in names)
        if any(index >= n_product_atoms for index in surface_indices):
            self.raise_named_exception("Surface indices include invalid numbers for the given products")
        if len(component_map) != n_product_atoms:
            self.raise_named_exception("Invalid product component map for the given products")
        product_surface_atoms = [0] * len(names)
        for index in surface_indices:
            product_surface_atoms[component_map[index]] += 1
        # do not categorize if only single surface atom, but assume this is a transfer from the surface to the product
        thresh = self.settings[self.job_key]["n_surface_atom_threshold"]
        return [db.Label.SURFACE_OPTIMIZED if n > thresh else db.Label.MINIMUM_OPTIMIZED for n in product_surface_atoms]

    def _tsopt_hess_irc_ircopt(self, tsguess_system_name: str, settings_manager: SettingsManager) \
            -> Tuple[List[str], Optional[List[str]]]:
        """
        Takes a TS guess and carries out:
        * TS optimization
        * Hessian calculation and check for valid TS
        * IRC calculation
        * random displacement of IRC points
        * Optimization with faster converging optimizer than Steepest Descent to arrive at true minima

        Parameters
        ----------
        tsguess_system_name : str
            The name of the system holding the TS guess
        settings_manager : SettingsManager
            The settings manager
        """
        import scine_readuct as readuct
        inputs = [tsguess_system_name]
        """ TSOPT JOB """
        self.setup_automatic_mode_selection("tsopt")
        print("TSOpt Settings:")
        print(self.settings["tsopt"], "\n")
        self.systems, success = self.observed_readuct_call(
            'run_tsopt_task', self.systems, inputs, **self.settings["tsopt"])
        self.throw_if_not_successful(
            success,
            self.systems,
            self.output("tsopt"),
            ["energy"],
            "TS optimization failed:\n",
        )

        """ TS HESSIAN """
        inputs = self.output("tsopt")
        self.systems, success = readuct.run_hessian_task(self.systems, inputs)
        self.throw_if_not_successful(
            success,
            self.systems,
            inputs,
            ["energy", "hessian", "thermochemistry"],
            "TS Hessian calculation failed.\n",
        )

        if self.n_imag_frequencies(inputs[0]) != 1:
            self._save_ts_for_restart(db.Label.TS_GUESS)
            self.raise_named_exception(f"Error: {self.name} failed with message: "
                                       f"TS has incorrect number of imaginary frequencies.")

        """ IRC JOB """
        # IRC (only a few steps to allow decent graph extraction)
        print("IRC Settings:")
        print(self.settings["irc"], "\n")
        self.systems, success = self.observed_readuct_call(
            'run_irc_task', self.systems, inputs, **self.settings["irc"])

        """ IRC OPT JOB """
        # Run a small energy minimization after initial IRC
        inputs = self.output("irc")
        print("IRC Optimization Settings:")
        print(self.settings["ircopt"], "\n")
        for i in inputs:
            atoms = self.systems[i].structure
            self.random_displace_atoms(atoms)
            self.systems[i].positions = atoms.positions
        self.systems, success = self.observed_readuct_call(
            'run_opt_task', self.systems, [inputs[0]], **self.settings["ircopt"])
        self.systems, success = self.observed_readuct_call(
            'run_opt_task', self.systems, [inputs[1]], **self.settings["ircopt"])

        """ Check whether we have a valid IRC """
        initial_charge = settings_manager.calculator_settings[utils.settings_names.molecular_charge]
        product_names, start_names = self.irc_sanity_checks_and_analyze_sides(
            initial_charge, self.check_charges, inputs, settings_manager.calculator_settings)
        if product_names is None:  # IRC did not pass checks, reason has been set as comment, complete job
            self.verify_connection()
            self.capture_raw_output()
            update_model(
                self.systems[self.output("tsopt")[0]],
                self._calculation,
                self.config,
            )
            raise breakable.Break
        return product_names, start_names

    def _tsopt_hess_irc_ircopt_postprocessing(self, tsguess_system_name: str, settings_manager: SettingsManager,
                                              program_helper: Optional[ProgramHelper]) -> None:
        """
        Takes a TS guess and carries out:
        * TS optimization
        * Hessian calculation and check for valid TS
        * IRC calculation
        * random displacement of IRC points
        * Faster optimization to arrive at true minima
        * Checks for the validity of the IRC and saving the results

        Notes
        -----
        All but last step are done in `_tsopt_hess_irc_ircopt`

        Parameters
        ----------
        tsguess_system_name : str
            The name of the system holding the TS guess
        settings_manager : SettingsManager
            The settings manager
        program_helper : Optional[ProgramHelper]
            The program helper
        """
        product_names, start_names = self._tsopt_hess_irc_ircopt(tsguess_system_name, settings_manager)
        """ Store new starting material conformer(s) """
        if start_names is not None:
            start_structures = self.store_start_structures(
                start_names, program_helper, "tsopt")
        else:
            start_structures = self._calculation.get_structures()

        self.react_postprocessing(product_names, program_helper, "tsopt", start_structures)
