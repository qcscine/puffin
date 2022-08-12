# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import sys
from copy import deepcopy
import scine_molassembler as masm

from .job import job_configuration_wrapper
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
        # to be extended by child:
        self.settings: Dict[str, Dict[str, Any]] = {
            self.job_key: {
                "imaginary_wavenumber_threshold": 0.0,
                "spin_propensity_check": 2,
                "store_full_mep": False,
                "store_all_structures": False,
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
            }
        }
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

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs():
        return ["database", "molassembler", "readuct", "utils"]

    def observed_readuct_call(self, call_str: str, systems, input_names, **kwargs):
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

    def reactive_complex_preparations(
        self,
    ) -> Tuple[SettingsManager, Union[ProgramHelper, None]]:
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
        import scine_utilities as utils
        import scine_database as db

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
        reactive_complex_graph = self.make_graph_from_calc(self.systems, self.rc_key)
        if not masm.JsonSerialization.equal_molecules(reactive_complex_graph, self.start_graph):
            self.raise_named_exception(
                "Reactive complex graph differs from combined start structure graphs."
            )
        return settings_manager, program_helper

    def check_structures(self, start_structures: Union[List, None] = None):
        """
        Perform sanity check whether we only have 1 or 2 structures in the configured calculation. Return a possible
        reference structure (largest one) for the construction of a ProgramHelper.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        start_structures :: List[Scine::Database::Structure]
            If given, this structure list is used instead of the list given in self._calculation.get_structures().

        Returns
        -------
        ref_structure :: db.Structure (Scine::Database::Structure)
            The largest structure of the calculation.
        """
        import scine_database as db
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
        else:
            self.raise_named_exception(
                "Reactive complexes built from more than 2 structures are not supported."
            )
        return db.Structure(ref_id, self._structures)

    def sort_settings(self, task_settings: dict):
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
                    "The key '{}' was not recognized.".format(key)
                )

        if "ircopt" in self.settings.keys() and "output" in self.settings["ircopt"]:
            self.raise_named_exception(
                "Cannot specify a separate output system for the optimization of the IRC end points"
            )

    def save_initial_graphs_and_charges(self, settings_manager: SettingsManager, structures: List):
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
        if len(structures) < 3:
            for i, s in enumerate(structures):
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
            self.determine_pes_of_rc(settings_manager, *structures)
        else:
            # should not be reachable
            self.raise_named_exception(
                "Reactive complexes built from more than 2 structures are not supported."
            )

    def _cbor_graph_from_structure(self, structure) -> str:
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

    @ staticmethod
    def _decision_list_from_structure(structure) -> Optional[str]:
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

    def build_reactive_complex(self, settings_manager: SettingsManager):
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
        import scine_database as db
        import scine_utilities as utils

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
            # Intermolecular reactions reactions require in situ generation of the reactive complex
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

        # should not be reachable
        self.raise_named_exception(
            "Reactive complexes built from more than 2 structures are not supported."
        )

    def determine_pes_of_rc(self, settings_manager: SettingsManager, s0, s1=None):
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

    def random_displace_atoms(self, atoms, displacement: float = 0.05):
        """
        Apply small seeded random displacement based on setting
        """
        np.random.seed(42)
        coords = np.array([atoms.get_position(i) for i in range(len(atoms))])
        coords += displacement * (np.random.rand(*coords.shape) - 0.5) * 2.0 / np.sqrt(3.0)
        atoms.positions = coords

    def setup_automatic_mode_selection(self, name: str):
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
        import scine_utilities as utils

        atoms = self.systems[name].structure
        modes_container = utils.normal_modes.calculate(self.systems[name].get_results().hessian, atoms)
        wavenumbers = modes_container.get_wave_numbers()

        return np.count_nonzero(np.array(wavenumbers) < self.settings[self.job_key]["imaginary_wavenumber_threshold"])

    def get_graph_charges_multiplicities(self, name: str, total_charge: int):
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

        Returns
        -------
        split_structures :: List[utils.AtomCollection]
            List of atom collections corresponding to the split molecules.
        graph_string :: str
            Sorted molassembler cbor graphs separated by semicolons.
        charges :: List[int]
            Charges of the molecules.
        multiplicities :: List[int]
            Minimal multiplicities of the molecules.
        decision_lists :: List[str]
            Molassembler decision lists for free dihedrals
        """
        import scine_readuct as readuct
        import scine_utilities as utils

        bond_orders = self.make_bond_orders_from_calc(self.systems, name)

        pbc_string = self.systems[name].settings.get(utils.settings_names.periodic_boundaries, "")
        masm_results = masm_helper.get_molecules_result(
            self.systems[name].structure,
            bond_orders,
            self.connectivity_settings,
            pbc_string,
        )

        split_structures = masm_results.component_map.apply(self.systems[name].structure)
        decision_lists = [masm_helper.get_decision_list_from_molecule(
            m, a) for m, a in zip(masm_results.molecules, split_structures)]

        # Get cbor graphs
        graphs = []
        for molecule in masm_results.molecules:
            graphs.append(masm_helper.get_cbor_graph_from_molecule(molecule))

        # Determine partial charges, charges per molecules and number of electrons per molecule
        bond_orders = self.make_bond_orders_from_calc(self.systems, name)
        partial_charges = self.systems[name].get_results().atomic_charges
        if partial_charges is None:
            self.systems, success = readuct.run_single_point_task(
                self.systems, [name], require_charges=True
            )
            self.throw_if_not_successful(
                success, self.systems, [name], ["energy", "atomic_charges"]
            )
            partial_charges = self.systems[name].get_results().atomic_charges
            self.systems[name].get_results().bond_orders = bond_orders

        charges = []
        n_electrons = []
        for i in range(len(split_structures)):
            charges.append(0.0)
        for i, c in zip(masm_results.component_map, partial_charges):
            charges[i] += c
        residual = []
        for i in range(len(split_structures)):
            residual.append(charges[i] - round(charges[i]))
            charges[i] = int(round(charges[i]))

        # Check if electrons were created or vanished by virtue of rounding
        electron_diff = int(round(sum(charges) - total_charge))
        if electron_diff < 0:
            # Remove electrons if need be
            max_charge_vals = np.array(charges).argsort()[-electron_diff:]
            for i in max_charge_vals:
                charges[i] += 1
        elif electron_diff > 0:
            # Add electrons if need be
            min_charge_vals = np.array(charges).argsort()[:electron_diff]
            for i in min_charge_vals:
                charges[i] -= 1
        for i in range(len(split_structures)):
            electrons = 0.0
            for elem in split_structures[i].elements:
                electrons += utils.ElementInfo.Z(elem)
            electrons -= charges[i]
            n_electrons.append(int(round(electrons)))

        # This assumes minimal multiplicity, product multiplicities are again checked later around this multiplicity
        multiplicities = [nel % 2 + 1 for nel in n_electrons]

        # Sort everything according to graphs and if these are equal according to charges and then multiplicities
        graphs, charges, multiplicities, decision_lists, structure_order = (
            list(start_val)
            for start_val in zip(*sorted(zip(
                graphs,
                charges,
                multiplicities,
                decision_lists,
                range(0, len(split_structures)))))
        )
        graph_string = ";".join(graphs)

        ordered_structures = [split_structures[i] for i in structure_order]

        return ordered_structures, graph_string, charges, multiplicities, decision_lists

    def check_for_barrierless_reaction(self):
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
        _, rc_opt_graph, _, _, rc_opt_decision_lists =  \
            self.get_graph_charges_multiplicities(self.rc_opt_system_name, sum(self.start_charges))

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
        for name, charge in zip(forward_names, forward_charges):
            s, g, _, _, d = self.get_graph_charges_multiplicities(name, charge)
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
        for name, charge in zip(backward_names, backward_charges):
            s, g, _, _, d = self.get_graph_charges_multiplicities(name, charge)
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
            return None, None

        # Do not expect matching charges if reactive complex charge differs from sum of start structure charges
        if masm.JsonSerialization.equal_molecules(forward_graph, self.start_graph)\
                and (not check_charges or forward_charges == self.start_charges):
            product_names = backward_names
            self.step_direction = "backward"
        elif masm.JsonSerialization.equal_molecules(backward_graph, self.start_graph) and (
            not check_charges or backward_charges == self.start_charges
        ):
            product_names = forward_names
            self.step_direction = "forward"
        elif ';' in self.start_graph:
            rc_opt_graph, _ = self.check_for_barrierless_reaction()
            print("Barrierless Check Graph:")
            print(rc_opt_graph)
            if rc_opt_graph is None:
                self._calculation.set_comment(self.name + ": No IRC structure matches starting structure.")
                return None, None
            if masm.JsonSerialization.equal_molecules(forward_graph, rc_opt_graph):
                self.step_direction = "backward"
                product_names = backward_names
                self.lhs_barrierless_reaction = True
            elif masm.JsonSerialization.equal_molecules(backward_graph, rc_opt_graph):
                self.step_direction = "forward"
                product_names = forward_names
                self.lhs_barrierless_reaction = True
            else:
                self._calculation.set_comment(self.name + ": No IRC structure matches starting structure.")
                return None, None
        else:
            self._calculation.set_comment(self.name + ": No IRC structure matches starting structure.")
            return None, None

        # Compare decision lists of start structures:
        original_decision_lists = self.start_decision_lists
        if self.step_direction == "backward":
            new_decision_lists = forward_decision_lists
        else:
            new_decision_lists = backward_decision_lists

        decision_lists_match: bool = True
        for new, orig in zip(new_decision_lists, original_decision_lists):
            if not masm.JsonSerialization.equal_decision_lists(new, orig):
                decision_lists_match = False
                break

        if not decision_lists_match:
            if self.step_direction == "backward":
                start_names = forward_names
            else:
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
                self._calculation.set_comment(self.name + ": IRC does not match double ended method")
                return None, None

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
        structures,
        structure_charges: List[int],
        structure_multiplicities: List[int],
        calculator_settings: dict,
        stop_on_error: Optional[bool] = True
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
        stop_on_error :: Optional[bool]
            If set to False, skip unsuccessful calculations and replace calculator with None

        Returns
        -------
        product_names :: List[str]
            A list of the access keys to the structures in the system map.
        """
        import scine_readuct as readuct
        import scine_utilities as utils

        structure_names = []
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
                    self._calculation.get_model().method_family,
                    **structure_calculator_settings,
                )
                self.systems[name] = new
            except BaseException as e:
                if stop_on_error:
                    raise e
                sys.stderr.write(f"{name} cannot be calculated because: {str(e)}")
                self.systems[name] = None

        print("Product Opt Settings:")
        print(self.settings["opt"], "\n")
        # Optimize structures, if they have more than one atom; otherwise just run a single point calculation
        for structure in structure_names:
            if self.systems[structure] is None:
                continue
            try:
                self.systems, success = readuct.run_single_point_task(
                    self.systems,
                    [structure],
                    spin_propensity_check=self.settings[self.job_key]["spin_propensity_check"],
                    require_bond_orders=True,
                )
                if len(self.systems[structure].structure) > 1:
                    print("Optimizing " + structure + ":\n")
                    self.systems, success = self.observed_readuct_call(
                        'run_opt_task', self.systems, [structure], **self.settings["opt"]
                    )
                    self.throw_if_not_successful(
                        success,
                        self.systems,
                        [structure],
                        ["energy"],
                        f"{name_stub.capitalize()} optimization failed:\n",
                    )
                    self.systems, success = readuct.run_single_point_task(
                        self.systems,
                        [structure],
                        spin_propensity_check=self.settings[self.job_key]["spin_propensity_check"],
                        require_bond_orders=True,
                    )
                self.throw_if_not_successful(
                    success,
                    self.systems,
                    [structure],
                    ["energy", "bond_orders"],
                    f"{name_stub.capitalize()} optimization failed:\n",
                )
            except BaseException as e:
                if stop_on_error:
                    raise e
                sys.stderr.write(f"{structure} cannot be calculated because: {str(e)}")
                self.systems[structure] = None
        return structure_names

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
        import scine_utilities as utils
        import os

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

        fpath = os.path.join(
            self.work_dir, f"irc_{rev_dir}", f"irc_{rev_dir}.opt.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(reversed(trj), reversed(energies)):
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)

        fpath = os.path.join(
            self.work_dir, f"irc_{rev_dir}", f"irc_{rev_dir}.irc.{rev_dir}.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(reversed(trj), reversed(energies)):
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)
        else:
            raise Exception(
                f"Missing IRC trajectory file: irc_{rev_dir}/irc_{rev_dir}.irc.{rev_dir}.trj.xyz"
            )

        fpath = os.path.join(self.work_dir, "ts", "ts.xyz")
        if os.path.isfile(fpath):
            ts_calc = self.systems[self.output(tsopt_task_name)[0]]
            results = ts_calc.get_results()
            ts_xyz, _ = utils.io.read(fpath)
            rpi.append_structure(ts_xyz, results.energy, True)
        else:
            raise Exception("Missing TS structure file: ts/ts.xyz")

        fpath = os.path.join(
            self.work_dir, f"irc_{dir}", f"irc_{dir}.irc.{dir}.trj.xyz"
        )
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(trj, energies):
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)
        else:
            raise Exception(
                f"Missing IRC trajectory file: irc_{dir}/irc_{dir}.irc.{dir}.trj.xyz"
            )

        fpath = os.path.join(self.work_dir, f"irc_{dir}", f"irc_{dir}.opt.trj.xyz")
        if os.path.isfile(fpath):
            trj, energies = read_trj(fpath)
            for pos, e in zip(trj, energies):
                rpi.append_structure(utils.AtomCollection(trj.elements, pos), e)

        # Get spline
        spline = rpi.spline(n_fit_points, degree)
        return spline

    def store_start_structures(
        self,
        start_structure_names: List[str],
        program_helper: Union[ProgramHelper, None],
        tsopt_task_name: str
    ):
        """
        Store the new start systems system in the database.

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

        Returns
        -------
        start_structure_ids :: List[scine_database.ID]
            A list of the database IDs of the start structures.
        """
        import scine_database as db

        # Update model to make sure there are no 'any' values left
        update_model(
            self.systems[self.output(tsopt_task_name)[0]],
            self._calculation,
            self.config,
        )

        start_structure_ids = []
        for name in start_structure_names:
            # Check if the new structures are actually duplicates
            duplicate: Optional[db.ID] = None
            dl = ';'.join(self.make_decision_lists_from_calc(self.systems, name))
            graph = self.make_graph_from_calc(self.systems, name)
            for initial_id in self._calculation.get_structures():
                initial_structure = db.Structure(initial_id)
                initial_structure.link(self._structures)
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
                    existing_structure = db.Structure(existing_structure_id)
                    existing_structure.link(self._structures)
                    existing_structure_dl = existing_structure.get_graph("masm_decision_list")
                    if masm.JsonSerialization.equal_decision_lists(dl, existing_structure_dl):
                        duplicate = existing_structure_id
            if duplicate is not None:
                start_structure_ids.append(duplicate)
                continue

            new_structure = self.create_new_structure(self.systems[name], db.Label.MINIMUM_OPTIMIZED)
            self.transfer_properties(self.ref_structure, new_structure)
            self.store_energy(self.systems[name], new_structure)
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                self.systems[name].get_results().bond_orders.matrix,
                self._calculation.get_model(),
                self._calculation,
                new_structure,
            )
            self.add_graph(new_structure, self.systems[name].get_results().bond_orders)
            if ";" in graph:
                new_structure.set_label(db.Label.COMPLEX_OPTIMIZED)
            if program_helper is not None:
                program_helper.calculation_postprocessing(self._calculation, self.ref_structure, new_structure)
            start_structure_ids.append(new_structure.id())
        return start_structure_ids

    def save_barrierless_reaction(self, product_graph: str, program_helper: Optional[ProgramHelper]):
        import scine_database as db
        self.lhs_barrierless_reaction = True
        print("Barrierless product Graph:")
        print(product_graph)
        print("Start Graph:")
        print(self.start_graph)
        print("Barrierless Reaction Found")
        db_results = self._calculation.get_results()
        db_results.clear()
        # Save RHS of barrierless step
        rhs_complex_label = self.rc_opt_system_name
        rhs_complex_system = self.systems[rhs_complex_label]
        if ";" in product_graph:
            rhs_complex = self.create_new_structure(rhs_complex_system, db.Label.COMPLEX_OPTIMIZED)
        else:
            rhs_complex = self.create_new_structure(rhs_complex_system, db.Label.MINIMUM_OPTIMIZED)
        db_results.add_structure(rhs_complex.id())
        self.transfer_properties(self.ref_structure, rhs_complex)
        self.store_energy(self.systems[rhs_complex_label], rhs_complex)
        bond_orders = self.make_bond_orders_from_calc(self.systems, rhs_complex_label)
        self.store_property(
            self._properties,
            "bond_orders",
            "SparseMatrixProperty",
            bond_orders.matrix,
            self._calculation.get_model(),
            self._calculation,
            rhs_complex,
        )
        self.add_graph(rhs_complex, bond_orders)
        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, self.ref_structure, rhs_complex)
        # Save step
        new_step = db.ElementaryStep()
        new_step.link(self._elementary_steps)
        new_step.create(self._calculation.get_structures(), [rhs_complex.id()])
        new_step.set_type(db.ElementaryStepType.BARRIERLESS)
        db_results.add_elementary_step(new_step.id())
        self._calculation.set_comment(self.name + ": Barrierless reaction found.")
        self._calculation.set_results(self._calculation.get_results() + db_results)

    def react_postprocessing(
        self,
        product_names: List[str],
        program_helper: Union[ProgramHelper, None],
        tsopt_task_name: str,
        reactant_structur_ids: List
    ):
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
        reactant_structur_ids :: List[scine_database.ID]
            A list of all structure IDs for the reactants.
        """
        import scine_database as db
        import scine_utilities as utils

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
        new_label = db.Label.MINIMUM_OPTIMIZED
        end_structures = []
        for product in product_names:
            new_structure = self.create_new_structure(self.systems[product], new_label)
            self.transfer_properties(self.ref_structure, new_structure)
            self.store_energy(self.systems[product], new_structure)
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                self.systems[product].get_results().bond_orders.matrix,
                self._calculation.get_model(),
                self._calculation,
                new_structure,
            )
            self.add_graph(new_structure, self.systems[product].get_results().bond_orders)
            if program_helper is not None:
                program_helper.calculation_postprocessing(self._calculation, self.ref_structure, new_structure)
            db_results.add_structure(new_structure.id())
            end_structures.append(new_structure.id())
        """ Save TS """
        ts_calc = self.systems[self.output(tsopt_task_name)[0]]
        new_ts = self.create_new_structure(ts_calc, db.Label.TS_OPTIMIZED)
        self.transfer_properties(self.ref_structure, new_ts)
        self.store_hessian_data(ts_calc, new_ts)
        if program_helper is not None:
            program_helper.calculation_postprocessing(self._calculation, self.ref_structure, new_ts)
        db_results.add_structure(new_ts.id())

        """ Save Complexes """
        if self.lhs_barrierless_reaction or self.lhs_complexation:
            if self.lhs_barrierless_reaction:
                lhs_complex_label = self.rc_opt_system_name
                lhs_complex_graph, _ = self.check_for_barrierless_reaction()
                lhs_complex_system = self.systems[lhs_complex_label]
                if ';' in lhs_complex_graph:
                    lhs_complex = self.create_new_structure(lhs_complex_system, db.Label.COMPLEX_OPTIMIZED)
                else:
                    lhs_complex = self.create_new_structure(lhs_complex_system, db.Label.MINIMUM_OPTIMIZED)
                db_results.add_structure(lhs_complex.id())
            else:
                lhs_complex_label = "irc_backward" if self.step_direction == "forward" else "irc_forward"
                lhs_complex_system = self.systems[lhs_complex_label]
                lhs_complex = self.create_new_structure(lhs_complex_system, db.Label.COMPLEX_OPTIMIZED)
            bond_orders = self.make_bond_orders_from_calc(self.systems, lhs_complex_label)
            self.transfer_properties(self.ref_structure, lhs_complex)
            self.store_energy(self.systems[lhs_complex_label], lhs_complex)
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                bond_orders.matrix,
                self._calculation.get_model(),
                self._calculation,
                lhs_complex,
            )
            self.add_graph(lhs_complex, bond_orders)
            # Keep track of the lhs structure calculation.
            if program_helper is not None:
                program_helper.calculation_postprocessing(self._calculation, self.ref_structure, lhs_complex)
            db_results.add_structure(lhs_complex.id())
        if self.rhs_complexation:
            rhs_complex_label = "irc_forward" if self.step_direction == "forward" else "irc_backward"
            bond_orders = self.make_bond_orders_from_calc(self.systems, rhs_complex_label)
            rhs_complex_system = self.systems[rhs_complex_label]
            rhs_complex = self.create_new_structure(rhs_complex_system, db.Label.COMPLEX_OPTIMIZED)
            self.transfer_properties(self.ref_structure, rhs_complex)
            self.store_energy(self.systems[rhs_complex_label], rhs_complex)
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                bond_orders.matrix,
                self._calculation.get_model(),
                self._calculation,
                rhs_complex,
            )
            self.add_graph(rhs_complex, bond_orders)
            if program_helper is not None:
                program_helper.calculation_postprocessing(self._calculation, self.ref_structure, rhs_complex)
            db_results.add_structure(rhs_complex.id())

        """ Save Steps """
        main_step_lhs = reactant_structur_ids
        main_step_rhs = end_structures
        if self.lhs_barrierless_reaction or self.lhs_complexation:
            new_step = db.ElementaryStep()
            new_step.link(self._elementary_steps)
            new_step.create(reactant_structur_ids, [lhs_complex.id()])
            new_step.set_type(db.ElementaryStepType.BARRIERLESS)
            db_results.add_elementary_step(new_step.id())
            main_step_lhs = [lhs_complex.id()]
        if self.rhs_complexation:
            new_step = db.ElementaryStep()
            new_step.link(self._elementary_steps)
            new_step.create([rhs_complex.id()], end_structures)
            new_step.set_type(db.ElementaryStepType.BARRIERLESS)
            db_results.add_elementary_step(new_step.id())
            main_step_rhs = [rhs_complex.id()]
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
        for rid in reactant_structur_ids:
            if rid not in original_start_structures:
                # TODO should duplicates be removed here?
                db_results.add_structure(rid)
        # intermediate function may have written directly to calculation
        #   results, therefore add to already existing
        self._calculation.set_results(self._calculation.get_results() + db_results)

    def save_mep_in_db(self, elementary_step, charge, multiplicity, model):
        """
        Store each point on the MEP as a structure in the database.
        Attaches `electronic_energy` properties for each point.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        tsopt_task_name :: str
            Name of the transition state task.
        """
        import scine_utilities as utils
        import scine_database as db
        import os

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
            raise Exception(
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
            raise Exception(
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
