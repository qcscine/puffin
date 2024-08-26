# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from math import ceil
from typing import Any, Dict, List, Tuple, Union, Optional, Set, TYPE_CHECKING
import sys
import os
from copy import deepcopy

import numpy as np

from .job import breakable, is_configured
from .scine_hessian_job import HessianJob
from .scine_optimization_job import OptimizationJob
from .scine_propensity_job import ScinePropensityJob
from scine_puffin.utilities.scine_helper import SettingsManager, update_model
from scine_puffin.utilities.program_helper import ProgramHelper
from scine_puffin.utilities import masm_helper
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency
from scine_puffin.utilities.task_to_readuct_call import SubTaskToReaductCall

if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")
if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_readuct") or TYPE_CHECKING:
    import scine_readuct as readuct
else:
    readuct = MissingDependency("scine_readuct")
if module_exists("scine_molassembler") or TYPE_CHECKING:
    import scine_molassembler as masm
else:
    masm = MissingDependency("scine_molassembler")


class ReactJob(ScinePropensityJob, OptimizationJob, HessianJob, ABC):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface to find new reactions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "ReactJob"  # to be overwritten by child
        self.exploration_key = ""  # to be overwritten by child
        self.rc_key = "rc"
        self.rc_opt_system_name = "rcopt"
        self.single_point_key = "sp"
        self.no_irc_structure_matches_start = False
        # to be extended by child:
        self.settings: Dict[str, Dict[str, Any]] = {
            **self.settings,
            self.job_key: {
                **self.settings[self.job_key],
                "imaginary_wavenumber_threshold": 0.0,
                "store_full_mep": False,
                "store_all_structures": False,
                "store_structures_with_frequency": {
                    task: 0 for task in SubTaskToReaductCall.__members__
                },
                "store_structures_with_fraction": {
                    task: 0.0 for task in SubTaskToReaductCall.__members__
                },
                "always_add_barrierless_step_for_reactive_complex": False,
                "allow_exhaustive_product_decomposition": False,
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
        self.start_graph = ""
        self.end_graph = ""
        self.start_charges: List[int] = []
        self.start_multiplicities: List[int] = []
        self.start_decision_lists: List[str] = []
        self.ref_structure: Optional[db.Structure] = None
        self.step_direction: Optional[str] = None
        self.lhs_barrierless_reaction = False
        self.lhs_complexation = False
        self.rhs_complexation = False
        self.complexation_criterion = -12.0 / 2625.5  # kj/mol
        self.check_charges = True
        self.systems: Dict[str, Optional[utils.core.Calculator]] = {}
        self._component_maps: Dict[str, List[int]] = {}
        self.products_component_map: Optional[List[int]] = None

    @classmethod
    def optional_settings_doc(cls) -> str:
        return super().optional_settings_doc() + """\n
        The following options are available for the reactive complex generation:

        rc_x_alignment_0 : List[float], length=9
            In case of two structures building the reactive complex, this option
            describes a rotation of the first structure (index 0) that aligns
            the reaction coordinate along the x-axis (pointing towards +x).
            The rotation assumes that the geometric mean position of all
            atoms in the reactive site (``nt_lhs_list``) is shifted into the
            origin.
        rc_x_alignment_1 : List[float], length=9
            In case of two structures building the reactive complex, this option
            describes a rotation of the second structure (index 1) that aligns
            the reaction coordinate along the x-axis (pointing towards -x).
            The rotation assumes that the geometric mean position of all
            atoms in the reactive site (``nt_rhs_list``) is shifted into the
            origin.
        rc_x_rotation : float
            In case of two structures building the reactive complex, this option
            describes a rotation angle around the x-axis of one of the two
            structures after ``rc_x_alignment_0`` and ``rc_x_alignment_1`` have
            been applied.
        rc_x_spread : float
            In case of two structures building the reactive complex, this option
            gives the distance by which the two structures are moved apart along
            the x-axis after ``rc_x_alignment_0``, ``rc_x_alignment_1``, and
            ``rc_x_rotation`` have been applied.
        rc_displacement : float
            In case of two structures building the reactive complex, this option
            adds a random displacement to all atoms (random direction, random
            length). The maximum length of this displacement (per atom) is set to
            be the value of this option.
        rc_spin_multiplicity : int
            This option sets the ``spin_multiplicity`` of the reactive complex.
            In case this is not given the ``spin_multiplicity`` of the initial
            structure or minimal possible spin of the two initial structures is
            used.
        rc_molecular_charge : int
            This option sets the ``molecular_charge`` of the reactive complex.
            In case this is not given the ``molecular_charge`` of the initial
            structure or sum of the charges of the initial structures is used.
            Note: If you set the ``rc_molecular_charge`` to a value different
            from the sum of the start structures charges the possibly resulting
            elementary steps will never be balanced but include removal or
            addition of electrons.
        rc_minimal_spin_multiplicity : bool
            True: The total spin multiplicity in a bimolecular reaction is
            based on the assumption of total spin recombination (s + t = s; t + t = s; d + s = d; d + t = d)
            False: No spin recombination is assumed (s + t = t; t + t = quin; d + s = d; d + t = quar)
            (default: False)

        The following options are available for the analysis of the single points of the optimized supersystems:

        expect_charge_separation : bool
            If true, fragment charges are no longer determined by rounding, i.e, if a product
            consists of multiple molecules (according to its graph), the charges are determined initially by rounding.
            However, then the residual (the difference of the integrated charge to the rounded one) is checked against
            <charge_separation_threshold>. If this residual exceeds the charge separation threshold, the charge is
            increased/lowered by one according to its sign. This is especially useful if a clear charge separation only
            occurs upon separation of the molecules which is often the case for DFT-based descriptions of the electronic
            structure.
            (default: False)
        charge_separation_threshold : float
            The threshold for the charge separation (vide supra).
            (default: 0.4)

        These additional settings are recognized:

        imaginary_wavenumber_threshold : float
            Threshold value in inverse centimeters below which a wavenumber
            is considered as imaginary when the transition state is analyzed.
            Negative numbers are interpreted as imaginary. (default: 0.0)
        allow_exhaustive_product_decomposition : bool
            Whether to allow the decomposition of the new products of a complex into further sub-products,
            e.g. of the complex A+B, the product B further decomposes during the optimization of new products
            to C and D.
            Might ignore possible elementary steps (the formation of B from C and D),
            hence the option should be activated with care.
            (default: False)
        store_full_mep : bool
            Whether all individual structures of the IRC and IRCOPT should be saved and attached to the ElementaryStep.
        always_add_barrierless_step_for_reactive_complex : bool
            Add a barrierless reaction for the flask formation of two compounds regardless of their complexation energy.
        """

    @classmethod
    def generated_data_docstring(cls) -> str:
        return super().generated_data_docstring() + """
          If successful (technically and chemically) the following data will be
          generated and added to the database:

          Elementary Steps
            If found, a single new elementary step with the associated transition
            state will be added to the database.

          Structures
            The transition state (TS) and also the separated products will be added
            to the database.

          Properties
            The ``hessian`` (``DenseMatrixProperty``), ``frequencies``
            (``VectorProperty``), ``normal_modes`` (``DenseMatrixProperty``),
            ``gibbs_energy_correction`` (``NumberProperty``) and
            ``gibbs_free_energy`` (``NumberProperty``) of the TS will be
            provided. The ``electronic_energy`` associated with the TS structure and
            each of the products will be added to the database.\n
        """

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "molassembler", "readuct", "utils"]

    def clear(self) -> None:
        self.systems = {}
        super().clear()

    @is_configured
    @requires("database")
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
        settings_manager, program_helper : Tuple[SettingsManager, Union[ProgramHelper, None]]
            A database property holding bond orders.
        """
        # preprocessing of structure
        self.ref_structure = self.check_structures()
        settings_manager, program_helper = self.create_helpers(self.ref_structure)  # type: ignore

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
                self.get_system(self.rc_key), self._calculation.get_settings())

        # Calculate bond orders and graph of reactive complex and compare to database graph of start structures
        reactive_complex_graph, self.systems = self.make_graph_from_calc(self.systems, self.rc_key)
        if not masm.JsonSerialization.equal_molecules(reactive_complex_graph, self.start_graph):
            print("Reactive complex graph differs from combined start structure graphs.")
            self.start_graph = reactive_complex_graph
        return settings_manager, program_helper

    @is_configured
    @requires("database")
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
        start_structures : List[db.ID]
            If given, this structure id list is used instead of the list given in self._calculation.get_structures().

        Returns
        -------
        ref_structure : db.Structure (Scine::Database::Structure)
            The largest structure of the calculation.
        """
        ref_id = None
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
                    break
            else:
                self.raise_named_exception(
                    "Could not identify adsorption structure in calculation with more than 2 structures."
                )
        else:
            self.raise_named_exception(
                "Reactive complexes built from more than 2 structures are not supported."
            )
        if ref_id is None:
            self.raise_named_exception(
                "Could not identify a reference structure in the calculation."
            )
            raise RuntimeError("Unreachable")  # for type checking

        return db.Structure(ref_id, self._structures)

    @requires("database")
    def save_initial_graphs_and_charges(self, settings_manager: SettingsManager, structures: List[db.Structure]) \
            -> None:
        """
        Save the graphs and charges of the reactants.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        settings_manager : SettingsManager
            The settings manager for the calculation.
        structures : List[scine_database.Structure]
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

    @staticmethod
    def _decision_list_from_structure(structure: db.Structure) -> Optional[str]:
        """
        Retrieve masm_decision_list from a database structure.
        Returns ``None`` if none present.

        Parameters
        ----------
        structure : db.Structure

        Returns
        -------
        masm_decision_list : Optional[str]
        """
        if not structure.has_graph("masm_decision_list"):
            return None
        return structure.get_graph("masm_decision_list")

    @is_configured
    @requires("database")
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
        settings_manager : SettingsManager
            The settings_manager in which the charge and multiplicity of the new atoms are set.

        Returns
        -------
        reactive_complex : utils.AtomCollection (Scine::Utilities::AtomCollection)
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
        raise RuntimeError("Unreachable")  # for type checking

    @requires("utilities")
    @is_configured
    def determine_pes_of_rc(self, settings_manager: SettingsManager, s0: db.Structure,
                            s1: Optional[db.Structure] = None) -> None:
        """
        Set charge and spin multiplicity within the settings_manager based on the reaction type (uni- vs. bimolecular)
        and the given settings for the reactive complex.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        settings_manager : SettingsManager
            The settings_manager in which the charge and multiplicity of the new atoms are set.
        s0 : db.Structure (Scine::Database::Structure)
            A structure of the configured calculation
        s1 : Union[db.Structure, None]
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
        coord1 : np.ndarray of shape (n,3)
            The coordinates of the first molecule
        coord2 : np.ndarray of shape (m,3)
            The coordinates of the second molecule

        Returns
        -------
        coord : np.ndarray of shape (n+m, 3)
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
        ----------
        name : str
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

    @requires("utilities")
    def n_imag_frequencies(self, name: str) -> int:
        """
        A helper function to count the number of imaginary frequencies based on the threshold in the settings.
        Does not carry out safety checks.

        Notes
        -----
        * May throw exception (system not present)

        Parameters
        ----------
        name : str
            The name of the system which holds Hessian results.
        """
        calc = self.systems.get(name)
        if calc is None:
            self.raise_named_exception(f"System '{name}' not found in systems.")
            return -1  # only for linter
        atoms = calc.structure
        hessian = calc.get_results().hessian
        if hessian is None:
            self.raise_named_exception(f"No Hessian found for system '{name}'.")
            return -1  # only for linter
        modes_container = utils.normal_modes.calculate(hessian, atoms)
        wavenumbers = modes_container.get_wave_numbers()

        return np.count_nonzero(np.array(wavenumbers) < self.settings[self.job_key]["imaginary_wavenumber_threshold"])

    @requires("utilities")
    @is_configured
    def get_graph_charges_multiplicities(self, name: str, total_charge: int,
                                         total_system_name: Optional[str] = None, split_index: Optional[int] = None) \
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
        name : str
            Index into systems dictionary to calculate bond orders for
        total_charge : str
            The charge of the system
        total_system_name : str
            The name of the total system which can be specified in case this method is called for a partial system.
            This can enable to assign the indices of the total system to the indices of the partial system.
        split_index : int
            The index of the system in the total system which is split. This is used to assign the indices of the total
            system to the indices of the partial system. Both total_system_name and split_index must be specified or
            neither must be specified.

        Returns
        -------
        ordered_structures : List[utils.AtomCollection]
            List of atom collections corresponding to the split molecules.
        graph_string : str
            Sorted molassembler cbor graphs separated by semicolons.
        charges : List[int]
            Charges of the molecules.
        multiplicities : List[int]
            Multiplicities of the molecules, total multiplicity before split influences these returned values based
            on a buff spread over all split structures, these values have to be checked with spin propensity checks
        decision_lists : List[str]
            Molassembler decision lists for free dihedrals
        """
        from scine_puffin.utilities.reaction_transfer_helper import ReactionTransferHelper

        all_surface_indices = self.surface_indices_all_structures()
        if total_system_name is None:
            surface_indices: Union[Set[int], List[int]] = all_surface_indices
        elif total_system_name not in self._component_maps:
            self.raise_named_exception(f"Total system name '{total_system_name}' not found in component maps")
            return [utils.AtomCollection()], "", [], [], []  # For type checking
        elif split_index is None:
            self.raise_named_exception(f"Split index must be given, "
                                       f"if total system name '{total_system_name}' is specified")
            return [utils.AtomCollection()], "", [], [], []  # For type checking
        else:
            split_surfaces_indices = \
                ReactionTransferHelper.map_total_indices_to_split_structure_indices(
                    all_surface_indices, self._component_maps[total_system_name])
            surface_indices = split_surfaces_indices[split_index]

        masm_results, self.systems = self.make_masm_result_from_calc(self.systems, name, surface_indices)

        split_structures = masm_results.component_map.apply(self.get_system(name).structure)
        decision_lists = [masm_helper.get_decision_list_from_molecule(m, a)
                          for m, a in zip(masm_results.molecules, split_structures)]

        # Get cbor graphs
        graphs = []
        for molecule in masm_results.molecules:
            graphs.append(masm_helper.get_cbor_graph_from_molecule(molecule))

        # Determine partial charges, charges per molecules and number of electrons per molecule
        bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, name, surface_indices)
        calc = self.get_system(name)
        partial_charges = calc.get_results().atomic_charges
        if partial_charges is None:
            self.systems, success = readuct.run_single_point_task(
                self.systems, [name], require_charges=True
            )
            self.throw_if_not_successful(
                success, self.systems, [name], ["energy", "atomic_charges"]
            )
            partial_charges = calc.get_results().atomic_charges
            # TODO replace with property setter if we have one in utils, this does not work
            calc.get_results().bond_orders = bond_orders

        charges, n_electrons, _ = self._integrate_charges(masm_results.component_map, partial_charges,
                                                          split_structures, total_charge)

        # Assign multiplicities where we try to spread the buff
        # (i.e. multiplicity difference before to singlet / duplet multiplicity)
        # --> if before 3 -> give one structure (largest) triplet, before 5 --> give each a triplet
        # this ensures that the spin propensity checks later can cover as much as possible
        # this should work with any multiplicity and any number of split structures
        multiplicity_before = self.get_multiplicity(self.get_system(name))
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
    def _custom_round(number: float, threshold: float = 0.5) -> Tuple[float, bool]:
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
        bool
            True if the number was rounded up, False if it was rounded down.
        """
        sign = np.copysign(1.0, number)
        number = abs(number)
        delta = number - np.trunc(number)
        if delta < threshold:
            return np.trunc(number) * sign, False
        else:
            return (np.trunc(number) + 1) * sign, True

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
            summed_partial_charges: List[float],
            changed_charge_indices: Union[List[int], None] = None) -> List[int]:
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
        changed_charge_indices : List[int]
            List of atom indices where the charge was changed.

        Returns
        -------
            charge_guess : List[float]
                The updated list of guessed charges where the sum equals the total charge of the supersystem.
        """
        residual = np.ma.array(self._calculate_residual(summed_partial_charges, charge_guess), mask=False)

        if self.settings[self.single_point_key]["expect_charge_separation"] and changed_charge_indices is not None:
            for i in changed_charge_indices:
                residual.mask[i] = True
        while sum(charge_guess) != total_charge:
            charge_diff = sum(charge_guess) - total_charge
            # too many electrons, add a charge
            if charge_diff < 0.0:
                # Add one charge to selection
                charge_guess[np.argmax(residual)] += 1
            # too little electrons, remove a charge
            else:
                # Subtract one charge from selection
                charge_guess[np.argmin(residual)] -= 1
            # Update residual
            residual = np.ma.array(self._calculate_residual(summed_partial_charges, charge_guess), mask=False)
            if self.settings[self.single_point_key]["expect_charge_separation"] and changed_charge_indices is not None:
                for i in changed_charge_indices:
                    residual.mask[i] = True
        # return updated charge guess
        return charge_guess

    @requires("utilities")
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
        # Update charges to charge guess, only containing ints
        changed_charge_indices: List[int] = []
        for i in range(len(split_structures)):
            if not self.settings[self.single_point_key]["expect_charge_separation"]:
                charges[i], round_up = self._custom_round(charges[i], 0.5)
                charges[i] = int(charges[i])
            else:
                charges[i], round_up = self._custom_round(
                    charges[i], self.settings[self.single_point_key]["charge_separation_threshold"])
                charges[i] = int(charges[i])
            if round_up:
                changed_charge_indices.append(i)

        # Check and re-distribute if necessary
        updated_charges = self._distribute_charge(total_charge, charges, summed_partial_charges, changed_charge_indices)
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
        rc_opt_graph : Optional[str]
            Sorted molassembler cbor graphs separated by semicolons of the
            reaction product if there was any.
        rc_opt_decision_lists : Optional[List[str]]
            Molassembler decision lists for free dihedrals of the reaction
            product if there was any.
        """
        # Check for barrierless reaction leading to new graphs
        if self.rc_opt_system_name not in self.systems:  # Skip if already done
            print("Running Reactive Complex Optimization")
            print("Settings:")
            print(self.settings[self.rc_opt_system_name], "\n")
            self.systems, success = self.observed_readuct_call(
                SubTaskToReaductCall.RCOPT, self.systems, [self.rc_key], **self.settings[self.rc_opt_system_name]
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
        name : str
            Index into systems dictionary to retrieve output for

        Returns
        -------
        outputs : List[str]
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

    def analyze_side(self, input_name: str, initial_charge: int, opt_name: str,
                     calculator_settings: utils.Settings) -> Union[Tuple[str, List[int], List[str], List[str]],
                                                                   Tuple[None, None, None, None]]:
        print(opt_name.capitalize() + " Bond Orders")
        (
            structures,
            full_graph,
            original_charges,
            original_multiplicities,
            original_decision_lists,
        ) = self.get_graph_charges_multiplicities(input_name, initial_charge)
        # Compress info and find unique structures
        full_info = [(graph, charge, multiplicity, decision_list)
                     for graph, charge, multiplicity, decision_list in zip(full_graph.split(';'),
                                                                           original_charges,
                                                                           original_multiplicities,
                                                                           original_decision_lists)]

        # Map unique info to indices
        unique_map: Dict[Tuple[str, int, int, str], List[int]] = {}
        min_indices: List[int] = []
        # Loop over molecule infos and collect indices of identical structures
        for i, info in enumerate(full_info):
            if info not in unique_map:
                unique_map[info] = []
            unique_map[info].append(i)
        # Extract min indices of each unique structure
        min_indices = [min(indices) for indices in unique_map.values()]

        # Optimize unique structures
        unique_names, self.systems = self.optimize_structures(opt_name, self.systems,
                                                              [structures[i] for i in min_indices],
                                                              [original_charges[i] for i in min_indices],
                                                              [original_multiplicities[i] for i in min_indices],
                                                              calculator_settings)
        # Map back to initial structures
        output_names = [""] * len(structures)
        unique_result: Dict[Tuple[str, int, int, str], str] = {}
        for min_index, name in zip(min_indices, unique_names):
            unique_result[full_info[min_index]] = name
            # Get indices
            for index in unique_map[full_info[min_index]]:
                output_names[index] = name

        output_graphs = []
        output_decision_lists = []
        split_molecules = []
        # Analyze separated molecules
        for i, (name, charge) in enumerate(zip(output_names, original_charges)):
            (tmp_structure, tmp_graph, tmp_charge,
             tmp_multiplicity, tmp_decision_list
             ) = self.get_graph_charges_multiplicities(name, charge, total_system_name=input_name, split_index=i)
            if len(tmp_structure) > 1:
                if not self.settings[self.job_key]['allow_exhaustive_product_decomposition']:
                    self._calculation.set_comment(self.name + ": " + opt_name.capitalize() +
                                                  ": IRC results keep decomposing (more than once).")
                    return None, None, None, None
                split_molecules.append((name, i,
                                        tmp_structure, tmp_graph.split(";"),
                                        tmp_charge, tmp_multiplicity, tmp_decision_list))
                continue

            output_graphs += tmp_graph.split(';')
            output_decision_lists += tmp_decision_list

        # Re-optimize split molecules, until they are no longer split
        # NOTE: If 'allow_exhaustive_product_decomposition' is enabled and a split product
        # decomposes during optimization, the new products are re-optimized until they are no longer decomposing.
        # The resulting barrierless elementary step will only connect the IRC end of the regular elementary step
        # with the final products and not save the intermediate sub-products.
        # E.g. for a complex A+B, the product B further decomposes during the optimization of new products to C and D.
        # The possible elementary step of C + D <-> B will not be checked for or captured with this option.
        # Instead the barrierless elementary step A+B <-> C + D is written as a result.
        if self.settings[self.job_key]["allow_exhaustive_product_decomposition"] and\
           len(split_molecules) > 0:
            indices_to_remove: List[int] = []
            while (len(split_molecules) > 0):
                # Get first entry
                (org_name, org_index,
                 structures, graphs, split_charges, multiplicities, decision_lists) = split_molecules[0]
                # Look if we already have optimized this structure
                new_structure_indices: List[int] = []
                stored_names: List[str] = []
                for i, molecule_key in enumerate(zip(graphs, split_charges, multiplicities, decision_lists)):
                    if molecule_key in unique_result.keys():
                        stored_names.append(unique_result[molecule_key])
                    else:
                        new_structure_indices.append(i)
                # Optimize unknown structures
                new_names, self.systems = self.optimize_structures(org_name, self.systems,
                                                                   [structures[i] for i in new_structure_indices],
                                                                   [split_charges[i] for i in new_structure_indices],
                                                                   [multiplicities[i] for i in new_structure_indices],
                                                                   calculator_settings)
                # Combine new and stored names
                new_names += stored_names
                # Remove entry due to split
                indices_to_remove.append(org_index)
                # Check new structures
                for i, (name, charge) in enumerate(zip(new_names, split_charges)):
                    (tmp_structure, tmp_graph, tmp_charge,
                     tmp_multiplicity, tmp_decision_list
                     ) = self.get_graph_charges_multiplicities(name, charge, total_system_name=input_name,
                                                               split_index=org_index)
                    if len(tmp_structure) > 1:
                        # Repeat re-optimization
                        split_molecules.append((name, org_index,
                                                tmp_structure, tmp_graph.split(";"),
                                                tmp_charge, tmp_multiplicity, tmp_decision_list))
                    else:
                        # Append new entries at the end
                        output_names.append(name)
                        original_charges.append(charge)
                        original_multiplicities += tmp_multiplicity
                        output_graphs += tmp_graph.split(';')
                        output_decision_lists += tmp_decision_list
                # Remove from split molecules
                split_molecules.pop(0)

            # Remove the name, the charge and multiplicity of original molecule
            output_names = [name for i, name in enumerate(output_names) if i not in indices_to_remove]
            original_charges = [charge for i, charge in enumerate(original_charges) if i not in indices_to_remove]
            original_multiplicities = [multiplicity for i, multiplicity in enumerate(original_multiplicities)
                                       if i not in indices_to_remove]

        # Sort everything again, based on graphs
        output_graphs, original_charges, original_multiplicities, output_decision_lists, \
            output_names = (
                list(start_val) for start_val in zip(*sorted(zip(
                    output_graphs,
                    original_charges,
                    original_multiplicities,
                    output_decision_lists,
                    output_names)))
            )
        output_graph = ';'.join(output_graphs)

        return output_graph, original_charges, output_decision_lists, output_names

    def _determine_complexation_energy(self, input_name: str, molecule_names: List[str]) -> float:
        complexation_energy = 0.0
        for name in molecule_names:
            complexation_energy -= self.get_energy(self.get_system(name))
        complexation_energy += self.get_energy(self.get_system(input_name))
        print("Complexation Energy:", complexation_energy * utils.KJPERMOL_PER_HARTREE, "kJ/mol")
        return complexation_energy

    @requires("database")
    @is_configured
    def irc_sanity_checks_and_analyze_sides(
            self,
            initial_charge: int,
            check_charges: bool,
            inputs: List[str],
            calculator_settings: utils.Settings) -> Union[Tuple[List[str], Optional[List[str]]], Tuple[None, None]]:
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
        initial_charge : int
            The charge of the reactive complex
        check_charges : bool
            Whether the charges must be checked
        inputs : List[str]
            The name of the IRC outputs to use as inputs
        calculator_settings : utils.Settings
            The general settings for the Scine calculator. Charge and spin multiplicity will be overwritten.

        Returns
        -------
        product_names : Optional[List[str]]
            A list of the access keys to the products in the system map.
        start_names : Optional[List[str]]
            A list of the access keys to the starting materials in the system map.
        """
        if len(inputs) != 2:
            self.raise_named_exception(
                "Requires to pass 2 systems to the IRC sanity check"
            )
        # All lists ordered according to graph - charges - multiplicities with decreasing priority
        # Get graphs, charges and minimal multiplicities of split forward and backward structures
        (
            forward_graph,
            forward_charges,
            forward_decision_lists,
            forward_names
        ) = self.analyze_side(inputs[0], initial_charge, "forward",
                              calculator_settings)
        if any(f_info is None for f_info in [forward_graph, forward_charges, forward_decision_lists, forward_names]):
            # NOTE: Maybe still save TS for restart here
            return None, None
        assert forward_graph
        assert forward_names
        (
            backward_graph,
            backward_charges,
            backward_decision_lists,
            backward_names
        ) = self.analyze_side(inputs[1], initial_charge, "backward",
                              calculator_settings)
        if any(b_info is None for b_info in [backward_graph, backward_charges,
                                             backward_decision_lists, backward_names]):
            # NOTE: Maybe still save TS for restart here
            return None, None
        assert backward_graph
        assert backward_names
        print("Forward charges: " + str(forward_charges))
        print("Backward charges: " + str(backward_charges))

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

        def no_match() -> Tuple[List[str], bool]:
            print(self.name + ": No IRC structure matches starting structure.")
            # Step direction must be forward to guarantee working logic downstream
            self.step_direction = "forward"
            self.products_component_map = self._component_maps[inputs[0]]
            # Trigger to set 'start_names' as 'backward_names'
            self.no_irc_structure_matches_start = True
            return forward_names, False

        compare_decision_lists = True
        if not self.start_graph:
            product_names, compare_decision_lists = no_match()
        # Do not expect matching charges if reactive complex charge differs from sum of start structure charges
        elif masm.JsonSerialization.equal_molecules(forward_graph, self.start_graph) \
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
                product_names, compare_decision_lists = no_match()
        else:
            product_names, compare_decision_lists = no_match()

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
            assert new_decision_lists
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
        if self.settings[self.job_key]["always_add_barrierless_step_for_reactive_complex"]:
            add_forward_step = len(forward_names) > 1
            add_backward_step = len(backward_names) > 1
        else:
            # # # Forward complexation
            forward_complexation_energy = self._determine_complexation_energy(inputs[0], forward_names)
            add_forward_step = bool(forward_complexation_energy < self.complexation_criterion)

            # # # Back complexation
            backward_complexation_energy = self._determine_complexation_energy(inputs[1], backward_names)
            add_backward_step = bool(backward_complexation_energy < self.complexation_criterion)
        # Decide whether to add complexation steps
        if add_forward_step:
            if self.step_direction == "backward":
                self.lhs_complexation = True
            else:
                self.rhs_complexation = True
        if add_backward_step:
            if self.step_direction == "backward":
                self.rhs_complexation = True
            else:
                self.lhs_complexation = True

        return product_names, start_names

    @is_configured
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

    def read_irc_and_irc_opt_trajectories(self, tsopt_task_name: str, irc_task_name: str) \
            -> Tuple[utils.MolecularTrajectory, int]:
        """
        Reads the IRC and IRC optimization trajectories in the correct order to get a cohesive minimum energy path.

        Parameters
        ----------
        tsopt_task_name : str
            The name of the transition state optimization task.
        irc_task_name : str
            The name of the IRC task.

        Raises
        ------
        RuntimeError
            If the IRC or IRC optimization trajectory files are missing.
        RuntimeError
            If the step_direction is not "forward" or "backward".

        Returns
        -------
        Tuple[utils.MolecularTrajectory, int]
            The IRC trajectory and the index of the transition state in the IRC trajectory.
        """
        if self.step_direction == "forward":
            forward_dir = self.output(irc_task_name)[0]
            backward_dir = self.output(irc_task_name)[1]
        elif self.step_direction == "backward":
            forward_dir = self.output(irc_task_name)[1]
            backward_dir = self.output(irc_task_name)[0]
        else:
            self.raise_named_exception("Could not determine elementary step direction.")
            raise RuntimeError("Unreachable")  # just for linter

        ts_calc = self.get_system(self.output(tsopt_task_name)[0])
        ts_energy = ts_calc.get_results().energy
        if ts_energy is None:
            self.raise_named_exception("Missing energy for transition state to construct spline.")
            raise RuntimeError("Unreachable")  # just for linter

        mep = utils.MolecularTrajectory(ts_calc.structure.elements, 0.0)

        def add_file_to_mep(file_name: str, read_in_reversed: bool = False) -> None:
            trj = utils.io.read_trajectory(utils.io.TrajectoryFormat.Xyz, file_name)
            energies = trj.get_energies()
            if read_in_reversed:
                for pos, e in zip(reversed(trj), reversed(energies)):
                    mep.push_back(pos, e)
            else:
                for pos, e in zip(trj, energies):
                    mep.push_back(pos, e)

        def file_name_from_dir_name(dir_name: str, is_irc: bool) -> str:
            if is_irc:
                return os.path.join(
                    self.work_dir, f"{dir_name}", f"{dir_name}.irc.{dir_name.split('_')[-1]}.trj.xyz"
                )
            return os.path.join(
                self.work_dir, f"{dir_name}", f"{dir_name}.opt.trj.xyz"
            )

        # we now combine ircopt backward - irc backward - ts - irc forward - ircopt forward
        # and we reverse the backward trajectories
        fpath = file_name_from_dir_name(backward_dir, is_irc=False)
        if os.path.isfile(fpath):
            add_file_to_mep(fpath, read_in_reversed=True)

        fpath = file_name_from_dir_name(backward_dir, is_irc=True)
        if os.path.isfile(fpath):
            add_file_to_mep(fpath, read_in_reversed=True)
        else:
            self.raise_named_exception(f"Missing IRC trajectory file: {fpath}")

        ts_index = mep.size()
        mep.push_back(ts_calc.structure.positions, ts_energy)

        fpath = file_name_from_dir_name(forward_dir, is_irc=True)
        if os.path.isfile(fpath):
            add_file_to_mep(fpath)
        else:
            self.raise_named_exception(f"Missing IRC trajectory file: {fpath}")

        fpath = file_name_from_dir_name(forward_dir, is_irc=False)
        if os.path.isfile(fpath):
            add_file_to_mep(fpath)

        return mep, ts_index

    @requires("utilities")
    def generate_spline(
            self, trajectory: utils.MolecularTrajectory, ts_index: int, n_fit_points: int = 23, degree: int = 3
    ) -> utils.bsplines.TrajectorySpline:
        """
        Using the transition state, IRC and IRC optimization outputs generates
        a spline that describes the trajectory of the elementary step, fitting
        both atom positions and energy.
        Removes all structures that have a higher energy than the transition state.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        trajectory : utils.MolecularTrajectory
            The trajectory of the elementary step.
        ts_index : int
            The index of the transition state in the trajectory.
        n_fit_points : str
            Number of fit points to use in the spline compression.
        degree : str
            Fit degree to use in the spline generation.

        Returns
        -------
        spline : utils.bsplines.TrajectorySpline
            The fitted spline of the elementary step trajectory.
        """

        if ts_index >= trajectory.size():
            self.raise_named_exception(f"Transition state index {ts_index} is out of bounds "
                                       f"for trajectory of size {trajectory.size()}.")

        energies = trajectory.get_energies()
        ts_energy = energies[ts_index]
        elements = trajectory.elements
        rpi = utils.bsplines.ReactionProfileInterpolation()
        for index, (pos, e) in enumerate(zip(trajectory, energies)):
            if index != ts_index and e > ts_energy:
                continue
            rpi.append_structure(utils.AtomCollection(elements, pos), e, is_the_transition_state=(index == ts_index))

        return rpi.spline(n_fit_points, degree)

    @requires("database")
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
        start_structure_names : List[str]
            The names of the start structure names in the system map.
        program_helper : Union[ProgramHelper, None]
            The ProgramHelper which might also want to do postprocessing
        tsopt_task_name : str
            The name of the task where the TS was output
        start_structures : Optional[List[db.ID]]
            Optional list of the starting structure ids. If no list is given. The input
            structures of the calculation are used.

        Returns
        -------
        start_structure_ids : List[scine_database.ID]
            A list of the database IDs of the start structures.
        """
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
        if not all(model.equal_without_periodic_boundary_check(start_model) for model in models):
            self.raise_named_exception("React job with mixed model input structures")

        # Update model to make sure there are no 'any' values left
        update_model(
            self.get_system(self.output(tsopt_task_name)[0]),
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
                aggregate: Union[db.Flask, db.Compound]
                if ';' in initial_graph:
                    aggregate = db.Flask(aggregate_id, self._flasks)
                else:
                    aggregate = db.Compound(aggregate_id, self._compounds)
                existing_structures = aggregate.get_structures()
                for existing_structure_id in existing_structures:
                    existing_structure = db.Structure(existing_structure_id, self._structures)
                    if existing_structure.get_label() in \
                            [db.Label.DUPLICATE, db.Label.MINIMUM_GUESS, db.Label.USER_GUESS,
                             db.Label.SURFACE_GUESS, db.Label.SURFACE_ADSORPTION_GUESS]:
                        continue
                    if not start_model.equal_without_periodic_boundary_check(existing_structure.get_model()):
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
            self.store_energy(self.get_system(name), new_structure)
            self.store_bond_orders(bond_orders, new_structure)
            self.add_graph(new_structure, bond_orders, surface_indices)
            start_structure_ids.append(new_structure.id())
        return start_structure_ids

    @requires("database")
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
        complex_name : str
            The name of the complex system in the systems map
        program_helper : Union[ProgramHelper, None]
            The ProgramHelper which might also want to do postprocessing
        Returns
        -------
        complex_structure_id : db.ID
            The id of the added structure
        """
        complex_system = self.systems[complex_name]
        complex_graph, self.systems = self.make_graph_from_calc(self.systems, complex_name)
        structure_label = self._determine_new_label_based_on_graph(complex_system, complex_graph)
        complex_structure = self.create_new_structure(complex_system, structure_label)
        bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, complex_name)
        if self.ref_structure is not None:
            self.transfer_properties(self.ref_structure, complex_structure)
        self.store_energy(self.get_system(complex_name), complex_structure)
        self.store_bond_orders(bond_orders, complex_structure)
        self.add_graph(complex_structure, bond_orders)
        if program_helper is not None and self.ref_structure is not None:
            program_helper.calculation_postprocessing(self._calculation, self.ref_structure, complex_structure)
        return complex_structure.id()

    @requires("database")
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
        product_names : List[str]
            A list of the access keys to the products in the system map.
        program_helper : Union[ProgramHelper, None]
            The ProgramHelper which might also want to do postprocessing
        tsopt_task_name : str
            The name of the task where the TS was output
        reactant_structure_ids : List[scine_database.ID]
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
            self.get_system(self.output(tsopt_task_name)[0]),
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
            not self.settings[self.propensity_key]["check_for_unimolecular_reaction"]
        for i, (label, product) in enumerate(zip(new_labels, product_names)):
            surface_indices = split_surfaces_indices[i]
            new_structure = self._store_structure_with_propensity_check(product, self.systems, label,
                                                                        enforce_to_save_base_name=single_molecule_mode,
                                                                        surface_indices=surface_indices)
            if program_helper is not None and self.ref_structure is not None:
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
        try:
            trajectory, ts_index = self.read_irc_and_irc_opt_trajectories(tsopt_task_name, "irc")
            spline = self.generate_spline(trajectory, ts_index)
            new_step.set_spline(spline)
            """ Save Reaction Path """
            if self.settings[self.job_key]["store_full_mep"]:
                charge = self.get_charge(ts_calc)
                multiplicity = self.get_multiplicity(ts_calc)
                model = self._calculation.get_model()
                self.save_mep_in_db(new_step, trajectory, ts_index, charge, multiplicity, model)
        except BaseException as e:
            # If the spline generation crashes we need to continue,
            # otherwise the database is in a broken state.
            # For now, just do not add a spline.
            print("Failed to generate spline interpolation for the reaction, continuing without adding the spline to"
                  " the database. The error was:\n", e)
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
        self.systems = self._add_propensity_systems(ts_name, self.systems)
        self.systems = self._spin_propensity_single_points(
            ts_name,
            self.systems,
            "Failed all spin propensity single points for TS, "
            "which means we could not recalculate the TS system. "
            "This points to a SCINE calculator error."
        )
        new_ts = self._store_structure_with_propensity_check(ts_name, self.systems, ts_label,
                                                             enforce_to_save_base_name=True)
        if self.ref_structure is not None:
            self.transfer_properties(self.ref_structure, new_ts)
        ts_calc = self.get_system(ts_name)
        if ts_label == db.Label.TS_OPTIMIZED or ts_calc.get_results().hessian is not None:
            self.store_hessian_data(ts_calc, new_ts)
        if program_helper is not None and self.ref_structure is not None:
            program_helper.calculation_postprocessing(self._calculation, self.ref_structure, new_ts)
        return ts_calc, new_ts

    @requires("database")
    def save_mep_in_db(self, elementary_step: db.ElementaryStep, trajectory: utils.MolecularTrajectory, ts_index: int,
                       charge: int, multiplicity: int, model: db.Model) \
            -> List[db.ID]:
        """
        Store each point on the MEP as a structure in the database.
        Attaches `electronic_energy` properties for each point.

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        elementary_step : scine_database.ElementaryStep
            The elementary step of which to store the MEP.
        trajectory : utils.MolecularTrajectory
            The trajectory of the elementary step.
        ts_index : int
            The index of the transition state in the trajectory.
        charge : int
            The total charge of the system.
        multiplicity : int
            The spin multiplicity of the system.
        model : scine_database.Model
            The model with which all energies in the elementary Step were
            calculated.
        """

        def generate_structure(atoms: utils.AtomCollection):
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

        structure_ids = []
        elements = trajectory.elements
        for i, pos in enumerate(trajectory):
            if i == ts_index:
                structure_ids.append(elementary_step.get_transition_state())
            else:
                structure_ids.append(generate_structure(utils.AtomCollection(elements, pos)))

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
        structure_id_list : List[db.ID]
            A list structure ids
        labels : List[db.Label]
            The required labels
        """
        return self._label_locations(structure_id_list, labels)[0] is not None

    @requires("database")
    @is_configured
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
        structure_id_list : List[db.ID]
            A list structure ids
        labels : List[db.Label]
            The required labels
        """
        for i, sid in enumerate(structure_id_list):
            structure_label = db.Structure(sid, self._structures).get_label()
            for j, label in enumerate(labels):
                if structure_label == label:
                    return i, j
        return None, None

    @requires("database")
    @is_configured
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
        calculator : Core::Calculator
            The calculator holding the structure
        graph_str : str
            The cbor graph of one or more molecules (separated by ';')
        Returns
        -------
        label : db.Label
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

    @requires("database")
    @is_configured
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
        names : List[str]
            The list of system names of the products in the systems map
        component_map : Optional[List[int]]
            The component map of the given systems, take product_component_map if None

        Returns
        -------
        labels : List[db.Label]
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
        n_product_atoms = sum(len(self.get_system(name).structure) for name in names)
        if any(index >= n_product_atoms for index in surface_indices):
            self.raise_named_exception("Surface indices include invalid numbers for the given products")
        if len(component_map) != n_product_atoms:
            self.raise_named_exception("Invalid product component map for the given products")
        product_surface_atoms = [0] * len(names)
        for index in surface_indices:
            product_surface_atoms[component_map[index]] += 1
        # do not categorize if only single surface atom, but assume this is a transfer from the surface to the product
        thresh = self.connectivity_settings["n_surface_atom_threshold"]
        return [db.Label.SURFACE_OPTIMIZED if n > thresh else db.Label.MINIMUM_OPTIMIZED for n in product_surface_atoms]

    @requires("database")
    @is_configured
    def _hess_irc_ircopt(self, ts_system_name: str, settings_manager: SettingsManager) \
            -> Tuple[List[str], Optional[List[str]]]:
        """
        Takes an optimized TS and carries out:
        * Hessian calculation and check for valid TS
        * IRC calculation
        * random displacement of IRC points
        * Optimization with faster converging optimizer than Steepest Descent to arrive at true minima

        Parameters
        ----------
        ts_system_name : str
            The name of the system holding the optimized TS
        settings_manager : SettingsManager
            The settings manager

        Returns
        -------
        product_names : List[str]
            The names of the products
        start_names : Optional[List[str]]
            The names of the start structures, if different to the structures of the react job
        """
        inputs = [ts_system_name]
        """ TS HESSIAN """
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
            SubTaskToReaductCall.IRC, self.systems, inputs, **self.settings["irc"])

        """ IRC OPT JOB """
        # Run a small energy minimization after initial IRC
        inputs = self.output("irc")
        print("IRC Optimization Settings:")
        print(self.settings["ircopt"], "\n")
        for i in inputs:
            calc = self.get_system(i)
            atoms = calc.structure
            self.random_displace_atoms(atoms)
            calc.positions = atoms.positions
        self.systems, success = self.observed_readuct_call(
            SubTaskToReaductCall.IRCOPT, self.systems, [inputs[0]], **self.settings["ircopt"])
        self.systems, success = self.observed_readuct_call(
            SubTaskToReaductCall.IRCOPT, self.systems, [inputs[1]], **self.settings["ircopt"])

        """ Check whether we have a valid IRC """
        initial_charge = settings_manager.calculator_settings[utils.settings_names.molecular_charge]
        product_names, start_names = self.irc_sanity_checks_and_analyze_sides(
            initial_charge, self.check_charges, inputs, settings_manager.calculator_settings)
        if product_names is None:  # IRC did not pass checks, reason has been set as comment, complete job
            self.verify_connection()
            self.capture_raw_output()
            update_model(
                self.get_system(self.output("tsopt")[0]),
                self._calculation,
                self.config,
            )
            raise breakable.Break
        return product_names, start_names

    @requires("database")
    @is_configured
    def _tsopt_hess_irc_ircopt(self, tsguess_system_name: str, settings_manager: SettingsManager) \
            -> Tuple[List[str], Optional[List[str]]]:
        """
        Takes a TS guess and carries out:
        * TS optimization
        * Hessian calculation and check for valid TS
        * IRC calculation
        * random displacement of IRC points
        * Optimization with faster converging optimizer than Steepest Descent to arrive at true minima (default: BFGS),
          however, one can select in principle any optimizer from those available in SCINE,
          which could also be a Steepest Descent optimizer to calculate an IRC exactly as defined
          (although at much increased computational costs due to its slow convergence)

        Parameters
        ----------
        tsguess_system_name : str
            The name of the system holding the TS guess
        settings_manager : SettingsManager
            The settings manager

        Returns
        -------
        product_names : List[str]
            The names of the products
        start_names : Optional[List[str]]
            The names of the start structures, if different to the structures of the react job
        """
        inputs = [tsguess_system_name]
        self.setup_automatic_mode_selection("tsopt")
        print("TSOpt Settings:")
        print(self.settings["tsopt"], "\n")
        self.systems, success = self.observed_readuct_call(
            SubTaskToReaductCall.TSOPT, self.systems, inputs, **self.settings["tsopt"])
        self.throw_if_not_successful(
            success,
            self.systems,
            self.output("tsopt"),
            ["energy"],
            "TS optimization failed:\n",
        )
        return self._hess_irc_ircopt(self.output("tsopt")[0], settings_manager)

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
        self._postprocessing_with_conformer_handling(product_names, start_names, program_helper)

    def _postprocessing_with_conformer_handling(self, product_names: List[str], start_names: Optional[List],
                                                program_helper: Optional[ProgramHelper]) -> None:
        """
        Stores the new start structures if given otherwise takes the input structures of the calculation and
        then carries out the postprocessing of the reaction.

        Parameters
        ----------
        product_names : List[str]
            The names of the products
        start_names : Optional[List[str]]
            The names of the start structures, if different to the structures of the react job
        program_helper : Optional[ProgramHelper]
            The program helper
        """

        """ Store new starting material conformer(s) """
        if start_names is not None:
            start_structures = self.store_start_structures(
                start_names, program_helper, "tsopt")
        else:
            start_structures = self._calculation.get_structures()

        self.react_postprocessing(product_names, program_helper, "tsopt", start_structures)

    def get_system(self, name: str) -> utils.core.Calculator:
        """
        Get a calculator from the system map by name and ensures the system is present

        Notes
        -----
        * May throw exception

        Parameters
        ----------
        name : str
            The name of the system to get
        """
        return self.get_calc(name, self.systems)
