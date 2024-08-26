# -*- coding: utf-8 -*-
from __future__ import annotations

__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC
from typing import Any, Dict, List, Tuple, Union, Optional, Iterator, Set, TYPE_CHECKING
import sys
from copy import deepcopy

from .job import is_configured
from .scine_job import CalculatorNotPresentException
from .scine_job_with_observers import ScineJobWithObservers
from scine_puffin.utilities.task_to_readuct_call import SubTaskToReaductCall
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utilities = MissingDependency("scine_utilities")
if module_exists("scine_readuct") or TYPE_CHECKING:
    import scine_readuct as readuct
else:
    readuct = MissingDependency("scine_readuct")


class ScinePropensityJob(ScineJobWithObservers, ABC):
    """
    A common interface for all jobs in Puffin that carry out calculations
    with different spin multiplicities for spin propensity checks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "PropensityJob"  # to be overwritten by child
        self.propensity_key = "spin_propensity"
        self.opt_key = "opt"
        # to be extended by child:
        self.settings: Dict[str, Dict[str, Any]] = {
            **self.settings,
            self.propensity_key: {
                "check_for_unimolecular_reaction": True,
                "energy_range_to_save": 200.0,
                "optimize_all": True,
                "energy_range_to_optimize": 500.0,
                "check": 2,
            },
            self.opt_key: {}
        }

    @classmethod
    def optional_settings_doc(cls) -> str:
        return super().optional_settings_doc() + """

        These settings are recognized for calculations checking for spin propensities:

        spin_propensity_check : int
            The range to check for possible multiplicities for products. A value
            of 2 (default) will check triplet and quintet for a singlet
            and will check singlet, quintet und septet for triplet.
        spin_propensity_energy_range_to_save : float
            The energy range in kJ/mol to save structures with different spin multiplicities.
        spin_propensity_energy_range_to_optimize : float
            The energy range in kJ/mol to optimize structures with different spin multiplicities.
        spin_propensity_optimize_all : bool
            If set to True, all spin states will be optimized regardless of their energy.
        spin_propensity_check_for_unimolecular_reaction : bool
            Applies to jobs searching for elementary steps. Determine if spin propensities should be checked even if
            the elementary step is purely unimolecular (reactant and product a single continuous graph.
        """

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "molassembler", "readuct", "utils"]

    @requires("utilities")
    @is_configured
    def optimize_structures(
            self,
            name_stub: str,
            systems: Dict[str, Optional[utils.core.Calculator]],
            structures: List[utils.AtomCollection],
            structure_charges: List[int],
            structure_multiplicities: List[int],
            calculator_settings: utils.Settings,
            stop_on_error: bool = True,
            readuct_task: SubTaskToReaductCall = SubTaskToReaductCall.OPT,
            task_settings_key: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Optional[utils.core.Calculator]]]:
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
        name_stub : str
            The stub for naming of the structures, example: `start` will generate
            systems `start_00`, `start_01`, and so on.
        systems : Dict[str, Optional[utils.core.Calculator]]
            The map of systems
        structures : List[utils.AtomCollection]
            The atoms of the structures in a list.
        structure_charges : List[int]
            The charges of the structures.
        structure_multiplicities : List[int]
            The spin multiplicities of the structures.
        calculator_settings : utils.Settings
            The general settings for the Scine calculator. Charge and spin multiplicity will be overwritten.
        stop_on_error : bool
            If set to False, skip unsuccessful calculations and replace calculator with None
        readuct_task : SubTaskToReaductCall
            The task to perform with readuct, by default SubTaskToReaductCall.OPT
        task_settings_key : Optional[str]
            The key in the settings dictionary to use for the task settings, by default None which will be the opt_key

        Returns
        -------
        product_names : List[str]
            A list of the access keys to the structures in the system map.
        systems : Dict[str, Optional[utils.core.Calculator]]
            The updated map of systems.
        """
        if task_settings_key is None:
            task_settings_key = self.opt_key
        structure_names: List[str] = []
        method_family = self._calculation.get_model().method_family
        # Generate structure systems
        for i, atoms in enumerate(structures):
            name = f"{name_stub}_{i:02d}"
            structure_names.append(name)
            utils.io.write(name + ".xyz", atoms)
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
                systems[name] = new
                systems = self._add_propensity_systems(name, systems)
            except RuntimeError as e:
                if stop_on_error:
                    raise e
                sys.stderr.write(f"{name} cannot be calculated because: {str(e)}")
                systems[name] = None

        print("Product Opt Settings:")
        print(self.settings[self.opt_key], "\n")
        required_properties = ["energy"]
        if not self.connectivity_settings['only_distance_connectivity']:
            required_properties.append("bond_orders")
        # Optimize structures, if they have more than one atom; otherwise run a single point calculation
        for structure_name in structure_names:
            if systems[structure_name] is None:
                continue
            try:
                if not self.settings[self.propensity_key]["check"]:
                    systems, success = readuct.run_single_point_task(
                        systems,
                        [structure_name],
                        require_bond_orders=not self.connectivity_settings['only_distance_connectivity'],
                    )
                    self.throw_if_not_successful(success, systems, [structure_name], required_properties,
                                                 f"{name_stub.capitalize()} single point failed:\n")
                else:
                    systems = self._spin_propensity_single_points(structure_name, systems,
                                                                  f"{name_stub.capitalize()} single point failed:\n")
                if len(self.get_calc(structure_name, systems).structure) > 1:
                    if len(structure_names) == 1 and len(self._calculation.get_structures()) == 1 and \
                            not self.settings[self.propensity_key]["check_for_unimolecular_reaction"]:
                        # optimize only base multiplicity
                        systems = self.observed_readuct_call_with_throw(
                            readuct_task, systems, [structure_name], required_properties,
                            f"{name_stub.capitalize()} optimization failed:\n",
                            **self.settings[task_settings_key]
                        )
                        # still do propensity SP to store close energy multiplicities in DB
                        systems = self._spin_propensity_single_points(
                            structure_name, systems, f"{name_stub.capitalize()} optimization failed:\n")
                    elif self.settings[self.propensity_key]["optimize_all"]:
                        systems = self._spin_propensity_optimizations(
                            structure_name, systems, f"{name_stub.capitalize()} optimization failed:\n",
                            readuct_task, task_settings_key
                        )
                    else:
                        systems = self._limited_spin_propensity_optimization(structure_name, systems, name_stub,
                                                                             required_properties,
                                                                             readuct_task, task_settings_key)
            except RuntimeError as e:
                if stop_on_error:
                    raise e
                sys.stderr.write(f"{structure_name} cannot be calculated because: {str(e)}")
                systems[structure_name] = None
        return structure_names, systems

    @requires("utilities")
    def _add_propensity_systems(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]]) \
            -> Dict[str, Optional[utils.core.Calculator]]:
        """
        Adds clone systems of the given name to the given systems map that have different spin multiplicities.

        Parameters
        ----------
        name : str
            The name of the system to add the propensity systems for.
        systems : Dict[str, Optional[utils.core.Calculator]]
            The map of systems.

        Returns
        -------
        Dict[str, Optional[utils.core.Calculator]]
            The updated systems map.

        Raises
        ------
        RuntimeError
            If the settings object of a calculator was replaced with a broken object
        """

        for shift_name, multiplicity in self._propensity_iterator(name, systems):
            if shift_name == name:
                continue
            shifted_calc = self.get_calc(name, systems).clone()
            systems[shift_name] = shifted_calc
            shifted_calc.delete_results()  # make sure results of clone are empty
            if utils.settings_names.spin_mode in shifted_calc.settings:
                dc = shifted_calc.settings.descriptor_collection
                if utils.settings_names.spin_mode not in dc:
                    self.raise_named_exception(f"{utils.settings_names.spin_mode} not in descriptor collection "
                                               f"of {name} system")
                    raise RuntimeError("Unreachable")  # just for linters
                descriptor = dc[utils.settings_names.spin_mode]
                if isinstance(descriptor, utils.OptionListDescriptor):
                    for suitable in ["unrestricted", "restricted_open_shell", "any"]:
                        if suitable in descriptor.options:
                            shifted_calc.settings[utils.settings_names.spin_mode] = suitable
                            break
                else:
                    shifted_calc.settings[utils.settings_names.spin_mode] = "any"
            shifted_calc.settings[utils.settings_names.spin_multiplicity] = multiplicity
        return systems

    @requires("utilities")
    def _propensity_iterator(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]]) \
            -> Iterator[Tuple[str, int]]:
        """
        Allows one to iterate over the names and spin multiplicities of the propensity systems of a given name.

        Parameters
        ----------
        name : str
            The base name of the system to iterate over; must not be a propensity system name.
        systems : Dict[str, Optional[utils.core.Calculator]]
            The map of systems.

        Yields
        ------
        Iterator[Tuple[str, int]]
            The system name and spin multiplicity of the propensity systems.
        """
        propensity_limit = self.settings[self.propensity_key]["check"]
        for shift in range(-propensity_limit, propensity_limit + 1):
            try:
                multiplicity = self.get_multiplicity(self.get_calc(name, systems)) + shift * 2
            except CalculatorNotPresentException:
                continue
            if multiplicity > 0:
                shift_name = f"{name}_multiplicity_shift_{shift}" if shift else name
                yield shift_name, multiplicity

    def _spin_propensity_single_points(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]],
                                       error_msg: str) -> Dict[str, Optional[utils.core.Calculator]]:
        """
        Carry out single point energy calculations for the spin propensities of a given name.

        Parameters
        ----------
        name : str
            The base name of the system
        systems : Dict[str, Optional[utils.core.Calculator]]
            The systems map
        error_msg : str
            The error message to give, should all calculations fail

        Returns
        -------
        Dict[str, Optional[utils.core.Calculator]]
            The updated systems map
        """

        info = f"Single point calculations of {name}"
        if self.settings[self.propensity_key]["check"]:
            info += " with potential spin propensities"
        info += ":\n"
        print(info)
        total_success = 0
        for shift_name, multiplicity in self._propensity_iterator(name, systems):
            if self.get_calc(shift_name, systems).get_results().energy is not None:
                # we already have energy for this system
                total_success += 1
                continue
            print(f"Carrying out single point calculation for the spin multiplicity of '{multiplicity}':")
            try:
                systems, success = readuct.run_single_point_task(
                    systems,
                    [shift_name],
                    require_bond_orders=not self.connectivity_settings['only_distance_connectivity'],
                    stop_on_error=False
                )
            except RuntimeError as e:
                sys.stderr.write(f"{shift_name} cannot be calculated because: {str(e)}\n")
                systems[shift_name] = None
                success = False
            if success:
                total_success += 1
            else:
                systems[shift_name] = None
        if not total_success:
            self.throw_if_not_successful(False, systems, [name], ["energy"], error_msg)
        return systems

    @is_configured
    def _spin_propensity_optimizations(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]],
                                       error_msg: str,
                                       readuct_task: SubTaskToReaductCall = SubTaskToReaductCall.OPT,
                                       task_settings_key: Optional[str] = None) \
            -> Dict[str, Optional[utils.core.Calculator]]:
        """
        Carry out structure optimizations for the spin propensities of a given name.

        Parameters
        ----------
        name : str
            The base name of the system
        systems : Dict[str, Optional[utils.core.Calculator]]
            The systems map
        error_msg : str
            The error message to give, should all calculations fail
        readuct_task : SubTaskToReaductCall
            The task to perform with readuct, by default SubTaskToReaductCall.OPT
        task_settings_key : Optional[str]
            The key in the settings dictionary to use for the task settings, by default None which will be the opt_key

        Returns
        -------
        Dict[str, Optional[utils.core.Calculator]]
            The updated systems map

        Raises
        ------
        RuntimeError
            If all calculations fail
        """
        info = f"Optimizing {name}"
        if self.settings[self.propensity_key]["check"]:
            info += " with potential spin propensities"
        info += ":\n"
        print(info)
        total_success = 0
        lowest_name, allowed_names = self._get_propensity_names_within_range(
            name, systems,
            self.settings[self.propensity_key]["energy_range_to_optimize"]
        )
        all_names = [lowest_name] + allowed_names
        if task_settings_key is None:
            task_settings_key = self.opt_key
        task_settings = deepcopy(self.settings[task_settings_key])
        wanted_output_name: Optional[List[str]] = task_settings.get("output")
        if wanted_output_name is not None and len(wanted_output_name) > 1:
            self.raise_named_exception("More than one output name is not allowed")
        if wanted_output_name is not None:
            del task_settings["output"]
        for shift_name, multiplicity in self._propensity_iterator(name, systems):
            if shift_name not in all_names:
                continue
            print(f"Carrying out structure optimization for the spin multiplicity of '{multiplicity}':")
            try:
                systems, success = self.observed_readuct_call(
                    readuct_task, systems, [shift_name], stop_on_error=False, **task_settings
                )
            except RuntimeError as e:
                sys.stderr.write(f"{shift_name} cannot be calculated because: {str(e)}\n")
                success = False
            if success:
                total_success += 1
            else:
                systems[shift_name] = None
        if not total_success:
            self.throw_if_not_successful(False, systems, [name], ["energy"], error_msg)
        if wanted_output_name is not None:
            lowest_name, _ = self._get_propensity_names_within_range(
                name, systems,
                self.settings[self.propensity_key]["energy_range_to_optimize"]
            )
            if lowest_name is None:
                self.raise_named_exception("No optimization was successful.")
                raise RuntimeError("Unreachable")
            systems[wanted_output_name[0]] = self.get_calc(lowest_name, systems)
        return systems

    @requires("readuct")
    @is_configured
    def _limited_spin_propensity_optimization(self, structure_name: str,
                                              systems: Dict[str, Optional[utils.core.Calculator]],
                                              name_stub: str, required_properties: List[str],
                                              readuct_task: SubTaskToReaductCall = SubTaskToReaductCall.OPT,
                                              task_settings_key: Optional[str] = None) \
            -> Dict[str, Optional[utils.core.Calculator]]:
        """
        Carries out structure optimizations for the spin propensities of a given name, but only for the lowest energy
        spin state and the base spin state.

        Parameters
        ----------
        structure_name : str
            The base name of the system
        systems : Dict[str, Optional[utils.core.Calculator]]
            The systems map
        name_stub : str
            The stub for naming of the structures
        required_properties : List[str]
            The properites expected to be present after a calculation
        readuct_task : SubTaskToReaductCall
            The task to perform with readuct, by default SubTaskToReaductCall.OPT
        task_settings_key : Optional[str]
            The key in the settings dictionary to use for the task settings, by default None which will be the opt_key

        Returns
        -------
        Dict[str, Optional[utils.core.Calculator]]
            The updated systems map

        Raises
        ------
        RuntimeError
            No single point calculation was previously successful
        RuntimeError
            All calculations fail
        """
        prev_lowest = None
        lowest_name, _ = self._get_propensity_names_within_range(
            structure_name, systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
        )
        if lowest_name is None:
            self.raise_named_exception(f"No calculation was successful for {structure_name}.")
            raise RuntimeError("Unreachable")
        if task_settings_key is None:
            task_settings_key = self.opt_key
        task_settings = deepcopy(self.settings[task_settings_key])
        wanted_output_name: Optional[List[str]] = task_settings.get("output")
        if wanted_output_name is not None:
            del task_settings["output"]
        if wanted_output_name is not None and len(wanted_output_name) > 1:
            self.raise_named_exception("More than one output name is not allowed")
        while lowest_name != prev_lowest:
            prev_lowest = lowest_name
            multiplicity = self._multiplicity_from_name(structure_name, lowest_name, systems)
            print(f"Optimizing {lowest_name}' with spin multiplicity {multiplicity}:")
            try:
                systems = self.observed_readuct_call_with_throw(
                    readuct_task, systems, [lowest_name], required_properties,
                    f"{name_stub.capitalize()} optimization failed:\n", **task_settings
                )
            except RuntimeError as e:
                sys.stderr.write(f"{lowest_name} cannot be calculated because: {str(e)}\n")
                systems[lowest_name] = None
                break
            systems = self._spin_propensity_single_points(
                structure_name, systems, f"{name_stub.capitalize()} optimization failed:\n")
            lowest_name, _ = self._get_propensity_names_within_range(
                structure_name, systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
            )
        if wanted_output_name is not None:
            lowest_name, _ = self._get_propensity_names_within_range(
                structure_name, systems,
                self.settings[self.propensity_key]["energy_range_to_optimize"]
            )
            if lowest_name is None:
                self.raise_named_exception("No optimization was successful.")
                raise RuntimeError("Unreachable")
            systems[wanted_output_name[0]] = self.get_calc(lowest_name, systems)
        return systems

    def _multiplicity_from_name(self, base_name: str, propensity_system_name: str,
                                systems: Dict[str, Optional[utils.core.Calculator]]) -> int:
        for shift_name, multiplicity in self._propensity_iterator(base_name, systems):
            if shift_name == propensity_system_name:
                return multiplicity
        self.raise_named_exception(f"Could not find multiplicity for {propensity_system_name}")
        raise RuntimeError("Unreachable")

    @requires("database")
    @is_configured
    def _store_structure_with_propensity_check(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]],
                                               label: db.Label, enforce_to_save_base_name: bool,
                                               surface_indices: Optional[Union[List[int], Set[int]]] = None) \
            -> db.Structure:
        """
        Stores the structure with the given name in the database, but checks for spin propensities and stores the
        structures with a relevant energy difference to the lowest energy structure as well.

        Parameters
        ----------
        name : str
            The name of the structure to store.
        systems : Dict[str, Optional[utils.core.Calculator]]
            The map of systems.
        label : db.Label
            The label to store the structure with.
        enforce_to_save_base_name : bool
            If set to True, the base name will be saved as well, even if it has a higher energy.
        surface_indices : Optional[Union[List[int], Set[int]]]
            The indices of the surface atoms in the structure.

        Returns
        -------
        db.Structure
            The structure that was stored in the database.
        """
        from scine_utilities import settings_names as sn

        def create_impl(structure_name: str, system_map: Dict[str, Optional[utils.core.Calculator]]) -> db.Structure:
            bond_orders, system_map = self.make_bond_orders_from_calc(system_map, structure_name, surface_indices)
            calc = self.get_calc(structure_name, system_map)
            new_structure = self.create_new_structure(calc, label)
            self.store_energy(calc, new_structure)
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
            name, systems, self.settings[self.propensity_key]["energy_range_to_save"]
        )
        if lowest_name is None:
            self.raise_named_exception(f"No successful calculation available for {name}")
            raise RuntimeError("Unreachable")
        spin_propensity_hit = lowest_name != name
        # Printing information
        if spin_propensity_hit:
            print(f"Noticed spin propensity. Lowest energy spin multiplicity of {name} is "
                  f"{self.get_calc(lowest_name, systems).settings[sn.spin_multiplicity]}")
        if names_to_save:
            print("Spin states with rel. energies to lowest state in kJ/mol which are also saved to the database:")
            print("name | multiplicity | rel. energy")
            base_energy = self.get_calc(lowest_name, systems).get_results().energy
            if base_energy is None:
                self.raise_named_exception(f"No energy calculated for {lowest_name}")
                raise RuntimeError("Unreachable")  # just for linters
            for n in names_to_save:
                system = self.get_calc(n, systems)
                multiplicity = system.settings[sn.spin_multiplicity]
                energy = system.get_results().energy
                if energy is None:
                    self.raise_named_exception(f"No energy calculated for {n}")
                    raise RuntimeError("Unreachable")  # just for linters
                rel_energy = (energy - base_energy) * utils.KJPERMOL_PER_HARTREE
                print(f"  {n} | {multiplicity} | {rel_energy}")
        if enforce_to_save_base_name:
            print(f"Still saving the base multiplicity of "
                  f"{self.get_calc(name, systems).settings[sn.spin_multiplicity]} in the elementary step")
            # overwrite names to simply safe and write as product of elementary step
            names_to_save += [lowest_name]
            if name in names_to_save:
                names_to_save.remove(name)
            lowest_name = name

        # Saving information
        name_to_structure_and_label_map: Dict[str, Tuple[db.Structure, db.Label]] = {}
        for n in names_to_save + [lowest_name]:
            # Store as Tuple[db.Structure, db.Label]
            structure = create_impl(n, systems)
            name_to_structure_and_label_map[n] = (structure, structure.get_label())

        # Decide which structure to return
        # Lowest name if no better spin state was found or if the lower spin state still has the same label as name
        if not spin_propensity_hit or \
                name_to_structure_and_label_map[lowest_name][1] == label or \
                enforce_to_save_base_name:
            return name_to_structure_and_label_map[lowest_name][0]
        return name_to_structure_and_label_map[name][0]

    @requires("utilities")
    def _get_propensity_names_within_range(self, name: str, systems: Dict[str, Optional[utils.core.Calculator]],
                                           allowed_energy_range: float) -> Tuple[Optional[str], List[str]]:
        """
        Gives the lowest-energy name and names within a given energy range of the lowest energy name.

        Parameters
        ----------
        name : str
            The base name of the system.
        systems : Dict[str, Optional[utils.core.Calculator]]
            The systems map.
        allowed_energy_range : float
            The allowed energy range in kJ/mol.

        Returns
        -------
        Tuple[Optional[str], List[str]]
            The lowest name and the names within the energy range.
        """
        energies: Dict[str, Optional[float]] = {}
        for shift_name, _ in self._propensity_iterator(name, systems):
            calc = systems.get(shift_name, None)
            energies[shift_name] = calc.get_results().energy if calc is not None else None
        cleared_energies = {k: v for k, v in energies.items() if v is not None}
        if not cleared_energies:
            sys.stderr.write(f"No energy calculated for any spin state of {name}\n")
            return None, []
        # get name with the lowest energy to save as product
        lowest_name = min(cleared_energies, key=cleared_energies.get)  # type: ignore
        lowest_energy = cleared_energies[lowest_name]
        names_within_range: List[str] = []
        for k, v in energies.items():
            if v is not None and k != lowest_name and \
                    abs(v - lowest_energy) * utils.KJPERMOL_PER_HARTREE < allowed_energy_range:
                names_within_range.append(k)
        return lowest_name, names_within_range
