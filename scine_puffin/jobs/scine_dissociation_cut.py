# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from scine_puffin.utilities import masm_helper, scine_helper
from scine_puffin.utilities.program_helper import ProgramHelper
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob
from copy import copy
from typing import Any, Dict, List, Tuple, Union
from itertools import combinations_with_replacement, permutations
import numpy as np


class ScineDissociationCut(ReactJob):
    """

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      All possible settings for this job are based on those available in SCINE
      ReaDuct. For a complete list see the
      `ReaDuct manual <https://scine.ethz.ch/static/download/readuct_manual.pdf>`_

      Given that this job consists of two separate tasks,
      it is possible to target each individually with the settings. In order to
      achieve this, each regular setting, such as ``convergence_max_iterations``
      has to be prepended with a tag, identifying which part of the job it is
      meant to impact. If the setting is meant to be added to the reactive
      complex optimization at the end of this job ``rcopt_convergence_max_iterations``
      should be used.
      Note that this may include a doubling of this style of flags, as ReaDuct
      uses a similar way of sorting options. Hence, choosing a non-default
      ``geoopt_coordinate_system`` in this task has to be done using
      ``rcopt_geoopt_coordinate_system``.

      The complete list prefixes for specific settings for the steps listed at
      the start of this section is:

       1. Single product optimizations ``opt_*``
       2. Optimization of the reactive complex: ``rcopt_*``


      The following settings are recognized without a prepending flag:

      add_based_on_distance_connectivity :: bool
          Whether to add the connectivity (i.e. add bonds) as derived from
          atomic distances when graphs are generated. (default: True)
      sub_based_on_distance_connectivity :: bool
          Whether to subtract the connectivity (i.e. remove bonds) as derived from
          atomic distances when graphs are generated. (default: True)
      only_distance_connectivity :: bool
          Whether to impose the connectivity solely from distances. (default: False)
      spin_propensity_check :: int
          The range to check for possible multiplicities for products. A value
          of 2 (default) will check triplet and quintet for a singlet
          and will check singlet, quintet und septet for triplet.
      charge_propensity_check :: int
          The range to check for possible charges for products. A value
          of 1 (default) will check +1 and -1 charges for both products,
          i.e. 6 different geometry optimizations for a single cut bond resulting in two molecules

      Additionally, all settings that are recognized by the chosen SCF program
      are also available. These settings are not required to be prepended with
      any flag.

      Common examples are:

      max_scf_iterations :: int
         The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Molassembler (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - A program implementing the SCINE Calculator interface, e.g. Sparrow

    **Generated Data**
      If successful the following data will be generated and added to the database:

      Elementary Steps
        If found, a single new barrierfree elementary step will be added to the database.

      Structures
        All the separated products and possible charges will be added to the database.

      Properties
        The ``bond_orders`` (``SparseMatrixProperty``), and
        ``electronic_energy`` (``NumberProperty``) of the given structure and all split products
        will be provided.
    """

    def __init__(self):
        super().__init__()
        self.name = "Scine React Job with bond cutting"
        opt_defaults: Dict[str, Any] = {
            "convergence_max_iterations": 500,
            "geoopt_coordinate_system": "cartesianWithoutRotTrans"
        }
        rcopt_defaults: Dict[str, Any] = {
            "optimizer": "bfgs",
            "bfgs_min_iterations": 5  # make sure that dissociated structure does not immediately signal convergence
        }
        self.settings = {
            **self.settings,
            "opt": opt_defaults
        }
        self.settings[self.rc_opt_system_name] = {**self.settings[self.rc_opt_system_name], **rcopt_defaults}
        self.diss = 'dissociations'
        self.charge_propensity = "charge_propensity_check"
        self.settings[self.job_key][self.diss]: List[int] = list()
        self.settings[self.job_key][self.charge_propensity] = 1

    @job_configuration_wrapper
    def run(self, _, calculation, config: Configuration) -> bool:

        import scine_database as db
        import scine_readuct as readuct
        import scine_utilities as utils

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            """ sanity checks """
            if len(calculation.get_structures()) != 1:
                self.raise_named_exception(f"{self.name} is only implemented for single molecule system.")
            settings_manager, program_helper = self.reactive_complex_preparations()
            if self.settings[self.rc_opt_system_name]['optimizer'].lower() != 'bfgs':
                # in case user specified different optimizer, delete default setting
                del self.settings[self.rc_opt_system_name]['bfgs_min_iterations']
            dissociations: List[int] = self.settings[self.job_key][self.diss]
            if not dissociations:
                self.raise_named_exception(f"Bond dissociation information is missing. It has to be "
                                           f"specified in the settings with '{self.diss}'.")
            if len(dissociations) % 2 != 0:
                self.raise_named_exception(f"Received an uneven number of entries in '{self.diss}', this does not "
                                           f"correspond to the expected format.")
            rc_atoms = self.systems[self.rc_key].structure
            for d in dissociations:
                if d < 0 or d >= len(rc_atoms):
                    self.raise_named_exception(f"Invalid entry '{d}' in '{self.diss}' for given structure "
                                               f"with '{len(rc_atoms)}' nuclei.")

            """ gather reactant info """
            bond_orders = self.make_bond_orders_from_calc(self.systems, self.rc_key)
            if not self.expected_results_check(self.systems, self.rc_key, ['energy', 'atomic_charges'])[0]:
                self.systems, success = readuct.run_sp_task(self.systems, [self.rc_key])
                self.throw_if_not_successful(success, self.systems, [self.rc_key], ['energy', 'atomic_charges'])
            rc_energy = self.systems[self.rc_key].get_results().energy
            rc_atomic_charges = self.systems[self.rc_key].get_results().atomic_charges

            """ cut bonds and determine new molecules """
            bond_breaks = []
            for i in range(0, len(dissociations), 2):
                lhs = dissociations[i]
                rhs = dissociations[i + 1]
                bond_breaks.append((lhs, rhs, 0.0))
            mol_result = masm_helper.get_molecules_result(
                rc_atoms,
                bond_orders,
                self.connectivity_settings,
                calculation.get_model().periodic_boundaries,
                self.surface_indices(db.Structure(calculation.get_structures()[0], self._structures)),
                bond_breaks
            )
            n_mols = len(mol_result.molecules)
            if n_mols == 1:
                self.raise_named_exception(f"Failed because the specified dissociations {dissociations} did not "
                                           f"suffice to generate multiple molecules.")
            # we split manually, because we need exact index mapping below
            split_molecules = [utils.AtomCollection() for _ in range(n_mols)]
            super_map: List[Tuple[int, int, int]] = []
            for source, target in enumerate(mol_result.component_map):
                # this is the extra information we need
                # it tells us which index in the split up molecule the atom has
                new_index = len(split_molecules[target])
                split_molecules[target].push_back(rc_atoms[source])
                super_map.append((source, target, new_index))

            """ Carry out various optimizations """
            propensity_range = self._get_propensity_range()
            print(f"Specified dissociations {dissociations} lead to {n_mols} separate molecules")
            print(f"Now optimizing each molecule with each a charge difference of {propensity_range} "
                  f"leading to {n_mols * len(propensity_range)} separate structure optimizations.")

            # get guess charges based on atomic charges of reactant
            mol_charges = self._get_charge_per_molecule(super_map, rc_atomic_charges)

            # gather most probable number of electrons and charge per molecule
            n_electrons = []
            guessed_charges: List[int] = []
            for i, molecule in enumerate(split_molecules):
                electrons = sum(utils.ElementInfo.Z(e) for e in molecule.elements)
                n_electrons.append(electrons)
                guessed_charges.append(int(round(mol_charges[i])))

            split_names: Dict[int, List[str]] = {}
            product_single_energies: Dict[int, List[Union[float, None]]] = {}
            for charge_diff in propensity_range:
                # This assumes minimal multiplicity, multiplicities are also checked for other values in calculations
                multiplicities = [(nel - charge - charge_diff) % 2 + 1
                                  for charge, nel in zip(guessed_charges, n_electrons)]
                charges = [charge + charge_diff for charge in guessed_charges]
                # We allow the optimizations to fail, but if this is the case the calculator becomes None
                # The reason is that is reasonable that some charge combinations might not be possible
                split_names[charge_diff] = self.optimize_structures(f"split_product_charge_difference_{charge_diff}",
                                                                    split_molecules,
                                                                    charges,
                                                                    multiplicities,
                                                                    settings_manager.calculator_settings,
                                                                    stop_on_error=False
                                                                    )
                # Now be careful due to possible None
                energies = []
                for name in split_names[charge_diff]:
                    calc = self.systems[name]
                    energies.append(None if calc is None else calc.get_results().energy)
                product_single_energies[charge_diff] = energies

            """ Deduce valid product energies """
            lowest_energy, lowest_combination, product_energies = \
                self._determine_diss_energies(product_single_energies, guessed_charges)

            # update model because job will be marked complete
            scine_helper.update_model(self.systems[self.rc_key], calculation, self.config)

            """ Print and save results so far """
            self._print_dissociation_energies(product_energies, split_names, rc_energy, guessed_charges)
            lowest_rhs_structures = self._save_dissociated_structures(split_names, lowest_combination,
                                                                      product_single_energies, program_helper)
            if lowest_energy < rc_energy:
                calculation.set_comment(f"The dissociation(s) {dissociations} has a formal negative electronic "
                                        "dissociation energy. This should be tested for a potential reaction with a "
                                        "barrier")
                raise breakable.Break

            """ Barrierless Reaction Check """
            # we need to write a newly generated reactive complex
            # we directly overwrite reactive complex in systems map
            self._setup_dissociated_reactive_complex(rc_atoms, super_map, bond_breaks, split_names, lowest_combination)

            # we optimize this new reactive complex and see whether we arrive at the original one
            rc_opt_graph, _ = self.check_for_barrierless_reaction()
            if rc_opt_graph is None:
                # this means we got the same graph as in the original structure -> it was successful
                print("Barrierless Reaction Found")
                graphs = []
                product_names = []
                for mol in range(n_mols):
                    charge = lowest_combination[mol]
                    name = split_names[charge][mol]
                    product_names.append(name)
                    graphs.append(self.make_graph_from_calc(self.systems, name))
                joined_graph = ";".join(graphs)
                print("Barrierless dissociation product graph:")
                print(joined_graph)
                print("Start Graph:")
                print(self.start_graph)
                db_results = self._calculation.get_results()
                # Save step
                new_step = db.ElementaryStep()
                new_step.link(self._elementary_steps)
                new_step.create(self._calculation.get_structures(), [rhs.id() for rhs in lowest_rhs_structures])
                new_step.set_type(db.ElementaryStepType.BARRIERLESS)
                db_results.add_elementary_step(new_step.id())
                self._calculation.set_comment(self.name + ": Barrierless reaction found.")
                self._calculation.set_results(self._calculation.get_results() + db_results)

        return self.postprocess_calculation_context()

    def _get_propensity_range(self) -> List[int]:
        propensity_limit = self.settings[self.job_key][self.charge_propensity]
        if propensity_limit < 0:
            self.raise_named_exception(f"'{self.charge_propensity}' is set to '{propensity_limit}', "
                                       f"but must be a positive number!")
        return list(range(-propensity_limit, propensity_limit + 1))

    def _determine_diss_energies(self, product_single_energies: Dict[int, List[Union[float, None]]],
                                 fragment_base_charges: List[int]) \
            -> Tuple[float, Tuple[int], Dict[Tuple[int], Union[float, None]]]:
        import scine_utilities as utils

        propensity_range = self._get_propensity_range()
        n_mols = len(product_single_energies[0])
        rc_charge: int = self.systems[self.rc_key].settings[utils.settings_names.molecular_charge]
        # we can only compare total energies that have a total charge identical to that of the rc,
        # This gives us the combinations that are possible based on the x charge differences and n molecules
        valid_combinations = [comb for comb in combinations_with_replacement(propensity_range, n_mols)
                              if sum([c + fragment_base_charges[i] for i, c in enumerate(comb)]) == rc_charge]
        # now we need to permute combinations to also get, e.g. (1, -1), and not only (-1, 1)
        # but avoid senseless duplicates such as [(0,0), (0,0)]
        permutated_unique_valid_combinations: List[Tuple[int]] = []
        for combination in valid_combinations:
            permutated_unique_valid_combinations += list(set(permutations(combination)))  # type: ignore
        print(f"Finding the lowest energy within the possible charge difference combinations "
              f"{permutated_unique_valid_combinations}")
        product_energies: Dict[Tuple[int], Union[float, None]] = {}
        lowest_energy = None
        lowest_combination = None
        for combination in permutated_unique_valid_combinations:
            # combination is a tuple of the charge difference for each molecule
            missing_energy = False
            energy = 0.0
            for mol_index, charge_diff in enumerate(combination):
                e = product_single_energies[charge_diff][mol_index]
                if e is None:
                    missing_energy = True
                    break
                energy += e
            if missing_energy:
                product_energies[combination] = None
                continue
            product_energies[combination] = energy
            if lowest_energy is None or (energy < lowest_energy):
                lowest_energy = energy
                lowest_combination = copy(combination)
        if lowest_energy is None or lowest_combination is None:
            self.raise_named_exception("Failed to optimize any valid product combination.")
            raise BaseException  # only for linters
        return lowest_energy, lowest_combination, product_energies

    def _print_dissociation_energies(self,
                                     product_energies: Dict[Tuple[int], Union[float, None]],
                                     split_names: Dict[int, List[str]],
                                     rc_energy: Union[float, None],
                                     fragment_base_charges: List[int]) -> None:
        import scine_utilities as utils
        if rc_energy is None:
            self.raise_named_exception("Calculation of reactant failed")
            return  # only for linters

        print("The evaluated dissociation energies in kJ/mol are:")
        print("Molecular charges | Multiplicities | Dissociation energy")
        print("---------------------------------------------------------")
        print_lengths = [18, 15]  # based on length of headers
        for charge_diffs, energy in product_energies.items():
            c_entry = str(list(np.array(charge_diffs) + np.array(fragment_base_charges)))
            if energy is None:
                m_entry: Union[str, List[int]] = "Not converged"
                e_entry = "Not converged"
            else:
                m_entry = [self.systems[split_names[charge][i]].settings[utils.settings_names.spin_multiplicity]
                           for i, charge in enumerate(charge_diffs)]
                e_entry = (energy - rc_energy) * utils.KJPERMOL_PER_HARTREE
            c_buffer = " " * (print_lengths[0] - len(c_entry))
            m_buffer = " " * (print_lengths[1] - len(str(m_entry)))
            print(f"{c_entry}{c_buffer}| {m_entry}{m_buffer}| {e_entry}")

    def _save_dissociated_structures(self, split_names: Dict[int, List[str]], lowest_combination: Tuple[int],
                                     product_single_energies: Dict[int, List[Union[float, None]]],
                                     program_helper: Union[ProgramHelper, None]) -> list:
        import scine_database as db
        # clear results
        db_results = self._calculation.get_results()
        db_results.clear()
        self._calculation.set_results(db_results)

        # store energy and bond orders for reactive complex, i.e. structure being dissociated
        start_structure = db.Structure(self._calculation.get_structures()[0], self._structures)
        self.store_energy(self.systems[self.rc_key], start_structure)
        bond_orders = self.make_bond_orders_from_calc(self.systems, self.rc_key)
        self.store_property(
            self._properties,
            "bond_orders",
            "SparseMatrixProperty",
            bond_orders.matrix,
            self._calculation.get_model(),
            self._calculation,
            start_structure
        )

        # save structure we have optimized and sensible energies
        # and return those with lowest energy
        lowest_rhs_structures = []
        for charge_diff, names in split_names.items():
            energies = product_single_energies[charge_diff]
            for mol, (name, energy) in enumerate(zip(names, energies)):
                if energy is None:
                    continue
                graph = self.make_graph_from_calc(self.systems, name)
                if ";" in graph:
                    rhs_structure = self.create_new_structure(self.systems[name], db.Label.COMPLEX_OPTIMIZED)
                else:
                    rhs_structure = self.create_new_structure(self.systems[name], db.Label.MINIMUM_OPTIMIZED)
                db_results.add_structure(rhs_structure.id())
                self.transfer_properties(self.ref_structure, rhs_structure)
                self.store_energy(self.systems[name], rhs_structure)
                bond_orders = self.make_bond_orders_from_calc(self.systems, name)
                self.store_property(
                    self._properties,
                    "bond_orders",
                    "SparseMatrixProperty",
                    bond_orders.matrix,
                    self._calculation.get_model(),
                    self._calculation,
                    rhs_structure,
                )
                self.add_graph(rhs_structure, bond_orders)
                if program_helper is not None:
                    program_helper.calculation_postprocessing(self._calculation, self.ref_structure, rhs_structure)
                if lowest_combination[mol] == charge_diff:
                    lowest_rhs_structures.append(rhs_structure)
        self._calculation.set_results(self._calculation.get_results() + db_results)
        if len(lowest_rhs_structures) != len(lowest_combination):
            self.raise_named_exception("Missed to save all lowest energy product structures")
        self.store_property(
            self._properties,
            "dissociated_structures",
            "StringProperty",
            ",".join([str(rhs.id()) for rhs in lowest_rhs_structures]),
            self._calculation.get_model(),
            self._calculation,
            rhs_structure,
        )
        return lowest_rhs_structures

    def _setup_dissociated_reactive_complex(self, rc_atoms, super_map: List[Tuple[int, int, int]],
                                            bond_breaks: List[Tuple[int, int, float]],
                                            split_names: Dict[int, List[str]], lowest_combination: Tuple[int],
                                            ) -> None:
        import scine_readuct as readuct
        import scine_utilities as utils

        n_mols = np.max(np.array(super_map)[:, 1]) + 1
        frankenstein_rc_atoms = copy(rc_atoms)
        # fit each molecule to the corresponding atoms in the old structure separately
        fitted_products = [utils.AtomCollection() for _ in range(n_mols)]
        target_sources = [utils.AtomCollection() for _ in range(n_mols)]
        for source, mol, mol_position in super_map:
            charge = lowest_combination[mol]
            name = split_names[charge][mol]
            product = self.systems[name].structure
            fitted_products[mol].push_back(product[mol_position])
            target_sources[mol].push_back(rc_atoms[source])
        for target, product in zip(target_sources, fitted_products):
            fit = utils.QuaternionFit(target.positions, product.positions)
            product.positions = fit.get_fitted_data()

        # set positions in new reactive complex to those part fitted ones
        for source, mol, mol_position in super_map:
            frankenstein_rc_atoms.set_position(source, fitted_products[mol].positions[mol_position])

        # shift separate molecules with vector of broken bond such that the broken bond distance is the sum of vdw radii
        for lhs, rhs, _ in bond_breaks:
            target_distance = sum(utils.ElementInfo.vdw_radius(rc_atoms.elements[i]) for i in [lhs, rhs])
            current_distance = np.linalg.norm(np.array(frankenstein_rc_atoms.positions[rhs] -
                                                       frankenstein_rc_atoms.positions[lhs]))
            direction = np.array(rc_atoms.positions[rhs] - rc_atoms.positions[lhs])
            direction /= np.linalg.norm(direction)  # normalize
            direction *= target_distance / current_distance  # shift to right length
            molecule_to_push = super_map[rhs][1]
            for i, pos in enumerate(frankenstein_rc_atoms.positions):
                if super_map[i][1] == molecule_to_push:
                    frankenstein_rc_atoms.set_position(i, pos + direction)
        # overwrite reactive complex in systems
        self.systems[self.rc_key].positions = frankenstein_rc_atoms.positions

        # check if we resemble the charge we expect
        self.systems, success = readuct.run_single_point_task(self.systems, [self.rc_key], require_charges=True)
        self.throw_if_not_successful(success, self.systems, [self.rc_key], ['atomic_charges'])
        new_rc_charges = self.systems[self.rc_key].get_results().atomic_charges
        mol_charges = self._get_charge_per_molecule(super_map, new_rc_charges)
        rounded_mol_charges = [int(round(c)) for c in mol_charges]
        lowest_combination_charges = []
        for mol, charge_diff in enumerate(lowest_combination):
            name = split_names[charge_diff][mol]
            lowest_combination_charges.append(self.systems[name].settings[utils.settings_names.molecular_charge])
        if rounded_mol_charges != lowest_combination_charges:
            self.raise_named_exception(f"The lowest energy charge separation {lowest_combination_charges} is not "
                                       f"present in our dissociated supermolecule, where we have a charge distribution "
                                       f"of {rounded_mol_charges}. This means we can likely not sample the energy "
                                       f"barrier without specialized methods.")

    @staticmethod
    def _get_charge_per_molecule(super_map: List[Tuple[int, int, int]], total_atomic_charges: np.ndarray) \
            -> np.ndarray:
        n_mols = np.max(np.array(super_map)[:, 1]) + 1
        mol_charges = np.zeros(n_mols)
        for source, mol, _ in super_map:
            mol_charges[mol] += total_atomic_charges[source]
        return mol_charges
