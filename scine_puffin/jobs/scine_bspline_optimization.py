# -*- coding: utf-8 -*-
from __future__ import annotations

__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import sys
from copy import deepcopy
from typing import TYPE_CHECKING, Dict

from scine_puffin.config import Configuration
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob
from typing import Optional, List
from scine_puffin.utilities.scine_helper import SettingsManager
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_database")


class ScineBsplineOptimization(ReactJob):
    __doc__ = ("""
    The job interpolated between two structures. Generates a transition state guess,
    optimizes the transition state, and verifies the IRC.

    The job performs the following steps:
    1. Optimize reactant and product structure.
    If reactant and product converge to structures with the same graph. Stop and add
    barrier-less reactions if at least one of the original structures was a flask
    (+ geometry optimization of the separated structures).
    Else:
    2. Run a B-spline interpolation and optimization to extract an TS guess.
    3. Optimize the transition state.
    4. See @scine_react_complex_nt2.py for all following steps (Hessian, IRC, ...).

    **Order Name**
      ``scine_bspline_optimization``

    **Required Input**
      The first structures corresponds to the reactants, the second structure
      to the product.

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      All possible settings for this job are based on those available in SCINE
      ReaDuct. For a complete list see the
      `ReaDuct manual <https://scine.ethz.ch/download/readuct>`_

      Given that this job does more than one, in fact many separate calculations
      it is possible to target each individually with the settings. In order to
      achieve this each regular setting, such as ``convergence_max_iterations``
      has to be prepended with a tag, identifying which part of the job it is
      meant to impact. If the setting is meant to be added to the IRC scan at
      the end of this job ``irc_convergence_max_iterations`` should be used.
      Note that this may include a doubling of this style of flags, as Readuct
      uses a similar way of sorting options. Hence, choosing a none default
      ``irc_mode`` in this task has to be done using ``irc_irc_mode``.

      The complete list prefixes for specific settings for the steps listed at
      the start of this section is:

       1. TS optimization: ``tsopt_*``
       2. Validation using an IRC scan: ``irc_*``
       3. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       4. Optimization of the products and reactants: ``opt_*``

    """ + "\n"
               + ReactJob.optional_settings_doc() + "\n"
               + ReactJob.general_calculator_settings_docstring() + "\n"
               + ReactJob.generated_data_docstring() + "\n"
               + ReactJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine double ended transition state optimization from b-splines"
        self.exploration_key = "bspline"
        tsopt_defaults = {
            "output": ["ts"],
            "optimizer": "bofill",
            "convergence_max_iterations": 200,
        }
        irc_defaults = {
            "output": ["irc_forward", "irc_backward"],
            "convergence_max_iterations": 50,
            "irc_initial_step_size": 0.1,
            "sd_factor": 1.0,
            "stop_on_error": False,
        }
        ircopt_defaults = {"stop_on_error": True, "convergence_max_iterations": 200}
        opt_defaults = {
            "convergence_max_iterations": 500,
        }
        bspline_defaults = {
            "output": ["tsguess"],
            "extract_ts_guess": True,
            "optimize": True,
        }
        self.settings = {
            **self.settings,
            "bspline": bspline_defaults,
            "tsopt": tsopt_defaults,
            "irc": irc_defaults,
            "ircopt": ircopt_defaults,
            self.opt_key: opt_defaults,
        }

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_readuct as readuct
        import scine_molassembler as masm
        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            if len(calculation.get_structures()) != 2:
                self.raise_named_exception(f"{self.name} requires 2 input structures.")
            r_structure = db.Structure(calculation.get_structures()[0], self._structures)
            p_structure = db.Structure(calculation.get_structures()[1], self._structures)
            if len(r_structure.get_atoms()) != len(p_structure.get_atoms()):
                self.raise_named_exception(f"{self.name} requires that the input structures are the same molecule.")
            if r_structure.get_model() != p_structure.get_model():
                self.raise_named_exception(f"{self.name} requires that the input structures have the same model.")
            if r_structure.get_multiplicity() != p_structure.get_multiplicity() or \
                    r_structure.get_charge() != p_structure.get_charge():
                self.raise_named_exception(f"{self.name} requires that the input structures have the same "
                                           f"molecular charge and spin multiplicity.")
            settings_manager, program_helper = self.create_helpers(r_structure)
            settings_manager.separate_settings(self._calculation.get_settings())
            settings_manager.update_calculator_settings(r_structure, self._calculation.get_model(),
                                                        self.config["resources"])
            self.sort_settings(settings_manager.task_settings)

            self.ref_structure = r_structure
            # Prepare the structures by
            # 1. optimizing both spline ends.
            # 2. set attributes of parent class start_graph, start_charges etc.
            reactant_name, product_name, opt_r_graph, opt_p_graph = self.__prepare_structures(settings_manager,
                                                                                              r_structure, p_structure)
            # Stop the calculation if both spline ends collapsed to the same species.
            if masm.JsonSerialization.equal_molecules(opt_r_graph, opt_p_graph):
                self.__check_barrierless_reactions(settings_manager, reactant_name, product_name, r_structure,
                                                   p_structure)
                calculation.set_comment(self.name + " Spline ends transform barrier-less!")
                self.capture_raw_output()
                raise breakable.Break

            # self.save_initial_graphs_and_charges(settings_manager, [opt_reactant])
            # The spline ends are true minima. Therefore, their must be a transition state between them.
            """ B-Spline Optimization """
            inputs = [reactant_name, product_name]
            print("\nBspline Settings:")
            print(self.settings["bspline"], "\n")
            self.systems, success = readuct.run_bspline_task(
                self.systems, inputs, **self.settings["bspline"]
            )
            self.throw_if_not_successful(
                success,
                self.systems,
                inputs,
                ["energy"],
                "B-Spline optimization failed:\n",
            )

            """ TSOPT-Hess-IRC """
            inputs = self.output("bspline")
            product_names, start_names = self._tsopt_hess_irc_ircopt(inputs[0], settings_manager)

            """ Store new starting material conformer(s) """
            r_tuple = None
            # This may be re-enabled in the future. Therefore, I would like to keep it as a comment.
            # p_tuple = self.__check_barrierless_alternative_reactions(settings_manager, product_name, p_structure,
            #                                                          product_names, "product_00")
            if start_names is not None:
                if not self.no_irc_structure_matches_start:
                    r_tuple = self.__check_barrierless_alternative_reactions(settings_manager, reactant_name,
                                                                             r_structure, start_names, "reactant_00")
                start_structures = self.store_start_structures(
                    start_names, program_helper, "tsopt", [r_structure.id()])
            else:
                if r_structure.get_model() == self._calculation.get_model():
                    start_structures = [self._calculation.get_structures()[0]]
                else:
                    start_structures = self.store_start_structures(
                        [reactant_name], program_helper, "tsopt", [r_structure.id()])

            # If the lhs or rhs of the reaction decomposes into fragment through a barrier-less reaction and these
            # fragments are different from the fragments of the original lhs or rhs, e.g,
            # final: A + B --> AB --> C
            # orig:  D + E --> DE --> F  optimized into  D' + E' --> AB --> C
            # we have to add a barrier-less reaction transforming between the fragments. A simple example for such
            # a situation is a barrier-less protonation that is only barrier-less with the "new" electronic structure
            # model.
            # Note that this logic only applies if the individual endpoints used as input for the spline, are
            # rediscovered by the IRC. Since this is only checked for the lhs, the corresponding fragmentation
            # embedding for the rhs is disabled at the moment.
            lhs, _, _ = self.react_postprocessing(product_names, program_helper, "tsopt", start_structures)
            if r_tuple is not None:
                self.__add_barrierless_reaction(r_tuple[0], r_tuple[1], r_tuple[2], lhs, r_tuple[3])
            # if p_tuple is not None:
            #     self.__add_barrierless_reaction(p_tuple[0], p_tuple[1], p_tuple[2], rhs, p_tuple[3])
        return self.postprocess_calculation_context()

    def __set_up_calculator(self, structure: db.Structure, settings_manager: SettingsManager, name: str):
        """
        Create a calculator for the given structure.
        """
        xyz_name = name + ".xyz"
        utils.io.write(xyz_name, structure.get_atoms())
        structure_calculator_settings = deepcopy(settings_manager.calculator_settings)
        structure_calculator_settings[utils.settings_names.molecular_charge] = structure.get_charge()
        structure_calculator_settings[
            utils.settings_names.spin_multiplicity] = structure.get_multiplicity()
        reactant = utils.core.load_system_into_calculator(
            xyz_name,
            self._calculation.get_model().method_family,
            **structure_calculator_settings,
        )
        self.systems[name] = reactant

        fragment_atoms, graph_string, charges, multiplicities, decision_lists = self.get_graph_charges_multiplicities(
            name, structure.get_charge())

        if structure.has_graph("masm_cbor_graph") and structure.has_graph("masm_decision_list"):
            structure.set_graph("masm_cbor_graph", graph_string)
            structure.set_graph("masm_decision_list", ";".join(decision_lists))

        return fragment_atoms, graph_string, charges, multiplicities, decision_lists

    def __check_barrierless_alternative_reactions(self, settings_manager: SettingsManager, opt_name_reactant: str,
                                                  r_structure: db.Structure, opt_reactant_fragment_names: List[str],
                                                  r_name: str):
        """
        Check if there are multiple plausible decomposition paths for the given complex or structure, i.e, the structure
        reacted barrier-less leading to a different fragmentation path as for the original elementary step.
        """
        r_fragments, _, r_charges, r_multi, _ = self.__set_up_calculator(r_structure, settings_manager, r_name)

        if len(r_fragments) == 1:
            return None

        opt_r_orig_names, opt_r_orig_fragment_graphs, _ = self.__optimize_and_get_graphs_and_energies(
            "opt_" + r_name + "_orig_fragments", r_fragments, r_charges, r_multi, settings_manager
        )

        if not self.__same_molecules(opt_r_orig_names, opt_reactant_fragment_names):
            opt_r_graph, self.systems = self.make_graph_from_calc(self.systems, opt_name_reactant)
            return opt_name_reactant, opt_r_orig_fragment_graphs, opt_r_graph, opt_r_orig_names
        return None

    def __check_barrierless_reactions(self, settings_manager: SettingsManager, opt_name_reactant: str,
                                      opt_name_product: str, r_structure: db.Structure, p_structure: db.Structure):
        """
        Check if both input structures collapsed to the same molecule. We will compare the optimized structures of
        the input's lhs and rhs.
        """
        import scine_molassembler as masm
        results = self._calculation.get_results()
        results.clear()
        self._calculation.set_results(results)
        charge = r_structure.get_charge()
        r_name = "reactant_00"
        p_name = "product_00"

        r_fragments, r_graph, r_charges, r_multi, _ = self.__set_up_calculator(
            r_structure, settings_manager, r_name)
        p_fragments, p_graph, p_charges, p_multi, _ = self.__set_up_calculator(
            p_structure, settings_manager, p_name)
        # check graph of spline ends
        opt_r_fragments, opt_r_graph, opt_r_charges, opt_r_multiplicities, _ = \
            self.get_graph_charges_multiplicities(opt_name_reactant, charge)
        opt_p_fragments, opt_p_graph, opt_p_charges, opt_p_multiplicities, _ = \
            self.get_graph_charges_multiplicities(opt_name_product, charge)
        # create structures for optimized ends
        if ";" in opt_r_graph:
            opt_r_fragment_names, opt_r_fragment_graphs, _ = self.__optimize_and_get_graphs_and_energies(
                "opt_r_fragments", opt_r_fragments, opt_r_charges, opt_r_multiplicities, settings_manager)
            opt_reactant_structure_ids = self.__add_barrierless_reaction(opt_name_reactant, opt_r_fragment_graphs,
                                                                         opt_r_graph, None, opt_r_fragment_names)
            # optimization changed the initial complex. Add barrier-less step between previous fragments and the
            # optimized fragment
            if ";" in r_graph and not masm.JsonSerialization.equal_molecules(r_graph, opt_r_graph):
                opt_r_orig_framnet_names, opt_r_orig_fragment_graphs, _ = self.__optimize_and_get_graphs_and_energies(
                    "opt_r_orig_fragments", r_fragments, r_charges, r_multi, settings_manager
                )
                if not self.__same_molecules(opt_r_orig_framnet_names, opt_r_fragment_names):
                    self.__add_barrierless_reaction(opt_name_reactant, opt_r_orig_fragment_graphs, opt_r_graph,
                                                    opt_reactant_structure_ids, opt_r_orig_framnet_names)

        if ";" in opt_p_graph and not masm.JsonSerialization.equal_molecules(opt_p_graph, opt_r_graph):
            opt_p_fragment_names, opt_p_frgagment_graphs, _ = self.__optimize_and_get_graphs_and_energies(
                "opt_p_fragments",
                opt_p_fragments,
                opt_p_charges,
                opt_p_multiplicities,
                settings_manager)
            opt_structure_ids = self.__add_barrierless_reaction(opt_name_product, opt_p_frgagment_graphs, opt_p_graph,
                                                                None, opt_p_fragment_names)

            if ";" in p_graph and not masm.JsonSerialization.equal_molecules(p_graph, opt_p_graph):
                opt_p_orig_framnet_names, opt_p_orig_fragment_graphs, _ = self.__optimize_and_get_graphs_and_energies(
                    "opt_p_orig_fragments", p_fragments, p_charges, p_multi, settings_manager
                )
                if not self.__same_molecules(opt_p_orig_framnet_names, opt_p_fragment_names):
                    self.__add_barrierless_reaction(opt_name_product, opt_p_orig_fragment_graphs, opt_p_graph,
                                                    opt_structure_ids, opt_p_orig_framnet_names)

    def __prepare_structures(self, settings_manager, reactant_structure, products_structure):
        """
        Optimize the input structures + generate graphs.
        """
        # optimize spline ends
        """ Reactant """
        opt_name_reactant, self.systems = self.optimize_structures("opt_reactant", self.systems,
                                                                   [reactant_structure.get_atoms()],
                                                                   [reactant_structure.get_charge()],
                                                                   [reactant_structure.get_multiplicity()],
                                                                   deepcopy(
                                                                       settings_manager.calculator_settings.as_dict()))
        if len(opt_name_reactant) != 1:
            self.raise_named_exception("The optimization of the reactant structure failed.")
            raise RuntimeError("Unreachable")
        lowest_r_name, _ = self._get_propensity_names_within_range(
            opt_name_reactant[0], self.systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
        )
        if lowest_r_name is None:
            self.raise_named_exception("No reactant optimization was successful.")
            raise RuntimeError("Unreachable")
        if lowest_r_name != opt_name_reactant[0]:
            sys.stderr.write(f"Warning: Detected a lower energy spin multiplicity of "
                             f"{self.get_multiplicity(self.get_system(lowest_r_name))} for the reactant.")
            opt_name_reactant = [lowest_r_name]

        """ Product """
        opt_name_product, self.systems = self.optimize_structures("opt_product", self.systems,
                                                                  [products_structure.get_atoms()],
                                                                  [products_structure.get_charge()],
                                                                  [products_structure.get_multiplicity()],
                                                                  deepcopy(
                                                                      settings_manager.calculator_settings.as_dict()))
        if len(opt_name_product) != 1:
            self.raise_named_exception("The optimization of the product structure failed.")
            raise RuntimeError("Unreachable")
        lowest_p_name, _ = self._get_propensity_names_within_range(
            opt_name_product[0], self.systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
        )
        if lowest_p_name is None:
            self.raise_named_exception("No product optimization was successful.")
            raise RuntimeError("Unreachable")
        if lowest_p_name != opt_name_product[0]:
            sys.stderr.write(f"Warning: Detected a lower energy spin multiplicity of "
                             f"{self.get_multiplicity(self.get_system(lowest_p_name))} for the product.")
            opt_name_product = [lowest_p_name]

        if self.get_multiplicity(self.get_system(opt_name_reactant[0])) != \
                self.get_multiplicity(self.get_system(opt_name_product[0])):
            self.raise_named_exception("The optimized reactant and product have different spin multiplicities.")
            raise RuntimeError("Unreachable")

        _, opt_r_graph, opt_r_charges, opt_r_multiplicities, opt_r_decision_list = \
            self.get_graph_charges_multiplicities(opt_name_reactant[0], products_structure.get_charge())
        _, opt_p_graph, _, _, _ = \
            self.get_graph_charges_multiplicities(opt_name_product[0], products_structure.get_charge())

        self.start_charges = opt_r_charges
        self.start_multiplicities = opt_r_multiplicities
        self.start_graph = opt_r_graph
        self.start_decision_lists = opt_r_decision_list
        self.rc_opt_system_name = opt_name_reactant[0]

        return opt_name_reactant[0], opt_name_product[0], opt_r_graph, opt_p_graph

    def __create_complex_or_minimum(self, graph: str, calculator_name: str):
        """
        Create a structure with the correct label according to its Molassembler serialization (graph).
        """
        label = db.Label.MINIMUM_OPTIMIZED if ";" not in graph else db.Label.COMPLEX_OPTIMIZED
        new_structure = self.create_new_structure(self.systems[calculator_name], label)
        if self.ref_structure is None:
            self.raise_named_exception("The reference structure is not set.")
            raise RuntimeError("Unreachable")  # for mypy
        self.transfer_properties(self.ref_structure, new_structure)
        bond_orders, self.systems = self.make_bond_orders_from_calc(self.systems, calculator_name)
        self.store_energy(self.get_system(calculator_name), new_structure)
        self.store_bond_orders(bond_orders, new_structure)
        self.store_property(
            self._properties,
            "atomic_charges",
            "VectorProperty",
            self.get_system(calculator_name).get_results().atomic_charges,
            self._calculation.get_model(),
            self._calculation,
            new_structure,
        )
        self.add_graph(new_structure, bond_orders)

        results = self._calculation.get_results()
        results.add_structure(new_structure.id())
        self._calculation.set_results(self._calculation.get_results() + results)
        return new_structure

    def __optimize_and_get_graphs_and_energies(self, fragment_base_name: str, fragments: List[utils.AtomCollection],
                                               charges: List[int], multiplicities: List[int],
                                               settings_manager: SettingsManager):
        """
        Optimize molecular fragments and return their names, graphs, and energies.
        """
        opt_fragment_names, self.systems = (
            self.optimize_structures(fragment_base_name, self.systems, fragments,
                                     charges, multiplicities,
                                     deepcopy(settings_manager.calculator_settings.as_dict()))
        )
        opt_f_graphs = []
        fragment_energies = []
        for name, charge in zip(opt_fragment_names, charges):
            _, opt_f_graph, _, _, _ = self.get_graph_charges_multiplicities(name, charge)
            if ";" in opt_f_graph:
                self.raise_named_exception(
                    "Fragments in flask keep dissociating in the job "
                    + self.name
                )
            opt_f_graphs.append(opt_f_graph)
            fragment_energies.append(self.get_system(name).get_results().energy)

        return opt_fragment_names, opt_f_graphs, fragment_energies

    def __add_barrierless_reaction(self, opt_name: str, opt_f_graphs: List[str], opt_graph: str,
                                   opt_structure_ids: Optional[List[db.ID]], opt_fragment_names: List[str]):
        """
        Add a barrier-less reaction to the database between the optimized structure and its fragments.
        """
        self.__assert_conserved_atom(opt_fragment_names, [opt_name])
        db_results = self._calculation.get_results()
        db_results.clear()
        fragment_structures = []
        for name, graph in zip(opt_fragment_names, opt_f_graphs):
            fragment_structures.append(self.__create_complex_or_minimum(graph, name))

        if opt_structure_ids is None:
            opt_structure_ids = [self.__create_complex_or_minimum(opt_graph, opt_name).id()]
        new_step = db.ElementaryStep()
        new_step.link(self._elementary_steps)
        new_step.create([s.id() for s in fragment_structures], opt_structure_ids)
        new_step.set_type(db.ElementaryStepType.BARRIERLESS)
        db_results.add_elementary_step(new_step.id())
        self._calculation.set_comment(self.name + ": Barrier-less reaction found.")
        self._calculation.set_results(self._calculation.get_results() + db_results)
        return opt_structure_ids

    def __assert_conserved_atom(self, lhs_names: List[str], rhs_names: List[str]):
        """
        Assert that the number of atoms did not change between the calculators.
        """
        lhs_atoms = [self.get_system(name).structure for name in lhs_names]
        rhs_atoms = [self.get_system(name).structure for name in rhs_names]
        lhs_counts = self.__get_elements_in_atom_collections(lhs_atoms)
        rhs_counts = self.__get_elements_in_atom_collections(rhs_atoms)
        print("Atom counts lhs", lhs_counts)
        print("Atom counts rhs", rhs_counts)
        if lhs_counts != rhs_counts:
            raise RuntimeError("Error: Non stoichiometric elementary step detected. The structures are likely wrong.")

    @staticmethod
    def __get_elements_in_atom_collections(atom_collections: List[utils.AtomCollection]) -> Dict[str, int]:
        """
        Builds a dictionary containing the element symbols and the number of their occurrence in a given atom
        collection.
        """
        elements: List[str] = []
        for atom_collection in atom_collections:
            elements += [str(e) for e in atom_collection.elements]
        return {e: elements.count(e) for e in elements}

    def __same_molecules(self, names_one, names_two):
        """
        Check if two molecules/calculators are the same according to charges, Molassembler (graphs), and multiplicities.
        """
        import scine_molassembler as masm
        graphs_one, charges_one, multies_one = self.__get_sorted_graphs_charges_multiplicities(names_one)
        graphs_two, charges_two, multies_two = self.__get_sorted_graphs_charges_multiplicities(names_two)

        total_graph_one = ";".join(graphs_one)
        total_graph_two = ";".join(graphs_two)

        return charges_one == charges_two and multies_one == multies_two and masm.JsonSerialization.equal_molecules(
            total_graph_one, total_graph_two)

    def __get_sorted_graphs_charges_multiplicities(self, names_one: List[str]):
        """
        Get the sorted Molassembler serializations (graphs), charges, and multiplicites of the calculators corresponding
        to the given names
        """
        charges_one = []
        multies_one = []
        graphs_one = []
        for name in names_one:
            system = self.get_system(name)
            charges_one.append(system.settings[utils.settings_names.molecular_charge])
            multies_one.append(system.settings[utils.settings_names.spin_multiplicity])
            graphs_one.append(self.make_graph_from_calc(self.systems, name)[0])
        graphs, charges, multiplicities = (
            list(start_val)
            for start_val in zip(*sorted(zip(
                graphs_one,
                charges_one,
                multies_one)))
        )
        return graphs, charges, multiplicities
