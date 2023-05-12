# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob


class ScineBsplineOptimization(ReactJob):
    """
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
      Readuct. For a complete list see the
      `ReaDuct manual <https://scine.ethz.ch/static/download/readuct_manual.pdf>`_

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

      The following settings are recognized without a prepending flag:

      add_based_on_distance_connectivity :: bool
          Whether to add the connectivity (i.e. add bonds) as derived from
          atomic distances when graphs are generated. (default: True)
      sub_based_on_distance_connectivity :: bool
          Whether to subtract the connectivity (i.e. remove bonds) as derived from
          atomic distances when graphs are generated. (default: True)
      only_distance_connectivity :: bool
          Whether to impose the connectivity solely from distances. (default: False)
      imaginary_wavenumber_threshold :: float
          Threshold value in inverse centimeters below which a wavenumber
          is considered as imaginary when the transition state is analyzed.
          Negative numbers are interpreted as imaginary. (default: 0.0)
      spin_propensity_check :: int
          The range to check for possible multiplicities for products. A value
          of 2 (default) will check triplet and quintet for a singlet
          and will check singlet, quintet und septet for triplet.

      Additionally all settings that are recognized by the SCF program chosen.
      are also available. These settings are not required to be prepended with
      any flag.

      Common examples are:

      max_scf_iterations :: int
         The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: molassembler (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - A program implementing the SCINE Calculator interface, e.g. Sparrow

    **Generated Data**
      If successful (technically and chemically) the following data will be
      generated and added to the database:

      Elementary Steps
        If found, a single new elementary step with the associated transition
        state will be added to the database.

      Structures
        The transition state (TS) and also the separated products and reactants
        will be added to the database.

      Properties
        The ``hessian`` (``DenseMatrixProperty``), ``frequencies``
        (``VectorProperty``), ``normal_modes`` (``DenseMatrixProperty``),
        ``gibbs_energy_correction`` (``NumberProperty``) and
        ``gibbs_free_energy`` (``NumberProperty``) of the TS will be
        provided. The ``electronic_energy`` associated with the TS structure and
        each of the products will be added to the database.
    """

    def __init__(self):
        super().__init__()
        self.name = "Scine double eneded transition state optimization from b-splines"
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
            "opt": opt_defaults,
        }

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_readuct as readuct
        import scine_utilities as utils
        import scine_database as db
        import scine_molassembler as masm
        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            r_structure = db.Structure(calculation.get_structures()[0], self._structures)
            p_structure = db.Structure(calculation.get_structures()[1], self._structures)
            settings_manager, program_helper = self.create_helpers(r_structure)
            settings_manager.separate_settings(self._calculation.get_settings())
            settings_manager.update_calculator_settings(r_structure, self._calculation.get_model(),
                                                        self.config["resources"])
            self.sort_settings(settings_manager.task_settings)

            self.ref_structure = r_structure
            # Prepare the structures by
            # 1. optimizing both spline ends.
            # 2. set attributes of parent class start_graph, start_charges etc.
            reactant_name, product_name, opt_r_graph, opt_p_graph = self.prepare_structures(settings_manager,
                                                                                            r_structure, p_structure)
            # Stop the calculation if both spline ends collapsed to the same species.
            if masm.JsonSerialization.equal_molecules(opt_r_graph, opt_p_graph):
                self.check_barrierless_reactions(settings_manager, reactant_name, product_name, r_structure,
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

            """ TSOPT JOB """
            print("TSOpt Settings:")
            print(self.settings["tsopt"], "\n")
            inputs = self.output("bspline")
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
                self.raise_named_exception(
                    "Error: "
                    + self.name
                    + " failed with message: "
                    + "TS has incorrect number of imaginary frequencies."
                )
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
                scine_helper.update_model(
                    self.systems[self.output("tsopt")[0]],
                    self._calculation,
                    self.config,
                )
                raise breakable.Break

            """ Store new starting material conformer(s) """
            r_tuple = None
            p_tuple = self.check_barrierless_alternative_reactions(settings_manager, product_name, p_structure,
                                                                   product_names, "product_00")
            if start_names is not None:
                r_tuple = self.check_barrierless_alternative_reactions(settings_manager, reactant_name, r_structure,
                                                                       start_names, "reactant_00")
                start_structures = self.store_start_structures(
                    start_names, program_helper, "tsopt", [])
            else:
                if r_structure.get_model() == self._calculation.get_model():
                    start_structures = [self._calculation.get_structures()[0]]
                else:
                    start_structures = self.store_start_structures(
                        [reactant_name], program_helper, "tsopt", [])

            # If the lhs or rhs of the reaction decomposes into fragment through a barrier-less reaction and these
            # fragments are different from the fragments of the original lhs or rhs, e.g,
            # final: A + B --> AB --> C
            # orig:  D + E --> DE --> F  optimized into  D' + E' --> AB --> C
            # we have to add a barrier-less reaction transforming between the fragments. A simple example for such
            # a situation is a barrier-less protonation that is only barrier-less with the "new" electronic structure
            # model.
            lhs, rhs = self.react_postprocessing(product_names, program_helper, "tsopt", start_structures)
            if r_tuple is not None:
                self.add_barrierless_reaction(r_tuple[0], r_tuple[1], r_tuple[2], lhs, r_tuple[3])
            if p_tuple is not None:
                self.add_barrierless_reaction(p_tuple[0], p_tuple[1], p_tuple[2], rhs, p_tuple[3])

        return self.postprocess_calculation_context()

    def set_up_calculator(self, structure, settings_manager, name):
        import scine_utilities as utils
        from copy import deepcopy

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

        if structure.has_graph("masm_cbor_graph") or structure.has_graph("masm_decision_list"):
            structure.set_graph("masm_cbor_graph", graph_string)
            structure.set_graph("masm_decision_list", ";".join(decision_lists))

        return fragment_atoms, graph_string, charges, multiplicities, decision_lists

    def check_barrierless_alternative_reactions(self, settings_manager, opt_name_reactant,
                                                r_structure, opt_reactant_fragment_names,
                                                r_name):
        r_fragments, _, r_charges, r_multi, _ = self.set_up_calculator(r_structure, settings_manager, r_name)

        if len(r_fragments) == 1:
            return None

        opt_r_orig_names, opt_r_orig_fragment_graphs, _ = self.optimize_and_get_graphs_and_energies(
            "opt_r_orig_fragments", r_fragments, r_charges, r_multi, settings_manager
        )

        if not self.same_molecules(opt_r_orig_names, opt_reactant_fragment_names):
            opt_r_graph = self.make_graph_from_calc(self.systems, opt_name_reactant)
            return opt_name_reactant, opt_r_orig_fragment_graphs, opt_r_graph, opt_r_orig_names
        return None

    def check_barrierless_reactions(self, settings_manager, opt_name_reactant, opt_name_product, r_structure,
                                    p_structure):
        import scine_molassembler as masm
        results = self._calculation.get_results()
        results.clear()
        self._calculation.set_results(results)
        charge = r_structure.get_charge()
        r_name = "reactant_00"
        p_name = "product_00"

        r_fragments, r_graph, r_charges, r_multi, _ = self.set_up_calculator(
            r_structure, settings_manager, r_name)
        p_fragments, p_graph, p_charges, p_multi, _ = self.set_up_calculator(
            p_structure, settings_manager, p_name)
        # check graph of spline ends
        opt_r_fragments, opt_r_graph, opt_r_charges, opt_r_multiplicities, _ =\
            self.get_graph_charges_multiplicities(opt_name_reactant, charge)
        opt_p_fragments, opt_p_graph, opt_p_charges, opt_p_multiplicities, _ =\
            self.get_graph_charges_multiplicities(opt_name_product, charge)
        # create structures for optimized ends
        if ";" in opt_r_graph:
            opt_r_fragment_names, opt_r_frgagment_graphs, _ = self.optimize_and_get_graphs_and_energies(
                "opt_r_fragments", opt_r_fragments, opt_r_charges, opt_r_multiplicities, settings_manager)
            opt_reactant_structure_ids = self.add_barrierless_reaction(opt_name_reactant, opt_r_frgagment_graphs,
                                                                       opt_r_graph, None, opt_r_fragment_names)

            # optimization changed the initial complex. Add barrier-less step between previous fragments and the
            # optimized fragment
            if ";" in r_graph and not masm.JsonSerialization.equal_molecules(r_graph, opt_r_graph):
                opt_r_orig_framnet_names, opt_r_orig_fragment_graphs, _ = self.optimize_and_get_graphs_and_energies(
                    "opt_r_orig_fragments", r_fragments, r_charges, r_multi, settings_manager
                )
                if not self.same_molecules(opt_r_orig_framnet_names, opt_r_fragment_names):
                    self.add_barrierless_reaction(opt_name_reactant, opt_r_orig_fragment_graphs, opt_r_graph,
                                                  opt_reactant_structure_ids, opt_r_orig_framnet_names)

        if ";" in opt_p_graph and not masm.JsonSerialization.equal_molecules(opt_p_graph, opt_r_graph):
            opt_p_fragment_names, opt_p_frgagment_graphs, _ = self.optimize_and_get_graphs_and_energies(
                "opt_p_fragments",
                opt_p_fragments,
                opt_p_charges,
                opt_p_multiplicities,
                settings_manager)
            opt_structure_ids = self.add_barrierless_reaction(opt_name_product, opt_p_frgagment_graphs, opt_p_graph,
                                                              None, opt_p_fragment_names)

            if ";" in p_graph and not masm.JsonSerialization.equal_molecules(p_graph, opt_p_graph):
                opt_p_orig_framnet_names, opt_p_orig_fragment_graphs, _ = self.optimize_and_get_graphs_and_energies(
                    "opt_p_orig_fragments", p_fragments, p_charges, p_multi, settings_manager
                )
                if not self.same_molecules(opt_p_orig_framnet_names, opt_p_fragment_names):
                    self.add_barrierless_reaction(opt_name_product, opt_p_orig_fragment_graphs, opt_p_graph,
                                                  opt_structure_ids, opt_p_orig_framnet_names)

    def prepare_structures(self, settings_manager, reactant_structure, products_structure):
        from copy import deepcopy
        # optimize spline ends
        opt_name_reactant = self.optimize_structures("opt_reactant", [reactant_structure.get_atoms()],
                                                     [reactant_structure.get_charge()],
                                                     [reactant_structure.get_multiplicity()],
                                                     deepcopy(settings_manager.calculator_settings.as_dict()))
        opt_name_product = self.optimize_structures("opt_product", [products_structure.get_atoms()],
                                                    [products_structure.get_charge()],
                                                    [products_structure.get_multiplicity()],
                                                    deepcopy(settings_manager.calculator_settings.as_dict()))
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

    def create_complex_or_minimum(self, graph, calculator_name):
        import scine_database as db
        label = db.Label.MINIMUM_OPTIMIZED if ";" not in graph else db.Label.COMPLEX_OPTIMIZED
        new_structure = self.create_new_structure(self.systems[calculator_name], label)
        self.transfer_properties(self.ref_structure, new_structure)
        self.store_energy(self.systems[calculator_name], new_structure)
        self.store_property(
            self._properties,
            "bond_orders",
            "SparseMatrixProperty",
            self.systems[calculator_name].get_results().bond_orders.matrix,
            self._calculation.get_model(),
            self._calculation,
            new_structure,
        )
        self.store_property(
            self._properties,
            "atomic_charges",
            "VectorProperty",
            self.systems[calculator_name].get_results().atomic_charges,
            self._calculation.get_model(),
            self._calculation,
            new_structure,
        )
        self.add_graph(new_structure, self.systems[calculator_name].get_results().bond_orders)

        results = self._calculation.get_results()
        results.add_structure(new_structure.id())
        self._calculation.set_results(self._calculation.get_results() + results)
        return new_structure

    def optimize_and_get_graphs_and_energies(self, fragment_base_name, fragments, charges, multiplicities,
                                             settings_manager):
        from copy import deepcopy
        opt_fragment_names = self.optimize_structures(fragment_base_name, fragments,
                                                      charges, multiplicities,
                                                      deepcopy(settings_manager.calculator_settings.as_dict()))
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
            fragment_energies.append(self.systems[name].get_results().energy)

        return opt_fragment_names, opt_f_graphs, fragment_energies

    def add_barrierless_reaction(self, opt_name, opt_f_graphs, opt_graph, opt_structure_ids,
                                 opt_fragment_names):
        import scine_database as db
        db_results = self._calculation.get_results()
        db_results.clear()
        fragment_structures = []
        for name, graph in zip(opt_fragment_names, opt_f_graphs):
            fragment_structures.append(self.create_complex_or_minimum(graph, name))

        if opt_structure_ids is None:
            opt_structure_ids = [self.create_complex_or_minimum(opt_graph, opt_name).id()]
        new_step = db.ElementaryStep()
        new_step.link(self._elementary_steps)
        new_step.create([s.id() for s in fragment_structures], opt_structure_ids)
        new_step.set_type(db.ElementaryStepType.BARRIERLESS)
        db_results.add_elementary_step(new_step.id())
        self._calculation.set_comment(self.name + ": Barrier-less reaction found.")
        self._calculation.set_results(self._calculation.get_results() + db_results)
        return opt_structure_ids

    def same_molecules(self, names_one, names_two):
        import scine_molassembler as masm
        graphs_one, charges_one, multies_one = self.get_sorted_graphs_charges_multiplicities(names_one)
        graphs_two, charges_two, multies_two = self.get_sorted_graphs_charges_multiplicities(names_two)

        total_graph_one = ";".join(graphs_one)
        total_graph_two = ";".join(graphs_two)

        return charges_one == charges_two and multies_one == multies_two and masm.JsonSerialization.equal_molecules(
            total_graph_one, total_graph_two)

    def get_sorted_graphs_charges_multiplicities(self, names_one):
        import scine_utilities as utils
        charges_one = []
        multies_one = []
        graphs_one = []
        for name in names_one:
            charges_one.append(self.systems[name].settings[utils.settings_names.molecular_charge])
            multies_one.append(self.systems[name].settings[utils.settings_names.spin_multiplicity])
            graphs_one.append(self.make_graph_from_calc(self.systems, name))
        graphs, charges, multiplicities = (
            list(start_val)
            for start_val in zip(*sorted(zip(
                graphs_one,
                charges_one,
                multies_one)))
        )
        return graphs, charges, multiplicities
