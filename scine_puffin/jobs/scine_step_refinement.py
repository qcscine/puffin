# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob
from scine_puffin.utilities import masm_helper


class ScineStepRefinement(ReactJob):
    """
    A job that tries to refine an elementary step using a different electronic
    structure method. The job is essentially a scine_react job that uses a
    previously optimized transition state as an initial guess.

    The list of calculations/steps done is the following:

      1. Optimization of the reactant structures using the refinement model.
      2. Optimization of the TS guess / old TS to the actual TS.
      3. Hessian calculation of the TS structure
      4. Validate the TS using a combination of IRC scan and optimization of the
         resulting structures.
      5. Check if the IRC generated the input on one side and a new set of
         structures on the other side of the TS.
      6. Optimize the products separately.
      7. Store the new elementary step in the database.

    **Order Name**
      ``scine_step_refinement``

    **Required Input**
      The first up to two structures correspond to the reactants of the reaction.
      The original transition state must be given through the auxiliaries of the calculation as
       "transition-state-id": id-of-transition state. Furthermore, the reactive sites in the complex
      that shall be pressed onto one another need to be given using:

      nt_nt_associations :: int
         This specifies list of indices of atoms pairs to be forced onto
         another. Pairs are given in a flat list such that indices ``2i``
         and ``2i+1`` form a pair.
      nt_nt_dissociations :: int
         This specifies list of indices of atoms pairs to be forced away from
         another. Pairs are given in a flat list such that indices ``2i``
         and ``2i+1`` form a pair.

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

       1. Newton trajectory scan: ``nt_*`` (used exclusively for the mode selection
        in the transition state optimization)
       2. TS optimization: ``tsopt_*``
       3. Validation using an IRC scan: ``irc_*``
       4. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       5. Optimization of the products and reactants: ``opt_*``

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
        self.name = "Scine React Job starting from previous transition state"
        self.exploration_key = "nt"
        nt_defaults = {
            "output": ["nt"],
            "stop_on_error": False,
        }
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
        self.settings = {
            **self.settings,
            "tsopt": tsopt_defaults,
            "irc": irc_defaults,
            "ircopt": ircopt_defaults,
            "opt": opt_defaults,
            "nt": nt_defaults,
        }

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_readuct as readuct
        import scine_utilities as utils
        import scine_database as db
        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            settings_manager, program_helper = self.reactive_complex_preparations()

            all_struc_ids = self._calculation.get_structures()
            ts_struc = db.Structure(calculation.get_auxiliaries()["transition-state-id"], self._structures)
            start_structures = [db.Structure(ident, self._structures) for ident in all_struc_ids]
            settings_manager.separate_settings(self._calculation.get_settings())
            self.sort_settings(settings_manager.task_settings)
            """ TSOPT JOB """
            ts_guess, keys = settings_manager.prepare_readuct_task(ts_struc,
                                                                   self._calculation,
                                                                   self._calculation.get_settings(),
                                                                   config["resources"])
            self.systems[keys[0]] = ts_guess[keys[0]]
            self.setup_automatic_mode_selection("tsopt")
            print("TSOpt Settings:")
            print(self.settings["tsopt"], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_tsopt_task', self.systems, keys, **self.settings["tsopt"])
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
            if start_names is not None:
                start_structures = self.store_start_structures(
                    start_names, program_helper, "tsopt")
            else:
                start_names, start_structure_objects = self.optimize_reactants(start_structures,
                                                                               settings_manager,
                                                                               config)
                start_structures = [o.id() for o in start_structure_objects]

            """ Save the elementary step, transition state, and product """
            self.react_postprocessing(product_names, program_helper, "tsopt", start_structures)

        return self.postprocess_calculation_context()

    def optimize_reactants(self, reactant_structures, settings_manager, config):
        """
        Optimize the reactant structures and saves them in the database.

        Notes
        -----
        * writes reactant calculators to self.systems
        * May throw exception.

        Parameters
        ----------
        reactant_structures :: List[scine_database.Structure]
            The original structures of the elementary step to be optimized.
        settings_manager :: SettingsManager
            The settings_manager which is used to construct the reactant calculators.
        config :: : scine_puffin.config.Configuration
            The run configuration.

        Returns
        -------
        reactant_names :: List[str]
            The names of the reactant calculators in self.systems.
        optimized_structures :: List[scine_database.Structure]
            The optimized reactant structures.
        """
        import scine_readuct as readuct
        import scine_utilities as utils
        import scine_database as db

        print("Reactant Opt Settings:")
        print(self.settings["opt"], "\n")

        reactant_names = []
        optimized_structures = []
        # Optimize reactants, if they have more than one atom; otherwise just run a single point calculation
        for i, structure in enumerate(reactant_structures):
            # Build the initial calculator.
            tmp_calculator_set, keys = settings_manager.prepare_readuct_task(structure, self._calculation,
                                                                             self._calculation.get_settings(),
                                                                             config["resources"])
            name = "reactant_opt_{:02d}".format(i)
            reactant_names.append(name)
            self.systems[name] = tmp_calculator_set[keys[0]]
            print("Optimizing " + name + " :\n")
            # Run the optimization if more than one atom is present.
            if len(self.systems[name].structure) > 1:
                self.systems, success = self.observed_readuct_call(
                    'run_opt_task', self.systems, [name], **self.settings["opt"])
                self.throw_if_not_successful(
                    success,
                    self.systems,
                    [name],
                    ["energy"],
                    "Reactant optimization failed:\n",
                )

            # Calculate the bond orders
            self.systems, success = readuct.run_single_point_task(
                self.systems,
                [name],
                spin_propensity_check=self.settings[self.job_key]["spin_propensity_check"],
                require_bond_orders=True,
            )
            self.throw_if_not_successful(
                success,
                self.systems,
                [name],
                ["energy", "bond_orders"],
                "Reactant optimization failed:\n",
            )

            pbc_string = self.systems[name].settings.get(utils.settings_names.periodic_boundaries, "")
            masm_results = masm_helper.get_molecules_result(
                self.systems[name].structure,
                self.make_bond_orders_from_calc(self.systems, name),
                self.connectivity_settings,
                pbc_string,
            )
            structure_label = db.Label.MINIMUM_OPTIMIZED
            if len(masm_results.molecules) > 1:
                structure_label = db.Label.COMPLEX_OPTIMIZED

            new_structure = self.create_new_structure(self.systems[name], structure_label)
            self.transfer_properties(structure, new_structure)
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
            optimized_structures.append(new_structure)
        return reactant_names, optimized_structures
