# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import scine_molassembler as masm

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob


class ScineReactComplexAfir(ReactJob):
    """
    A job that tries to enforce a reaction, given a reactive complex and its parts.
    The reactive complex is expected to be generated as two structures placed
    next to one another or, for intramolecular reactions, to be equal to the start structure.
    The job then forces the specified sites of the structure(s) onto one another and
    analyzes the resulting structure(s) when relaxing. Alternatively, pushing apart the
    specified sites can be evoked via the AFIR settings.

    The list of calculations/steps done is the following:

      1. Set up of a reactive complex based on the given specifications
      2. Enforce collision of structure (AFIR optimization)
      3. Relaxation of the AFIR result (free optimization)
      4. Check if new structures were formed. If not, stop.
      5. Optimization of a minimum energy pathway connecting start and end,
         generation of a transition state (TS) guess.
      6. Optimization of the TS guess to the actual TS.
      7. Hessian calculation of the TS structure
      8. Validate the TS using a combination of IRC scan and optimization of the
         resulting structures.
      9. Check if the IRC matches start and endpoint. If not, stop.
      10. Optimize the products separately.
      11. Store the new elementary step in the database.

    **Order Name**
      ``scine_react_complex_afir``

    **Required Input**
      The reactive complex, and the structures it was made from in the list of
      structures handed to the task. If only one structure is provided, an
      intramolecular reaction is set up. Furthermore, the reactive sites in the
      complex that shall be pressed onto one another need to be given using:

      afir_afir_rhs_list :: int
         This specifies list of indices of atoms to be artificially forced onto
         or away from those in the LHS list.
      afir_afir_lhs_list :: int
         This specifies list of indices of atoms to be artificially forced onto
         or away from those in the RHS list.

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

       1. AFIR optimization: ``afir_*``
       2. Relaxation of the AFIR result (free optimization): ``opt_*``
       3. Optimization of a minimum energy pathway: ``bspline_*``
       4. TS optimization: ``tsopt_*``
       5. Validation using an IRC scan: ``irc_*``
       6. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       7. Optimization of new products: ``opt_*``
       8. (Optional) optimization of the reactive complex: ``rcopt_*``

      The following options are available for the reactive complex generation:

      rc_x_alignment_0 :: List[float], length=9
          In case of two structures building the reactive complex, this option
          describes a rotation of the first structure (index 0) that aligns
          the reaction coordinate along the x-axis (pointing towards +x).
          The rotation assumes that the geometric mean position of all
          atoms in the reactive site (``afir_lhs_list``) is shifted into the
          origin.
      rc_x_alignment_1 :: List[float], length=9
          In case of two structures building the reactive complex, this option
          describes a rotation of the second structure (index 1) that aligns
          the reaction coordinate along the x-axis (pointing towards -x).
          The rotation assumes that the geometric mean position of all
          atoms in the reactive site (``afir_rhs_list``) is shifted into the
          origin.
      rc_x_rotation :: float
          In case of two structures building the reactive complex, this option
          describes a rotation angle around the x-axis of one of the two
          structures after ``rc_x_alignment_0`` and ``rc_x_alignment_1`` have
          been applied.
      rc_x_spread :: float
          In case of two structures building the reactive complex, this option
          gives the distance by which the two structures are moved apart along
          the x-axis after ``rc_x_alignment_0``, ``rc_x_alignment_1``, and
          ``rc_x_rotation`` have been applied.
      rc_displacement :: float
          In case of two structures building the reactive complex, this option
          adds a random displacement to all atoms (random direction, random
          length). The maximum length of this displacement (per atom) is set to
          be the value of this option.
      rc_spin_multiplicity :: int
          This option sets the ``spin_multiplicity`` of the reactive complex.
          In case this is not given the ``spin_multiplicity`` of the initial
          structure or minimal possible spin of the two initial structures is
          used.
      rc_molecular_charge :: int
          This option sets the ``molecular_charge`` of the reactive complex.
          In case this is not given the ``molecular_charge`` of the initial
          structure or sum of the charges of the initial structures is used.
          Note: If you set the ``rc_molecular_charge`` to a value different
          from the sum of the start structures charges the possibly resulting
          elementary steps will never be balanced but include removal or
          addition of electrons.
      rc_minimal_spin_multiplicity :: bool
          True: The total spin multiplicity in a bimolecular reaction is
          based on the assumption of total spin recombination (s + t = s; t + t = s; d + s = d; d + t = d)
          False: No spin recombination is assumed (s + t = t; t + t = quin; d + s = d; d + t = quar)
          (default: False)

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
        The transition state (TS) and also the separated products will be added
        to the database.

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
        self.name = "Scine React Job with AFIR"
        self.exploration_key = "afir"
        afir_defaults = {
            "output": ["afir"],
            "stop_on_error": False,
        }
        bspline_defaults = {
            "output": ["tsguess"],
            "extract_ts_guess": True,
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
            "output": ["end"],
            "convergence_max_iterations": 500,
        }
        self.settings = {
            **self.settings,
            "afir": afir_defaults,
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

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            settings_manager, program_helper = self.reactive_complex_preparations()

            """ AFIR Optimization """
            print("Afir Settings:")
            print(self.settings["afir"], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_afir_task', self.systems, [self.rc_key], **self.settings["afir"]
            )
            if not success:
                self.verify_connection()
                """ Barrierless Reaction Check """
                if ';' in self.start_graph:
                    rc_opt_graph, _ = self.check_for_barrierless_reaction()
                else:
                    rc_opt_graph = None
                if rc_opt_graph is not None:
                    self.save_barrierless_reaction(rc_opt_graph, program_helper)
                else:
                    calculation.set_comment(self.name + " AFIR Job: No TS guess found.")
                self.capture_raw_output()
                # update model because job will be marked complete
                # use start calculator because afir might have last failed calculation
                scine_helper.update_model(
                    self.systems[self.rc_key], calculation, self.config
                )
                raise breakable.Break

            """ Endpoint Optimization """
            inputs = self.output("afir")
            print("Endpoint Opt Settings:")
            print(self.settings["opt"], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_optimization_task', self.systems, inputs, **self.settings["opt"]
            )
            self.throw_if_not_successful(
                success,
                self.systems,
                inputs,
                ["energy"],
                "Endpoint optimization failed:\n",
            )

            """ Check whether we have new structure(s) """
            initial_charge = settings_manager.calculator_settings[utils.settings_names.molecular_charge]
            (
                _,
                self.end_graph,
                end_charges,
                _,
                _
            ) = self.get_graph_charges_multiplicities(self.output("opt")[0], initial_charge)

            print("Start Graph:")
            print(self.start_graph)
            print("End Graph:")
            print(self.end_graph)

            found_new_structures = bool(not masm.JsonSerialization.equal_molecules(self.start_graph, self.end_graph)
                                        or self.start_charges != end_charges)
            if not found_new_structures:
                self._calculation.set_comment("No new structure was discovered")
                scine_helper.update_model(
                    self.systems[self.output("opt")[0]], calculation, self.config
                )
                raise breakable.Break

            """ B-Spline Optimization """
            inputs = [self.rc_key] + self.output("opt")
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

            """ TS Optimization """
            inputs = self.output("bspline")
            self.setup_automatic_mode_selection("tsopt")
            print("TSOpt Settings:")
            print(self.settings["tsopt"], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_tsopt_task', self.systems, inputs, **self.settings["tsopt"]
            )
            self.throw_if_not_successful(
                success, self.systems, inputs, ["energy"], "TS optimization failed:\n"
            )

            """ TS Hessian """
            inputs = self.output("tsopt")
            self.systems, success = readuct.run_hessian_task(self.systems, inputs)
            self.throw_if_not_successful(
                success,
                self.systems,
                inputs,
                ["energy", "hessian", "thermochemistry"],
                "TS Hessian failed:\n",
            )

            """ IRC """
            inputs = self.output("tsopt")
            print("IRC Settings:")
            print(self.settings["irc"], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_irc_task', self.systems, inputs, **self.settings["irc"])

            """ IRC Opt"""
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
                start_structures = self._calculation.get_structures()

            self.react_postprocessing(product_names, program_helper, "tsopt", start_structures)

        return self.postprocess_calculation_context()
