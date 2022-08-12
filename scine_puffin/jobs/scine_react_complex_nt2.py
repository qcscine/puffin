# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob


class ScineReactComplexNt2(ReactJob):
    """
    A job that tries to force a reaction, given a reactive complex and its parts.
    The reactive complex is expected to be generated as two structures placed
    next to one another. The job then forces pairs of atoms onto or away from
    one another and analyzes the resulting structure(s) when relaxing.
    Multiple pairs and mixed choices for their movements are allowed in this
    version of the Newton Trajectory job.

    The list of calculations/steps done is the following:

      1. Set up of a reactive complex based on the given specifications
      2. Enforce collision of structure (NT2 optimization), and extraction of a
         transition state (TS) guess
      3. Optimization of the TS guess to the actual TS.
      4. Hessian calculation of the TS structure
      5. Validate the TS using a combination of IRC scan and optimization of the
         resulting structures.
      6. Check if the IRC generated the input on one side and a new set of
         structures on the other side of the TS.
      7. Optimize the products separately.
      8. Store the new elementary step in the database.

    **Order Name**
      ``scine_react_complex_nt2``

    **Required Input**
      The reactive complex (the structures it is made from) has to be defined in
      the list of structures handed to the task. If only one structure is
      provided, an intramolecular reaction is set up. Furthermore, the reactive
      sites in the complex that shall be pressed onto one another need to be
      given using:

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

       1. Reactive complex generation ``rc_*``
       2. Newton trajectory scan: ``nt_*``
       3. TS optimization: ``tsopt_*``
       4. Validation using an IRC scan: ``irc_*``
       5. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       6. Optimization of new products: ``opt_*``
       7. (Optional) optimization of the reactive complex: ``rcopt_*``

      The following options are available for the reactive complex generation:

      rc_x_alignment_0 :: List[float], length=9
          In case of two structures building the reactive complex, this option
          describes a rotation of the first structure (index 0) that aligns
          the reaction coordinate along the x-axis (pointing towards +x).
          The rotation assumes that the geometric mean position of all
          atoms in the reactive site (``nt_lhs_list``) is shifted into the
          origin.
      rc_x_alignment_1 :: List[float], length=9
          In case of two structures building the reactive complex, this option
          describes a rotation of the second structure (index 1) that aligns
          the reaction coordinate along the x-axis (pointing towards -x).
          The rotation assumes that the geometric mean position of all
          atoms in the reactive site (``nt_rhs_list``) is shifted into the
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
        self.name = "Scine React Job with Newton Trajectory 2"
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
            "nt": nt_defaults,
            "tsopt": tsopt_defaults,
            "irc": irc_defaults,
            "ircopt": ircopt_defaults,
            "opt": opt_defaults
        }

    @job_configuration_wrapper
    def run(self, _, calculation, config: Configuration) -> bool:

        import scine_readuct as readuct
        import scine_utilities as utils

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            settings_manager, program_helper = self.reactive_complex_preparations()

            """ NT JOB """
            print("NT Settings:")
            print(self.settings["nt"], "\n")
            self.systems, success = self.observed_readuct_call(
                'run_nt2_task', self.systems, [self.rc_key], **self.settings["nt"])
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
                    calculation.set_comment(self.name + " NT Job: No TS guess found.")
                self.capture_raw_output()
                # update model because job will be marked complete
                # use start calculator because nt might have last failed calculation
                scine_helper.update_model(
                    self.systems[self.rc_key], calculation, self.config
                )
                raise breakable.Break

            """ TSOPT JOB """
            inputs = self.output("nt")
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
