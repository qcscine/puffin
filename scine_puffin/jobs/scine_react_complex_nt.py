# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob
from ..utilities.task_to_readuct_call import SubTaskToReaductCall
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineReactComplexNt(ReactJob):
    __doc__ = ("""
    A job that tries to force a reaction, given a reactive complex and its parts.
    The reactive complex is expected to be generated as two structures placed
    next to one another. The job then forces groups of atoms onto or away from
    one another and analyzes the resulting structure(s) when relaxing.
    Only one pair of atom groups is allowed, and only repulsion or attraction
    of said groups can be screened at a time.

    The list of calculations/steps done is the following:

      1. Set up of a reactive complex based on the given specifications
      2. Enforce collision of structure (NT optimization), and extraction of a
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
      ``scine_react_complex_nt``

    **Required Input**
      The reactive complex (the structures it is made from) has to be defined in
      the list of structures handed to the task. If only one structure is
      provided, an intramolecular reaction is set up. Furthermore, the reactive
      sites in the complex that shall be pressed onto one another need to be
      given using:

      nt_nt_rhs_list : int
         This specifies list of indices of atoms to be forced onto
         or away from those in the LHS list.
      nt_nt_lhs_list : int
         This specifies list of indices of atoms to be forced onto
         or away from those in the RHS list.

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      All possible settings for this job are based on those available in SCINE
      Readuct. For a complete list see the
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

       1. Reactive complex generation ``rc_*``
       2. Newton trajectory scan: ``nt_*``
       3. TS optimization: ``tsopt_*``
       4. Validation using an IRC scan: ``irc_*``
       5. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       6. Analyze supersystem, derive individual products and assign charges: ``sp_*``
       7. Optimization of new products: ``opt_*``
       8. (Optional) optimization of the reactive complex: ``rcopt_*``

    """ + "\n"
               + ReactJob.optional_settings_doc() + "\n"
               + ReactJob.general_calculator_settings_docstring() + "\n"
               + ReactJob.generated_data_docstring() + "\n"
               + ReactJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine React Job with Newton Trajectory"
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
            self.opt_key: opt_defaults
        }

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            settings_manager, program_helper = self.reactive_complex_preparations()

            """ NT JOB """
            print("NT Settings:")
            print(self.settings["nt"], "\n")
            self.systems, success = self.observed_readuct_call(
                SubTaskToReaductCall.NT, self.systems, [self.rc_key], **self.settings["nt"]
            )
            if not success:
                self.verify_connection()
                """ Barrierless Reaction Check """
                if ';' in self.start_graph:
                    rc_opt_graph, _ = self.check_for_barrierless_reaction()
                else:
                    rc_opt_graph = None
                if rc_opt_graph is not None:
                    self.save_barrierless_reaction_from_rcopt(rc_opt_graph, program_helper)
                else:
                    calculation.set_comment(self.name + " NT Job: No TS guess found.")
                self.capture_raw_output()
                # update model because job will be marked complete
                # use start calculator because nt might have last failed calculation
                scine_helper.update_model(
                    self.get_system(self.rc_key), calculation, self.config
                )
                raise breakable.Break

            tsguess_name = self.output("nt")[0]
            try:
                self._tsopt_hess_irc_ircopt_postprocessing(tsguess_name, settings_manager, program_helper)
            except BaseException:
                _, tsguess_structure = self._store_ts_with_propensity_info(tsguess_name, program_helper,
                                                                           db.Label.TS_GUESS)
                self._calculation.set_restart_information("TS_GUESS", tsguess_structure.id())
                raise

        return self.postprocess_calculation_context()
