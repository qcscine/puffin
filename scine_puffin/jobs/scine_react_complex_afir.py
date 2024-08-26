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


class ScineReactComplexAfir(ReactJob):
    __doc__ = ("""
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

      afir_afir_rhs_list : int
         This specifies list of indices of atoms to be artificially forced onto
         or away from those in the LHS list.
      afir_afir_lhs_list : int
         This specifies list of indices of atoms to be artificially forced onto
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

       1. AFIR optimization: ``afir_*``
       2. Relaxation of the AFIR result (free optimization): ``opt_*``
       3. Optimization of a minimum energy pathway: ``bspline_*``
       4. TS optimization: ``tsopt_*``
       5. Validation using an IRC scan: ``irc_*``
       6. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       7. Analyze supersystem, derive individual products and assign charges: ``sp_*``
       8. Optimization of new products: ``opt_*``
       9. (Optional) optimization of the reactive complex: ``rcopt_*``

    """ + "\n"
               + ReactJob.optional_settings_doc() + "\n"
               + ReactJob.general_calculator_settings_docstring() + "\n"
               + ReactJob.generated_data_docstring() + "\n"
               + ReactJob.required_packages_docstring()
               )

    def __init__(self) -> None:
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
            self.opt_key: opt_defaults
        }

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_molassembler as masm
        import scine_readuct as readuct
        import scine_utilities as utils

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            settings_manager, program_helper = self.reactive_complex_preparations()

            """ AFIR Optimization """
            print("Afir Settings:")
            print(self.settings["afir"], "\n")
            self.systems, success = self.observed_readuct_call(
                SubTaskToReaductCall.AFIR, self.systems, [self.rc_key], **self.settings["afir"]
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
                    calculation.set_comment(self.name + " AFIR Job: No TS guess found.")
                self.capture_raw_output()
                # update model because job will be marked complete
                # use start calculator because afir might have last failed calculation
                scine_helper.update_model(
                    self.get_system(self.rc_key), calculation, self.config
                )
                raise breakable.Break

            """ Endpoint Optimization """
            inputs = self.output("afir")
            print("Endpoint Opt Settings:")
            print(self.settings[self.opt_key], "\n")
            self.systems, success = self.observed_readuct_call(
                SubTaskToReaductCall.OPT, self.systems, inputs, **self.settings[self.opt_key]
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
            ) = self.get_graph_charges_multiplicities(self.output(self.opt_key)[0], initial_charge)

            print("Start Graph:")
            print(self.start_graph)
            print("End Graph:")
            print(self.end_graph)
            found_new_structures = bool(not masm.JsonSerialization.equal_molecules(self.start_graph, self.end_graph)
                                        or self.start_charges != end_charges)
            if not found_new_structures:
                self._calculation.set_comment("No new structure was discovered")
                scine_helper.update_model(
                    self.get_system(self.output(self.opt_key)[0]), calculation, self.config
                )
                raise breakable.Break

            """ B-Spline Optimization """
            inputs = [self.rc_key] + self.output(self.opt_key)
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
            tsguess_name = self.output("bspline")[0]
            try:
                self._tsopt_hess_irc_ircopt_postprocessing(tsguess_name, settings_manager, program_helper)
            except BaseException:
                _, tsguess_structure = self._store_ts_with_propensity_info(tsguess_name, program_helper,
                                                                           db.Label.TS_GUESS)
                self._calculation.set_restart_information("TS_GUESS", tsguess_structure.id())
                raise

        return self.postprocess_calculation_context()
