# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import sys
from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from scine_puffin.utilities.task_to_readuct_call import SubTaskToReaductCall
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineReactTsGuess(ReactJob):
    __doc__ = ("""
    A job that tries to find an elementary step based on a transition state guess.

    The list of calculations/steps done is the following:

      1. Optimization of the TS guess to the actual TS.
      2. Hessian calculation of the TS structure
      3. Validate the TS using a combination of IRC scan and optimization of the
         resulting structures.
      4. Check if the IRC generated the input on one side and a new set of
         structures on the other side of the TS.
      5. Optimize the products separately.
      6. Store the new elementary step in the database.

    **Order Name**
      ``scine_react_ts_guess``

    **Required Input**
      A single structure that is the transition state guess.

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

       1. TS optimization: ``tsopt_*``
       2. Validation using an IRC scan: ``irc_*``
       3. Optimization of the structures obtained with the IRC scan : ``ircopt_*``
       4. Analyze supersystem, derive individual products and assign charges: ``sp_*``
       5. Optimization of the products and reactants: ``opt_*``

    """ + "\n"
               + ReactJob.optional_settings_doc() + "\n"
               + ReactJob.general_calculator_settings_docstring() + "\n"
               + ReactJob.generated_data_docstring() + "\n"
               + ReactJob.required_packages_docstring()
               )

    def __init__(self):
        super().__init__()
        self.name = "Scine React Job based on TS Guess"
        self.exploration_key = ""
        tsopt_defaults = {
            "output": ["ts"],
            "optimizer": "bofill",
            "convergence_max_iterations": 200,
            "automatic_mode_selection": []
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
        spin_propensity_defaults = {
            **self.settings[self.propensity_key],
            "ts_check": 2
        }
        self.settings = {
            **self.settings,
            "tsopt": tsopt_defaults,
            "irc": irc_defaults,
            "ircopt": ircopt_defaults,
            self.opt_key: opt_defaults,
            self.propensity_key: spin_propensity_defaults
        }

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            """ Prepare data structures """
            structures = calculation.get_structures()
            if len(structures) != 1:
                raise RuntimeError("The calculation must have exactly one structure.")
            ts_guess_structure = db.Structure(structures[0], self._structures)
            self.ref_structure = ts_guess_structure

            settings_manager, program_helper = self.create_helpers(ts_guess_structure)
            self.systems, keys = settings_manager.prepare_readuct_task(ts_guess_structure,
                                                                       self._calculation,
                                                                       self._calculation.get_settings(),
                                                                       config["resources"])
            self.sort_settings(settings_manager.task_settings)
            ts_guess_name = 'ts'
            self.systems[ts_guess_name] = self.get_system(keys[0])

            """ Propensity check for TS guess """
            # The setting does not make much sense here, because we don't know if the reaction is unimolecular
            if self.settings[self.propensity_key]["check"] and \
                    not self.settings[self.propensity_key]["check_for_unimolecular_reaction"]:
                sys.stderr.write("Warning: The setting 'check_for_unimolecular_reaction' is set to False, "
                                 "but the job does not have the information to decide if the reaction is unimolecular. "
                                 "The setting will be ignored.\n")
                self.settings[self.propensity_key]["check_for_unimolecular_reaction"] = True
            self.setup_automatic_mode_selection("tsopt")
            former_spin_propensity_range = self.settings[self.propensity_key]["check"]
            self.settings[self.propensity_key]["check"] = self.settings[self.propensity_key]["ts_check"]
            names, self.systems = self.optimize_structures(
                ts_guess_name,
                self.systems,
                [ts_guess_structure.get_atoms()],
                [ts_guess_structure.get_charge()],
                [ts_guess_structure.get_multiplicity()],
                settings_manager.calculator_settings,
                False,
                SubTaskToReaductCall.TSOPT,
                "tsopt"
            )
            if len(names) != 1:
                self.raise_named_exception("Optimization of the TS guess yielded multiple structures, "
                                           "which is not expected")

            lowest_name, names_within_range = self._get_propensity_names_within_range(
                names[0], self.systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
            )
            if lowest_name is None:
                self.raise_named_exception("No TS optimization was successful")
                raise RuntimeError("Unreachable")
            self.settings[self.propensity_key]["check"] = former_spin_propensity_range
            db_results = calculation.get_results()
            for name in [lowest_name] + names_within_range:
                # make sure to overwrite the output names with the new one
                self.settings["tsopt"]["output"] = [name]
                """ TSOPT Hessian IRC IRCOPT and Postprocessing"""
                try:
                    product_names, start_names = self._hess_irc_ircopt(name, settings_manager)
                    self._postprocessing_with_conformer_handling(product_names, start_names, program_helper)
                    # the postprocessing will remove previous results, so we have to keep track of them over the loop
                    db_results += calculation.get_results()
                except (BaseException, breakable.Break) as e:
                    if isinstance(e, BaseException):
                        sys.stderr.write(f"Reaction trial for {name} did not succeed: {e}\n")
            calculation.set_results(db_results)

        return self.postprocess_calculation_context()
