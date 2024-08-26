# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from ..utilities import scine_helper
from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_optimization_job import OptimizationJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineIrcScan(OptimizationJob):
    """
    A job scanning a single intrinsic reaction coordinate.

    **Order Name**
      ``scine_irc_scan``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings recognized by ReaDuct's IRC task. For a complete list see
      the
      `ReaDuct manual <https://scine.ethz.ch/download/readuct>`_

      Common examples are:

      stop_on_error : bool
         If ``False``, the optimization does not need to fully converge but will
         be accepted as a success even if it reaches the maximum amounts of
         optimization cycles. Also, the resulting structures will be flagged as
         ``minimum_guess`` if this option is set ot be ``False``.
         (Default: ``True``)
      irc_mode : int
         The mode to follow during the IRC scan. By default, the first mode (0).
         (mode with the larges imaginary frequency will be followed).

      All settings that are recognized by the SCF program chosen.

      Common examples are:

      max_scf_iterations : int
         The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - A program implementing the SCINE Calculator interface, e.g. Sparrow

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Structures
        Both the forward and backward optimized structures will be added to the
        database. They will be flagged as: ``minimum_optimized`` or
        ``minimum_guess`` if ``stop_on_error`` is set to ``False``.

      Properties
        The ``electronic_energy`` associated with both forward and backward
        structures.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine IRC Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        import scine_readuct as readuct

        # Get structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        # actual calculation
        with calculation_context(self):
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_calc(keys[0], systems), calculation.get_settings())
            # Task specific operation
            settings_manager.task_settings["output"] = ["forward", "backward"]
            stop_on_error = settings_manager.task_settings.get("stop_on_error", True)
            systems, success = readuct.run_irc_task(systems, keys, **settings_manager.task_settings)

            self.verify_connection()  # check valid connection
            results_check, results_err = self.expected_results_check(systems, ["forward", "backward"])
            if not results_check:
                self.raise_named_exception(results_err)
            scine_helper.update_model(self.get_calc(keys[0], systems), self._calculation, config)

            is_surface = structure.has_property("surface_atom_indices")
            label = db.Label.SURFACE_OPTIMIZED if is_surface else db.Label.MINIMUM_OPTIMIZED
            if not success and not stop_on_error:
                label = db.Label.SURFACE_GUESS if is_surface else db.Label.MINIMUM_GUESS
                calculation.set_comment(
                    "Optimization did not fully converge for one or both sides. 'forward' and "
                    "'backward' structures are stored as '"
                    + db.Label.MINIMUM_GUESS.name
                    + "'."
                )

            # clear results
            db_results = calculation.get_results()
            db_results.clear()
            calculation.set_results(db_results)
            for name in ["forward", "backward"]:
                # structure
                new_structure = self.create_new_structure(systems[name], label)
                db_results.add_structure(new_structure.id())

                # properties
                self.store_energy(self.get_calc(name, systems), new_structure)
                self.transfer_properties(structure, new_structure)

                if program_helper is not None:
                    program_helper.calculation_postprocessing(
                        calculation, structure, new_structure
                    )

        calculation.set_results(calculation.get_results() + db_results)
        return self.postprocess_calculation_context()
