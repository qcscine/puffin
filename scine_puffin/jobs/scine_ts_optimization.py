# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_optimization_job import OptimizationJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineTsOptimization(OptimizationJob):
    """
    A job searching the nearest saddlepoint on the potential energy surface.
    Optimizing a given structure's geometry, generating a new transition state
    structure, if successful.

    **Order Name**
      ``scine_ts_optimization``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings recognized by ReaDuct's transition state search

      Common examples are:

      optimizer : str
         The name of the optimizer to be used, e.g. 'bofill' or 'evf' or 'dimer'.
      convergence_max_iterations : int
         The maximum number of geometry optimization cycles.
      convergence_delta_value : float
         The convergence criterion for the electronic energy difference between
         two steps.
      convergence_gradient_max_coefficient : float
         The convergence criterion for the maximum absolute gradient.
         contribution.
      convergence_step_rms : float
         The convergence criterion for root mean square of the geometric
         gradient.
      convergence_step_max_coefficient : float
         The convergence criterion for the maximum absolute coefficient in the
         last step taken in the geometry optimization.
      convergence_gradient_rms : float
         The convergence criterion for root mean square of the last step taken
         in the geometry optimization.

      For a complete list see the
      `ReaDuct manual <https://scine.ethz.ch/download/readuct>`_

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
        A new transition state structure.
      Properties
        The ``electronic_energy`` associated with the new structure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Transition State Optimization"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        import scine_readuct as readuct

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        # actual calculation
        with calculation_context(self):
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_calc(keys[0], systems), calculation.get_settings())
            systems, success = readuct.run_tsopt_task(systems, keys, **settings_manager.task_settings)

            self.optimization_postprocessing(
                success, systems, keys, structure, db.Label.TS_OPTIMIZED, program_helper
            )

        return self.postprocess_calculation_context()
