# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_hessian_job import HessianJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineHessian(HessianJob):
    """
    A job generating a Hessian and derived data for a single structure.
    Derived data means the eigenvalues (frequencies) and eigenvectors
    (normalmodes) as well as thermochemical data (Gibbs free energy).

    **Order Name**
      ``scine_hessian``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings recognized by ReaDuct's Hessian task. For a complete list see
      the
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

      Properties
        The ``hessian`` (``DenseMatrixProperty``), ``frequencies``
        (``VectorProperty``), ``normal_modes`` (``DenseMatrixProperty``),
        ``gibbs_energy_correction`` (``NumberProperty``) and
        ``gibbs_free_energy` (``NumberProperty``) will be provided.
        Optionally the ``electronic_energy`` associated with the structure if it
        is present in the results of provided by the calculator interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Hessian Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        import scine_readuct as readuct

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        if len(structure.get_atoms()) < 2:
            calculation.set_comment("No Hessian generated for a single atom.\n")
            calculation.set_status(db.Status.FAILED)
            return False

        # actual calculation
        with calculation_context(self):
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_calc(keys[0], systems), calculation.get_settings())
            systems, success = readuct.run_hessian_task(systems, keys, **settings_manager.task_settings)

            self.sp_postprocessing(success, systems, keys, structure, program_helper)

            self.store_hessian_data(systems[keys[0]], structure)

        return self.postprocess_calculation_context()
