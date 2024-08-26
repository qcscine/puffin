# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List

from scine_puffin.config import Configuration
from scine_puffin.jobs.templates.job import calculation_context, job_configuration_wrapper
from scine_puffin.jobs.templates.scine_job import ScineJob
from scine_puffin.utilities.qm_mm_settings import prepare_optional_settings
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineSinglePoint(ScineJob):
    """
    A job calculating the electronic energy for a given structure with a given
    model.

    QM/MM and MM calculations expect the presence of bond orders ('bond_orders'), (optionally) atomic
    charges ('atomic_charges'), and the QM-atom selection ('qm_atoms') as properties of the structure.

    **Order Name**
      ``scine_single_point``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings that are recognized by the program chosen.
        Furthermore, all settings that are commonly understood by any program
        interface via the SCINE Calculator interface.

      Common examples are:

      max_scf_iterations : int
         The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - A program implementing the SCINE Calculator interface, i.e. Sparrow

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The ``electronic_energy`` associated with the given structure.
        The ``atomic_charges`` associated with the given structure (if available).
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Single Point Calculation Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_readuct as readuct

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        # actual calculation
        with calculation_context(self):
            prepare_optional_settings(structure, calculation, settings_manager, self._properties)
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_calc(keys[0], systems), calculation.get_settings())
            systems, success = readuct.run_sp_task(systems, keys, **settings_manager.task_settings)
            self.sp_postprocessing(success, systems, keys, structure, program_helper)

        return self.postprocess_calculation_context()

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "readuct", "utils"]
