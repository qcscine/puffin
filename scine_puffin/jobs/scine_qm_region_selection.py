# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, TYPE_CHECKING

from scine_puffin.config import Configuration
from scine_puffin.jobs.templates.job import calculation_context, job_configuration_wrapper
from scine_puffin.jobs.templates.scine_job import ScineJob
from scine_puffin.utilities.qm_mm_settings import prepare_optional_settings, is_qm_mm
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineQmRegionSelection(ScineJob):
    """
    This job implements the QM region selection presented in J. Chem. Theory Comput. 2021, 17, 3797-3813.
    In this approach, the QM region is selected such that an error in the forces acting on a manually selected
    set of atoms is minimized. For this purpose, a reference calculation for a very large QM region is run, then
    the QM region is expanded systematically to generate a set of model systems. For these model systems, we calculate
    the differences to the reference forces and select the smallest model system that is within 20% of the smallest
    error of all model systems.

    The calculation requires the presence of bond orders ('bond_orders') and (optionally) atomic
    charges ('atomic_charges'). Upon job completion, the "optimal" QM region is saved as a property ('qm_atoms').

    **Order Name**
      ``scine_qm_region_selection``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings that are recognized by the program chosen.
        Furthermore, all settings that are commonly understood by any program
        interface via the SCINE Calculator interface.

      Common examples are:

      electrostatic_embedding : bool
         Use electrostatic embedding.
      qm_region_center_atoms : List[int]
         The indices of the atoms for which the forces are converged.
      initial_radius : float
         The radius of the smallest/initial QM region around the selected atoms.
      cutting_probability : float
         A parameter that controls the random construction of QM regions, controlling the probability to cut bonds
         during the QM region expansion. If this is set to 1.0, the QM region is fixed by the radius and not sampled.
      tol_percentage_error : float
         Error percentage to tolerate with respect to the smallest error encountered in the candidate QM/MM models.
      qm_region_max_size : int
         Maximum number of atoms in the QM region.
      qm_region_min_size : int
         Minimum number of atoms in the QM region.
      ref_max_size : int
        Maximum number of atoms in the QM region of the reference calculation.
      tol_percentage_sym_score : float
        Only roughly symmetric QM regions. This score determines the acceptable tolerance.


    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - SCINE: Swoose
      - A program implementing the SCINE Calculator interface, e.g. Sparrow

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The `qm_atoms` selected by the algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine QM Region Selection Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_swoose as swoose

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        # actual calculation
        with calculation_context(self):

            if not is_qm_mm(calculation.get_model()):
                raise RuntimeError("QM region selection for QM/MM is only possible if the electronic structure model"
                                   "/method family is of type QM/MM. Your method family is: "
                                   + calculation.get_model().method_family)

            prepare_optional_settings(structure, calculation, settings_manager, self._properties, skip_qm_atoms=True)
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            calculator = self.get_calc(keys[0], systems)
            if program_helper is not None:
                program_helper.calculation_preprocessing(calculator, calculation.get_settings())

            qm_region_selector = swoose.QmRegionSelector()
            qm_region_selector.set_underlying_calculator(calculator)
            qm_region_selector.settings.update(settings_manager.task_settings)
            qm_region_selector.settings["mm_connectivity_file"] = calculator.settings["mm_connectivity_file"]
            qm_region_selector.settings["mm_parameter_file"] = calculator.settings["mm_parameter_file"]
            qm_region_selector.generate_qm_region(structure.get_atoms())
            qm_atom_indices = qm_region_selector.get_qm_region_indices()
            print("QM-region indices: ", qm_atom_indices, "\nNumber of atoms ", len(qm_atom_indices))
            self.save_results(structure, qm_atom_indices)
        return self.postprocess_calculation_context()

    def save_results(self, structure: db.Structure, qm_atom_indices: List[int]) -> None:
        self.store_property(
            self._properties,
            "qm_atoms",
            "VectorProperty",
            qm_atom_indices,
            self._calculation.get_model(),
            self._calculation,
            structure,
        )

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "swoose", "utils"]
