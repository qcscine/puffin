# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Dict, Optional, TYPE_CHECKING

from scine_puffin.config import Configuration
from scine_puffin.jobs.templates.job import calculation_context, job_configuration_wrapper
from scine_puffin.jobs.templates.scine_job import ScineJob
from scine_puffin.utilities.imports import module_exists, MissingDependency
from scine_puffin.utilities.qm_mm_settings import prepare_optional_settings, is_qm_mm
from scine_puffin.utilities.scine_helper import SettingsManager

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    db = MissingDependency("scine_utilities")


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

      excluded_residues :: List[str]
        List of residue labels that will be excluded from the QM-region selection. Example application: Avoid including
        solvent molecules into the QM region. Note that multiplicities and charges deviating from the calculator
        defaults for the QM-selection fragment must be provided through the calculation settings.

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
        self.__structure_represents_full_system = True
        self.__atom_info_property_name = "atom_info"
        self.__atom_info_file_name = "atom-info.dat"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_swoose as swoose

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        # actual calculation
        with (calculation_context(self)):
            if not is_qm_mm(calculation.get_model()):
                raise RuntimeError("QM region selection for QM/MM is only possible if the electronic structure model"
                                   "/method family is of type QM/MM. Your method family is: "
                                   + calculation.get_model().method_family)

            settings = calculation.get_settings()
            atom_collection = prepare_optional_settings(structure, calculation, settings_manager, self._properties,
                                                        skip_qm_atoms=True)
            print("Creating Calculator")
            systems, keys = self.__prepare_calculator(settings_manager, calculation.get_settings(),
                                                      calculation.get_model(), config["resources"], atom_collection,
                                                      structure if self.__structure_represents_full_system else None)
            print("Calculator Done!")
            calculator = systems[keys[0]]
            if program_helper is not None:
                program_helper.calculation_preprocessing(calculator, calculation.get_settings())

            print("Creating QM region selection")
            qm_region_selector = swoose.QmRegionSelector()
            qm_region_selector.set_underlying_calculator(calculator)
            qm_region_selector.settings.update(settings_manager.task_settings)
            qm_region_selector.settings["mm_connectivity_file"] = calculator.settings["mm_connectivity_file"]
            qm_region_selector.settings["mm_parameter_file"] = calculator.settings["mm_parameter_file"]
            if structure.has_property(self.__atom_info_property_name):
                print("Preparing atomic information file with partial charges")
                prop = db.StringProperty(structure.get_property(self.__atom_info_property_name), self._properties)
                with open(self.__atom_info_file_name, "w") as atom_info_file:
                    atom_info_file.write(prop.get_data())
                qm_region_selector.settings["atomic_info_file"] = self.__atom_info_file_name
            if "excluded_residues" in settings:
                qm_region_selector.settings["excluded_residue_labels"] = settings["excluded_residues"]

            qm_region_selector.generate_qm_region(atom_collection)
            print("QM region selection done!")
            qm_atom_indices = qm_region_selector.get_qm_region_indices()
            print("QM-region indices: ", qm_atom_indices, "\nNumber of atoms ", len(qm_atom_indices))
            self.save_results(structure, qm_atom_indices)
        return self.postprocess_calculation_context()

    @staticmethod
    def __prepare_calculator(settings_manager: SettingsManager, settings: utils.ValueCollection, model: db.Model,
                             resources: Dict, atom_collection: utils.AtomCollection, structure: Optional[db.Structure]):
        settings_manager.separate_settings(settings)
        settings_manager.update_calculator_settings(structure=structure, model=model, resources=resources)
        settings_manager.correct_non_applicable_settings()
        utils.io.write("system.xyz", atom_collection)
        system = utils.core.load_system_into_calculator(
            "system.xyz", model.method_family, **settings_manager.calculator_settings
        )
        return {"system": system}, ["system"]

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
