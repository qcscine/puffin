# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from copy import deepcopy
from typing import TYPE_CHECKING, List

import numpy as np

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_job import ScineJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineConceptualDft(ScineJob):
    __doc__ = ("""
    A job calculating conceptual DFT properties for a given structure with a given
    model. The properties are extracted from the atomic charges and energies
    of the given structure, as well the structure with the same geometry but
    one additional electron or one electron less based on the finite difference
    approximation.

    **Order Name**
      ``scine_conceptual_dft``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      spin_multiplicity_plus : int
          The spin multiplicity of the system with one additional electron.
          If not specified, the additional electron is assumed to pair up with
          any priorly existing unpaired electrons.
          I.e. the multiplicity is assumed to decrease by one for all open-shell
          structures and to be two if the start structure is closed-shell.
      spin_multiplicity_minus : int
          The spin multiplicity of the system with one electron less.
          If not specified, the deducted electron is assumed to be an unpaired
          one with the others not rearranging. I.e. the multiplicity is assumed
          to decrease by one for all open-shell structures and to be two if the
          start structure is closed-shell.

    """ + "\n"
               + ScineJob.optional_settings_doc() + "\n"
               + ScineJob.general_calculator_settings_docstring() + "\n"
               + ScineJob.generated_data_docstring() + "\n" +
               """
      If successful the following data will be generated and added to the
      database:

      Properties
        The condensed ``dual_descriptor`` indices associated with the atoms of
        the given structure.
        The condensed Fukui indices ``fukui_plus``, ``fukui_minus`` and
        ``fukui_radical`` associated with the atoms of the given structure.
        The ``chemical_potential`` associated with the given structure.
        The ``electronegativity`` associated with the given structure.
        The ``electrophilicity`` associated with the given structure.
        The ``hardness`` associated with the given structure.
        The ``softness`` associated with the given structure.
    """
               + ScineJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Conceptual DFT Calculation Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_readuct as readuct
        import scine_utilities as utils

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)
        calculation_settings = calculation.get_settings()

        # actual calculation
        success = False  # success might not be set if something throws in context -> ensure it exists in scope
        with calculation_context(self):

            # Get charge and multiplicity for systems with one electron less/extra
            # Plus denotes the case with one additional electron "N+1", minus with one electron less "N-1"
            charge = structure.get_charge()
            multiplicity = structure.get_multiplicity()

            # Check that N is >= 1 s.t. N-1 is >= 0
            n_electrons = 0
            elements = structure.get_atoms().elements
            for element in elements:
                n_electrons += utils.ElementInfo.Z(element)
            if charge >= n_electrons:
                raise RuntimeError("At least one electron required to calculate cDFT properties.")

            charge_plus = charge - 1
            charge_minus = charge + 1

            if "spin_multiplicity_plus" in calculation_settings:
                multiplicity_plus: int = calculation_settings["spin_multiplicity_plus"]  # type: ignore
                del calculation_settings["spin_multiplicity_plus"]
            else:
                multiplicity_plus = 2 if multiplicity == 1 else multiplicity - 1

            if "spin_multiplicity_minus" in calculation_settings:
                multiplicity_minus: int = calculation_settings["spin_multiplicity_minus"]  # type: ignore
                del calculation_settings["spin_multiplicity_minus"]
            else:
                multiplicity_minus = 2 if multiplicity == 1 else multiplicity - 1

            # Prepare calculations
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation_settings, config["resources"]
            )
            # Charges are required for localized descriptors
            settings_manager.task_settings["require_charges"] = True

            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_calc(keys[0], systems), calculation_settings)

            # N electron structure/structure of interest
            print("N ELECTRON CALCULATION")
            systems, success = readuct.run_sp_task(systems, keys, **settings_manager.task_settings)
            # Checks connection, success and whether expected results exist
            self.throw_if_not_successful(
                success, systems, keys, [
                    "energy", "atomic_charges"], "Single point calculation on N electron system failed.")
            energy = self.get_energy(self.get_calc(keys[0], systems))
            atomic_charges = self.get_calc(keys[0], systems).get_results().atomic_charges
            assert atomic_charges

            # N+1 electron "plus" system
            print("N+1 ELECTRON CALCULATION")
            print(f"Charge: {charge_plus} Multiplicity: {multiplicity_plus}")
            plus_calculator_settings = deepcopy(settings_manager.calculator_settings)
            plus_calculator_settings[utils.settings_names.molecular_charge] = charge_plus
            plus_calculator_settings[utils.settings_names.spin_multiplicity] = multiplicity_plus
            systems["plus"] = utils.core.load_system_into_calculator(
                "system.xyz", calculation.get_model().method_family, **plus_calculator_settings)
            systems, success = readuct.run_sp_task(systems, ["plus"], **settings_manager.task_settings)
            self.throw_if_not_successful(
                success, systems, ["plus"], [
                    "energy", "atomic_charges"], "Single point calculation on N+1 electron system failed.")
            energy_plus = self.get_energy(self.get_calc("plus", systems))
            atomic_charges_plus = self.get_calc("plus", systems).get_results().atomic_charges
            assert atomic_charges_plus

            # N-1 electron "minus" system
            print("N-1 ELECTRON CALCULATION")
            print(f"Charge: {charge_minus} Multiplicity: {multiplicity_plus}")
            minus_calculator_settings = deepcopy(settings_manager.calculator_settings)
            minus_calculator_settings[utils.settings_names.molecular_charge] = charge_minus
            minus_calculator_settings[utils.settings_names.spin_multiplicity] = multiplicity_minus
            systems["minus"] = utils.core.load_system_into_calculator(
                "system.xyz", calculation.get_model().method_family, **minus_calculator_settings)
            systems, success = readuct.run_sp_task(systems, ["minus"], **settings_manager.task_settings)
            self.throw_if_not_successful(
                success, systems, ["minus"], [
                    "energy", "atomic_charges"], "Single point calculation on N-1 electron system failed.")
            energy_minus = self.get_energy(self.get_calc("minus", systems))
            atomic_charges_minus = self.get_calc("minus", systems).get_results().atomic_charges
            assert atomic_charges_minus

            # Deduce conceptual DFT properties
            print("cDFT PROPERTIES")
            cDFT_container = utils.conceptual_dft.calculate(
                energy, np.asarray(atomic_charges),
                energy_plus, np.asarray(atomic_charges_plus),
                energy_minus, np.asarray(atomic_charges_minus)
            )

            # Calculation postprocessing
            self.verify_connection()
            # clear existing results
            db_results = self._calculation.get_results()
            db_results.clear()
            self._calculation.set_results(db_results)
            # update model
            scine_helper.update_model(self.get_calc(keys[0], systems), self._calculation, self.config)

            # Store cDFT properties
            # Localized
            self.store_property(
                self._properties,
                "dual_descriptor",
                "VectorProperty",
                cDFT_container.local_v.dual_descriptor,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "fukui_plus",
                "VectorProperty",
                cDFT_container.local_v.fukui_plus,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "fukui_minus",
                "VectorProperty",
                cDFT_container.local_v.fukui_minus,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "fukui_radical",
                "VectorProperty",
                cDFT_container.local_v.fukui_radical,
                self._calculation.get_model(),
                self._calculation,
                structure)
            # Global
            self.store_property(
                self._properties,
                "chemical_potential",
                "NumberProperty",
                cDFT_container.global_v.chemical_potential,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "electronegativity",
                "NumberProperty",
                cDFT_container.global_v.electronegativity,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "electrophilicity",
                "NumberProperty",
                cDFT_container.global_v.electrophilicity,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "hardness",
                "NumberProperty",
                cDFT_container.global_v.hardness,
                self._calculation.get_model(),
                self._calculation,
                structure)
            self.store_property(
                self._properties,
                "softness",
                "NumberProperty",
                cDFT_container.global_v.softness,
                self._calculation.get_model(),
                self._calculation,
                structure)

            # Print results to raw output
            print("\n{:30s}: {:6f}".format("Chemical potential", cDFT_container.global_v.chemical_potential))
            print("{:30s}: {:6f}".format("Electronegativity", cDFT_container.global_v.electronegativity))
            print("{:30s}: {:6f}".format("Electrophilicity", cDFT_container.global_v.electrophilicity))
            print("{:30s}: {:6f}".format("Hardness", cDFT_container.global_v.hardness))
            print("{:30s}: {:6f}".format("Softness", cDFT_container.global_v.softness))
            print("\nDual descriptor")
            print("\n".join("{:6d} {:2s}    :{:12.6f}".format(
                i, str(elements[i]), v) for i, v in enumerate(cDFT_container.local_v.dual_descriptor)))
            print("\nFukui plus")
            print("\n".join("{:6d} {:2s}    :{:12.6f}".format(
                i, str(elements[i]), v) for i, v in enumerate(cDFT_container.local_v.fukui_plus)))
            print("\nFukui minus")
            print("\n".join("{:6d} {:2s}    :{:12.6f}".format(
                i, str(elements[i]), v) for i, v in enumerate(cDFT_container.local_v.fukui_minus)))
            print("\nFukui radical")
            print("\n".join("{:6d} {:2s}    :{:12.6f}".format(
                i, str(elements[i]), v) for i, v in enumerate(cDFT_container.local_v.fukui_radical)))

        return self.postprocess_calculation_context()

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "readuct", "utils"]
