# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""
from copy import deepcopy

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_job import ScineJob


class ScineConceptualDft(ScineJob):
    """
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

      spin_multiplicity_plus :: int
          The spin multiplicity of the system with one additional electron.
          If not specified, the additional electron is assumed to pair up with
          any priorly existing unpaired electrons.
          I.e. the multiplicity is assumed to decrease by one for all open-shell
          structures and to be two if the start structure is closed-shell.
      spin_multiplicity_minus :: int
          The spin multiplicity of the system with one electron less.
          If not specified, the deducted electron is assumed to be an unpaired
          one with the others not rearranging. I.e. the multiplicity is assumed
          to decrease by one for all open-shell structures and to be two if the
          start structure is closed-shell.

      All settings that are recognized by the program chosen.
        Furthermore, all settings that are commonly understood by any program
        interface via the SCINE Calculator interface.

      Common examples are:

      max_scf_iterations :: int
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

    def __init__(self):
        super().__init__()
        self.name = "Scine Conceptual DFT Calculation Job"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_database as db
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
                multiplicity_plus = calculation_settings["spin_multiplicity_plus"]
                del calculation_settings["spin_multiplicity_plus"]
            else:
                multiplicity_plus = 2 if multiplicity == 1 else multiplicity - 1

            if "spin_multiplicity_minus" in calculation_settings:
                multiplicity_minus = calculation_settings["spin_multiplicity_minus"]
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
                program_helper.calculation_preprocessing(systems[keys[0]], calculation_settings)

            # N electron structure/structure of interest
            print("N ELECTRON CALCULATION")
            systems, success = readuct.run_sp_task(systems, keys, **settings_manager.task_settings)
            # Checks connection, success and whether expected results exist
            self.throw_if_not_successful(
                success, systems, keys, [
                    "energy", "atomic_charges"], "Single point calculation on N electron system failed.")
            energy = systems[keys[0]].get_results().energy
            atomic_charges = systems[keys[0]].get_results().atomic_charges

            # N+1 electron "plus" system
            print("N+1 ELECTRON CALCULATION")
            print("Charge: {:4d} Multiplicity: {:4d}".format(charge_plus, multiplicity_plus))
            plus_calculator_settings = deepcopy(settings_manager.calculator_settings)
            plus_calculator_settings[utils.settings_names.molecular_charge] = charge_plus
            plus_calculator_settings[utils.settings_names.spin_multiplicity] = multiplicity_plus
            systems["plus"] = utils.core.load_system_into_calculator(
                "system.xyz", calculation.get_model().method_family, **plus_calculator_settings)
            systems, success = readuct.run_sp_task(systems, ["plus"], **settings_manager.task_settings)
            self.throw_if_not_successful(
                success, systems, ["plus"], [
                    "energy", "atomic_charges"], "Single point calculation on N+1 electron system failed.")
            energy_plus = systems["plus"].get_results().energy
            atomic_charges_plus = systems["plus"].get_results().atomic_charges

            # N-1 electron "minus" system
            print("N-1 ELECTRON CALCULATION")
            print("Charge: {:4d} Multiplicity: {:4d}".format(charge_minus, multiplicity_minus))
            minus_calculator_settings = deepcopy(settings_manager.calculator_settings)
            minus_calculator_settings[utils.settings_names.molecular_charge] = charge_minus
            minus_calculator_settings[utils.settings_names.spin_multiplicity] = multiplicity_minus
            systems["minus"] = utils.core.load_system_into_calculator(
                "system.xyz", calculation.get_model().method_family, **minus_calculator_settings)
            systems, success = readuct.run_sp_task(systems, ["minus"], **settings_manager.task_settings)
            self.throw_if_not_successful(
                success, systems, ["minus"], [
                    "energy", "atomic_charges"], "Single point calculation on N-1 electron system failed.")
            energy_minus = systems["minus"].get_results().energy
            atomic_charges_minus = systems["minus"].get_results().atomic_charges

            # Deduce conceptual DFT properties
            print("cDFT PROPERTIES")
            cDFT_container = utils.conceptual_dft.calculate(
                energy, atomic_charges, energy_plus, atomic_charges_plus, energy_minus, atomic_charges_minus)

            # Calculation postprocessing
            self.verify_connection()
            # clear existing results
            db_results = self._calculation.get_results()
            db_results.clear()
            self._calculation.set_results(db_results)
            # update model
            scine_helper.update_model(systems[keys[0]], self._calculation, self.config)

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
    def required_programs():
        return ["database", "readuct", "utils"]
