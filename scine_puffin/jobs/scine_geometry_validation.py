# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import numpy as np
import sys

from scine_puffin.config import Configuration
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from .templates.scine_react_job import HessianJob, OptimizationJob, ConnectivityJob

# TODO: Guess this should inherit from a template


class ScineGeometryValidation(HessianJob, OptimizationJob, ConnectivityJob):

    def __init__(self):
        super().__init__()
        self.name = "Scine Geometry Validation Job"
        self.validation_key = "val"
        self.opt_key = "opt"
        self.job_key = self.validation_key

        val_defaults = {
            "imaginary_wavenumber_threshold": 0.0,
            "fix_distortion_step_size": -1.0,
            "distortion_inversion_point": 2.0,
            "optimization_attempts": 0,
        }
        opt_defaults = {
            "stop_on_error": False,
            "convergence_max_iterations": 50,
            "geoopt_coordinate_system": "cartesianWithoutRotTrans"
        }

        self.settings = {
            "val": val_defaults,
            "opt": opt_defaults,
        }
        self.start_graph = ""
        self.start_key = ""
        self.end_graph = ""
        self.end_key = ""
        self.systems = {}
        self.inputs = []
        self.optimization_attempts_count = 0

    @job_configuration_wrapper
    def run(self, _, calculation, config: Configuration) -> bool:

        import scine_database as db
        import scine_readuct as readuct
        import scine_molassembler as masm

        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            # preprocessing of structure
            structure = db.Structure(calculation.get_structures()[0], self._structures)
            settings_manager, program_helper = self.create_helpers(structure)

            self.systems, _ = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )

            # default keys are ['system']
            self.inputs = [key for key in self.systems.keys()]
            self.start_key = self.inputs[0]
            # Safety check
            if not structure.has_graph("masm_cbor_graph"):
                self.raise_named_exception("Given structure has no graph.")
            self.start_graph = structure.get_graph("masm_cbor_graph")

            if program_helper is not None:
                program_helper.calculation_preprocessing(self.systems[self.start_key], calculation.get_settings())

            # # # # Extract Job settings
            self.sort_settings(settings_manager.task_settings)
            print("Validation Settings:")
            print(self.settings[self.validation_key], "\n")
            # Boolean's for logic in while loop
            opt_success = True
            clear_to_write = False
            self.end_key = self.start_key

            # Enter Run Loop for number of allowed attempts
            while self.optimization_attempts_count <= self.settings[self.validation_key]['optimization_attempts']:
                """ HESSIAN JOB"""
                self.systems, success = readuct.run_hessian_task(
                    self.systems, self.inputs)
                self.throw_if_not_successful(
                    success,
                    self.systems,
                    self.inputs,
                    ["energy", "hessian", "thermochemistry"],
                    "Hessian calculation failed.\n",
                )
                # Process hessian calculation
                _ = self.calculation_postprocessing(success, self.systems, self.inputs, [
                                                    "energy", "hessian", "thermochemistry"])

                # Frequency check
                hessian_results = self.systems[self.end_key].get_results()
                false_minimum, mode_container = self.has_wavenumber_below_threshold(
                    hessian_results,
                    self.systems[self.end_key].structure,
                    self.settings[self.validation_key]["imaginary_wavenumber_threshold"]
                )

                if not false_minimum and opt_success:
                    """ SP JOB """
                    # Copy calculator and delete its previous results
                    end_sp_key = self.end_key + "_sp"
                    self.systems[end_sp_key] = self.systems[self.end_key].clone()
                    self.systems[end_sp_key].delete_results()

                    # Check graph
                    self.end_graph, self.systems = self.make_graph_from_calc(self.systems, end_sp_key)

                    print("Start Graph:")
                    print(self.start_graph)
                    print("End Graph:")
                    print(self.end_graph)
                    # Compare start and end graph
                    if not masm.JsonSerialization.equal_molecules(self.start_graph, self.end_graph):
                        self._calculation.set_comment(self.name + ": End structure does not match starting structure.")
                        clear_to_write = False
                    else:
                        clear_to_write = True
                    # # # Leave while loop
                    break

                # Still counts left to optimize and false minium
                elif (false_minimum or not opt_success) and self.optimization_attempts_count < \
                        self.settings[self.validation_key]["optimization_attempts"]:
                    """ DISTORT AND OPT JOB """
                    self.optimization_attempts_count += 1
                    print("Optimization Attempt: " + str(self.optimization_attempts_count))
                    # # # Distort only, if it is still a false minimum
                    if false_minimum:
                        # # # Distort, write into calculator
                        self._distort_structure_and_load_calculator(mode_container, settings_manager)

                    print("Optimization Settings:")
                    print(self.settings[self.opt_key], "\n")
                    # Prepare optimization
                    end_opt_key = "distorted_opt_" + str(self.optimization_attempts_count)
                    self.settings[self.opt_key]["output"] = [end_opt_key]
                    # # # Optimization, per default stop on error is false
                    self.systems, opt_success = readuct.run_opt_task(
                        self.systems, self.inputs, **self.settings[self.opt_key])
                    # Update inputs and end key for next round
                    self.inputs = self.settings[self.opt_key]["output"]
                    self.end_key = self.inputs[0]

                    # One could adjust the convergence criteria of the optimization here

                else:
                    sys.stderr.write("Warning: Unable to do anything with this structure.")
                    break

            # Verify before writing
            self.verify_connection()

            if clear_to_write:
                final_sp_results = self.systems[end_sp_key].get_results()
                # # # Store Energy and Bond Orders overwrites existing results of identical model
                self.store_energy(self.systems[end_sp_key], structure)
                self.store_property(self._properties,
                                    "bond_orders", "SparseMatrixProperty",
                                    final_sp_results.bond_orders.matrix,
                                    self._calculation.get_model(), self._calculation, structure)
                # Store hessian information
                self.store_hessian_data(self.systems[self.end_key], structure)

                # Only overwrite positions, if an optimization was attempted
                if self.optimization_attempts_count != 0:
                    # Overwrite positions
                    org_atoms = structure.get_atoms()
                    position_shift = self.systems[self.end_key].structure.positions - org_atoms.positions
                    # # # Store Position Shift
                    self.store_property(self._properties, "position_shift", "DenseMatrixProperty",
                                        position_shift, self._calculation.get_model(), self._calculation, structure)
                    structure.set_atoms(self.systems[self.end_key].structure)
                    # # # Overwrite graph if structure has changed, decision list and idx map might have changed
                    self.add_graph(structure, final_sp_results.bond_orders)
            else:
                self.store_hessian_data(self.systems[self.start_key], structure)
                self.capture_raw_output()
                self.raise_named_exception(
                    "Structure could not be validated to be a minimum. Hessian information is stored anyway."
                )

        return self.postprocess_calculation_context()

    @staticmethod
    # TODO: add proper typing
    def has_wavenumber_below_threshold(calc_results, atoms, wavenumber_threshold: float):
        import scine_utilities as utils
        true_minimum = False
        # Get normal modes and frequencies
        modes_container = utils.normal_modes.calculate(calc_results.hessian, atoms.elements, atoms.positions)
        # Wavenumbers in cm-1
        wavenumbers = modes_container.get_wave_numbers()
        # Get minimal frequency
        min_wavenumber = np.min(wavenumbers)
        if min_wavenumber < 0.0 and abs(min_wavenumber) > wavenumber_threshold:
            true_minimum = True

        return true_minimum, modes_container

    def _distort_structure_and_load_calculator(self, mode_container, settings_manager):
        import scine_utilities as utils
        wavenumbers = np.asarray(mode_container.get_wave_numbers())
        img_wavenumber_indices = np.where(wavenumbers < 0.0)[0]
        modes = [utils.normal_modes.mode(wavenumbers[i], mode_container.get_mode(i))
                 for i in img_wavenumber_indices]

        # Distortion according to inversion point
        if self.settings[self.job_key]['fix_distortion_step_size'] == -1.0:
            max_steps = [utils.normal_modes.get_harmonic_inversion_point(
                wavenumbers[i], self.settings[self.job_key]['distortion_inversion_point'])
                for i in img_wavenumber_indices]
        else:
            max_steps = [self.settings[self.job_key]['fix_distortion_step_size'] * len(modes)]

        # Only one direction, could be improved by distorting in other direction
        # # # Displace along modes with img wavenumbers and load calculator
        distorted_positions = utils.geometry.displace_along_modes(
            self.systems[self.end_key].structure.positions,
            modes, max_steps)
        distorted_key = "distorted_guess_" + str(self.optimization_attempts_count)
        xyz_name = distorted_key + ".xyz"
        # Write file and load into calculator
        distorted_atoms = utils.AtomCollection(
            self.systems[self.end_key].structure.elements, distorted_positions)
        utils.io.write(xyz_name, distorted_atoms)
        distorted_calculator = utils.core.load_system_into_calculator(
            xyz_name,
            self._calculation.get_model().method_family,
            **settings_manager.calculator_settings,
        )
        # Load into systems and update inputs for next step
        self.systems[distorted_key] = distorted_calculator
        self.inputs = [distorted_key]

    def sort_settings(self, task_settings: dict):
        """
        Take settings of configured calculation and save them in class member. Throw exception for unknown settings.

        Notes
        -----
        * Requires run configuration
        * May throw exception

        Parameters
        ----------
        task_settings :: dict
            A dictionary from which the settings are taken
        """
        self.extract_connectivity_settings_from_dict(task_settings)
        # Dissect settings into individual user task_settings
        for key, value in task_settings.items():
            for task in self.settings.keys():
                if task == self.job_key:
                    if key in self.settings[task].keys():
                        self.settings[task][key] = value
                        break  # found right task, leave inner loop
                else:
                    indicator_length = len(task) + 1  # underscore to avoid ambiguities
                    if key[:indicator_length] == task + "_":
                        self.settings[task][key[indicator_length:]] = value
                        break  # found right task, leave inner loop
            else:
                self.raise_named_exception(
                    "The key '{}' was not recognized.".format(key)
                )
