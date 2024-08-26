# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
from typing import Any, Tuple, TYPE_CHECKING, List
from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.turbomole_job import TurbomoleJob
from ..utilities.turbomole_helper import TurbomoleHelper
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_database")


class TurbomoleGeometryOptimization(TurbomoleJob):

    """
    A job optimizing the geometry of a given structure with the Turbomole program,
    in search of a local minimum on the potential energy surface.
    Optimizing a given structure's geometry, generating a new minimum energy
    structure, if successful.

    **Order Name**
      ``turbomole_geometry_optimization``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      cartesian_constraints : List[int]
          A list of atom indices of the atoms which positions will be
          constrained during the optimization.
      max_scf_iterations : int
          The number of allowed SCF cycles until convergence. Default value is 30.
      transform_coordinates : bool
           Switch to transform the input coordinates from redundant internal
           to cartesian coordinates. Setting this value to True and hence performing
           the calculation in Cartesian coordinates is helpful in rare occasions
           where the calculation with redundant internal coordinates fails.
           The optimization will take more time but is more likely to end
           successfully. The default is False.
      convergence_max_iterations : int
          The maximum number of geometry optimization cycles.
      scf_damping : bool
          Increases damping during the SCF by modifying the $scfdamp parameter in the control file.
          The default is False.
      scf_orbitalshift : float
          Shifts virtual orbital energies upwards. Default value is 0.1.
      convergence_delta_value : int
          The convergence criterion for the electronic energy difference between
          two steps. The default value is 1E-6.
      calculate_loewdin_charges : bool
          Calculates the Loewdin partial charges. The default is False.
      self_consistence_criterion : float
          The self consistence criterion corresponding to the maximum
          energy change between two SCF cycles resulting in convergence.
          Default value ist 1E-6.
      spin_mode : string
          Sets the spin mode. If no spin mode is set, Turbomole's default for the corresponding
          system is chosen. Options are: restricted, unrestricted.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - The Turbomole program has to be available

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Structures
        A new minimum energy structure.
      Properties
        The ``electronic_energy`` associated with the new structure and ``atomic_charges`` for all atoms, if requested.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_structure = "system.xyz"
        self.converged_file = "GEO_OPT_CONVERGED"
        self.output_structure = "optimized_structure.xyz"
        self.trajectory = "path.xyz"

        # Create instance of TurbomoleHelper
        self.tm_helper = TurbomoleHelper()

    # Executes structure optimization using the jobex script
    def execute_optimization(self, settings: utils.ValueCollection, job: db.Job) -> None:

        os.environ["PARA_ARCH"] = "SMP"
        os.environ["PARNODES"] = str(job.cores)
        # Max. number of cycles
        max_iterations_set = "convergence_max_iterations" in settings
        # Total energy convergence criterion
        convergence_delta_value_set = "convergence_delta_value" in settings
        jobex_flags = ""
        if max_iterations_set:
            jobex_flags += " -c " + str(settings["convergence_max_iterations"])
        if convergence_delta_value_set:
            jobex_flags += " -energy " + str(settings["convergence_delta_value"])
        if job.cores > 1:
            jobex_flags += " -np {}".format(job.cores)

        args = os.path.join(self.turboscripts, "jobex") + jobex_flags
        self.tm_helper.execute(args, error_test=True, stdout_tofile=True)

        if not os.path.exists(self.converged_file):
            raise RuntimeError("Structure optimization failed.")

    @requires("utilities")
    def parse_results(self) -> Tuple[Any, float]:
        """
        Parse energy and extract output structure from coord file
        """

        successful = False

        # Parse energy from file "energy"
        parsed_energy = self.tm_helper.parse_energy_file()

        if parsed_energy is not None:
            successful = True

        if successful:
            args = os.path.join(self.turboscripts, "t2x") + " -c"
            self.tm_helper.execute(args, error_test=False, stdout_tofile=True)
            os.rename("t2x.out", self.output_structure)
            self.tm_helper.execute(
                os.path.join(self.turboscripts, "t2x"),
                error_test=False,
                stdout_tofile=True,
            )
            os.rename("t2x.out", self.trajectory)
            optimized_structure, _ = utils.io.read(self.output_structure)
            return optimized_structure, parsed_energy
        else:
            raise RuntimeError

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        # Gather all required collections
        structures = manager.get_collection("structures")
        calculations = manager.get_collection("calculations")
        properties = manager.get_collection("properties")

        # Link calculation if needed
        if not calculation.has_link():
            calculation.link(calculations)

        # Get structure and number of atoms
        structure = db.Structure(calculation.get_structures()[0])
        structure.link(structures)
        natoms = len(structure.get_atoms())

        # Get model
        model = calculation.get_model()

        # Get job
        job = calculation.get_job()

        # New label
        # TODO: These labels are not necessarily correct; during the optimization, a complex could be created
        label = structure.get_label()
        if label == db.Label.MINIMUM_GUESS or label == db.Label.MINIMUM_OPTIMIZED:
            new_label = db.Label.MINIMUM_OPTIMIZED
        elif label == db.Label.USER_GUESS or label == db.Label.USER_OPTIMIZED:
            new_label = db.Label.USER_OPTIMIZED
        elif label == db.Label.SURFACE_GUESS or label == db.Label.SURFACE_OPTIMIZED:
            new_label = db.Label.SURFACE_OPTIMIZED
        else:
            with open(os.path.join(self.work_dir, "errors"), "a") as f:
                error = "Unknown label of input structure: '" + str(label) + "'\n"
                f.write(error)
                calculation.set_comment(error)
                calculation.set_status(db.Status.FAILED)
                return False

        if not self.turboexe:
            calculation.set_status(db.Status.FAILED)
            raise RuntimeError("Turbomole executables are not available.")

        calculation_settings = calculation.get_settings()
        # Do calculation
        with calculation_context(self):
            # Prepare calculation
            self.prepare_calculation(structure, calculation_settings, model, job)
            # Execute Program
            self.execute_optimization(calculation.get_settings(), job)
            # Parse output file
            optimized_structure, parsed_energy = self.parse_results()
            # Get Loewdin charges if requested
            atomic_charges_set, atomic_charges = self.tm_helper.get_loewdin_charges(natoms, calculation_settings)
            spin_mode = self.tm_helper.evaluate_spin_mode(calculation_settings)

        # After the calculation, verify that the connection to the database still exists
        self.verify_connection()

        # Capture raw output
        stdout_path = os.path.join(self.work_dir, "output")
        stderr_path = os.path.join(self.work_dir, "errors")
        with open(stdout_path, "r") as stdout, open(stderr_path, "r") as stderr:
            error = stderr.read()
            calculation.set_raw_output(stdout.read() + "\n" + error)

        # Get the error message to be eventually added to the calculation comment
        if len(error.strip()) > 0:
            error_msg = error.replace("\n", " ")
        else:
            error_msg = (
                "Geometry Optimization Job Error: Turbomole Geometry Optimization job failed with an unspecified error."
            )

        if not os.path.exists(os.path.join(self.work_dir, "success")):
            calculation.set_comment(error_msg)
            calculation.set_status(db.Status.FAILED)
            return False

        # A sanity check
        if not os.path.exists(os.path.join(self.work_dir, self.converged_file)):
            calculation.set_comment(error_msg)
            calculation.set_status(db.Status.FAILED)
            return False

        # Check second property
        if len(optimized_structure) != natoms:
            calculation.set_comment(
                "Geometry Optimization Job Error: Optimized structure has incorrect number of atoms"
            )
            calculation.set_status(db.Status.FAILED)
            return False

        # Update model
        model.program = "turbomole"
        model.spin_mode = spin_mode
        model.version = config.programs()[model.program]["version"]
        calculation.set_model(model)

        # Generate database results
        db_results = calculation.get_results()
        db_results.clear()

        # New structure
        new_structure = db.Structure()
        new_structure.link(structures)
        new_structure.create(
            optimized_structure,
            structure.get_charge(),
            structure.get_multiplicity(),
            model,
            new_label,
        )
        db_results.add_structure(new_structure.id())
        calculation.set_results(db_results)

        # Store energy
        self.store_property(
            properties,
            "electronic_energy",
            "NumberProperty",
            parsed_energy,
            model,
            calculation,
            new_structure,
        )
        # Store atomic charges if available
        if atomic_charges_set:
            self.store_property(
                properties,
                "loewdin_charges",
                "VectorProperty",
                atomic_charges,
                model,
                calculation,
                new_structure,
            )

        calculation.set_status(db.Status.COMPLETE)
        return True

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "utils", "turbomole"]
