# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List
import os
from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.turbomole_job import TurbomoleJob
from ..utilities.turbomole_helper import TurbomoleHelper
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class TurbomoleSinglePoint(TurbomoleJob):

    """
    A job calculating the electronic energy for a given structure with the
    Turbomole program.

    **Order Name**
      ``turbomole_single_point``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      self_consistence_criterion : float
          The self consistence criterion corresponding to the maximum
          energy change between two SCF cycles resulting in convergence.
          Default value ist 1E-6.
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
           successfully. The default is True.
      scf_damping : bool
          Increases damping during the SCF by modifying the $scfdamp parameter in the control file.
          The default is False.
      scf_orbitalshift : float
          Shifts virtual orbital energies upwards. Default value is 0.1.
      calculate_loewdin_charges : bool
          Calculates the Loewdin partial charges. The default is False.
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

    Properties
      The ``electronic_energy`` associated with the structure.
      ``atomic_charges`` for all atoms, if requested.

    """

    def __init__(self) -> None:
        super().__init__()
        self.input_structure = "system.xyz"
        self.tm_helper = TurbomoleHelper()

    # Executes a single point calculation using the dscf script
    def execute_single_point_calculation(self, job: db.Job) -> None:
        if job.cores > 1:
            os.environ["PARA_ARCH"] = "SMP"
            os.environ["PARNODES"] = str(job.cores)
            self.tm_helper.execute(os.path.join(self.smp_turboexe, "ridft"))
        else:
            self.tm_helper.execute("{}".format(os.path.join(self.turboexe, "ridft")))

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

        if not self.turboexe:
            calculation.set_status(db.Status.FAILED)
            raise RuntimeError("Turbomole executables are not available.")

        calculation_settings = calculation.get_settings()

        # Do calculations
        with calculation_context(self):
            # Prepare calculation
            self.prepare_calculation(structure, calculation_settings, model, job)
            # Execute Program
            self.execute_single_point_calculation(job)
            # Parse output file
            parsed_energy = self.tm_helper.parse_energy_file()
            # Get loewdin charges if requested
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

        # A sanity check
        if not os.path.exists(os.path.join(self.work_dir, "success")):
            if len(error.strip()) > 0:
                calculation.set_comment(error.replace("\n", " "))
            else:
                calculation.set_comment(
                    "Single Point Job Error: Turbomole Single Point job failed with an unspecified error."
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
        calculation.set_results(db_results)

        # Store energy
        self.store_property(
            properties,
            "electronic_energy",
            "NumberProperty",
            parsed_energy,
            model,
            calculation,
            structure,
        )
        # Store atomic charges if requested
        if atomic_charges_set:
            self.store_property(
                properties,
                "loewdin_charges",
                "VectorProperty",
                atomic_charges,
                model,
                calculation,
                structure,
            )
        calculation.set_status(db.Status.COMPLETE)
        return True

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "utils", "turbomole"]
