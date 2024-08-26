# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List
import os

import numpy as np

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.turbomole_job import TurbomoleJob
from .turbomole_single_point import TurbomoleSinglePoint
from ..utilities.turbomole_helper import TurbomoleHelper
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class TurbomoleHessian(TurbomoleJob):

    """
    A job generating a Hessian and derived data for a single structure.

    **Order Name**
      ``turbomole_hessian``

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
      spin_mode : str
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
        The ``hessian`` and the ``electronic_energy`` associated with the structure.
        ``atomic_charges`` for all atoms, if requested
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_structure = "system.xyz"
        self.hessian_file = "hessian"
        self.normal_modes_file = "vibspectrum"
        self.aoforce_file = "aoforce.out"
        self.tm_helper = TurbomoleHelper()
        self.tm_single_point = TurbomoleSinglePoint()

    # Run hessian calculation
    def execute_hessian_calculation(self, job: db.Job) -> None:
        if job.cores > 1:
            os.environ["PARA_ARCH"] = "SMP"
            os.environ["PARNODES"] = str(job.cores)
            self.tm_helper.execute(os.path.join(self.smp_turboexe, "aoforce"))
        else:
            self.tm_helper.execute(os.path.join(self.turboexe, "aoforce"))

    # Read hessian matrix from file "hessian"
    def read_hessian_matrix(self, natoms: int) -> np.ndarray:

        hessian_list = []
        with open(self.hessian_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
            hess_lines = [line.split() for line in lines[1:-1]]
            for line in hess_lines:
                hess_row = []
                for item in line:
                    try:
                        int(item)
                    except ValueError:
                        hess_row.append(float(item))
                hessian_list.append(hess_row)
            flat_hessian_list = [item for sublist in hessian_list for item in sublist]
            hessian = np.array([flat_hessian_list[i: i + 3 * natoms]
                                for i in range(0, len(flat_hessian_list), 3 * natoms)])
            # Check if Hessian has correct dimension and is symmetric
            if (len(hessian) == 3 * natoms) and (np.allclose(np.transpose(hessian), hessian)):
                return hessian
            else:
                raise RuntimeError("Something went wrong while parsing the Hessian matrix.")

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        # Gather all required collections
        structures = manager.get_collection("structures")
        calculations = manager.get_collection("calculations")
        properties = manager.get_collection("properties")

        # Link calculation if needed
        if not calculation.has_link():
            calculation.link(calculations)

        # Get structure
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
        # Do calculation
        with calculation_context(self):
            # Prepare calculation
            self.prepare_calculation(structure, calculation_settings, model, job)
            # Execute single_point necessary for subsequent hessian calculation
            self.tm_single_point.execute_single_point_calculation(job)
            # Parse output file
            parsed_energy = self.tm_helper.parse_energy_file()
            # Execute hessian calculation
            self.execute_hessian_calculation(job)
            # Read hessian
            hessian = self.read_hessian_matrix(natoms)
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

        if not os.path.exists(os.path.join(self.work_dir, "success")):
            if len(error.strip()) > 0:
                calculation.set_comment(error.replace("\n", " "))
            else:
                calculation.set_comment("Hessian Job Error: Turbomole Hessian job failed with an unspecified error.")
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
        # Store hessian
        self.store_property(
            properties,
            "hessian",
            "DenseMatrixProperty",
            hessian,
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
