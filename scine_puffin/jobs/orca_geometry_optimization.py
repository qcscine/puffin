# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import re
import subprocess
import glob
from typing import Any, List, Tuple, TYPE_CHECKING
from scine_puffin.config import Configuration
from .templates.job import Job, calculation_context, job_configuration_wrapper
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class OrcaGeometryOptimization(Job):
    """
    A job optimizing the geometry of a given structure with the Orca program,
    in search of a local minimum on the potential energy surface.
    Optimizing a given structure's geometry, generating a new minimum energy
    structure, if successful.

    **Order Name**
      ``orca_geometry_optimization``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      convergence_max_iterations : int
         The maximum number of geometry optimization cycles.
      cartesian_constraints : List[int]
         A list of atom indices of the atoms which positions will be
         constrained during the optimization.
      max_scf_iterations : int
         The number of allowed SCF cycles until convergence.
      self_consistence_criterion : float
         The self consistence criterion corresponding to the maximum
         energy change between two SCF cycles resulting in convergence.
      scf_damping : bool
         Switches damping on or off during the SCF by employing the Orca
         keyword SlowConv. The default is False.
      calculate_hirshfeld_charges : bool
         Calculates atomic partial charges based on the Hirshfeld population
         analysis for the optimized structure and stores these into the
         database. The default is False.
      transform_coordinates : bool
         Switch to transform the input coordinates from Cartesian to redundant
         internal coordinates. Setting this value to False and hence performing
         the calculation in Cartesian coordinates is helpful in rare occasions
         where the calculation with redundant internal coordinates fails.
         The optimization will take more time but is more likely to end
         successfully. The default is True.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - The Orca program has to be available

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Structures
        A new minimum energy structure.
      Properties
        The ``electronic_energy`` associated with the new structure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_file = "orca_calc.inp"
        self.output_file = "orca_calc.out"
        self.input_structure = "system.xyz"
        self.output_structure = "orca_calc.xyz"

    def write_input_file(self, structure, model, job, id: str, settings: dict):
        import scine_utilities.settings_names as sn

        with open(self.input_file, "w") as input:
            # Method
            input.write("! {} {} Opt\n".format(model.method, model.basis_set))
            # Add damping for SCF if requested
            if sn.scf_damping in settings:
                if settings[sn.scf_damping]:
                    input.write("! SlowConv\n")
            # Perform optimization in Cartesian coordinates
            if "geoopt_coordinate_system" in settings:
                if settings["geoopt_coordinate_system"].lower() == "cartesian":
                    input.write("! COPT\n")
            # Title with puffin id
            input.write("# Orca calculation created by {}\n\n".format(id))
            # Memory for one core
            input.write("%maxcore {}\n".format(int(job.memory * 1024 / job.cores)))
            # Number of cores
            if job.cores > 1:
                input.write("%pal\nnprocs {}\nend\n".format(job.cores))
            # Hirshfeld charges
            if "calculate_hirshfeld_charges" in settings:
                if settings["calculate_hirshfeld_charges"]:
                    input.write("%output\nPrint[P_Hirshfeld] 1\nend\n")

            # Settings for geom
            max_iterations_set = "convergence_max_iterations" in settings
            constraints_set = "cartesian_constraints" in settings
            if max_iterations_set or constraints_set:
                input.write("%geom\n")
                if max_iterations_set:
                    input.write("MaxIter {}\n".format(int(settings["convergence_max_iterations"])))
                if constraints_set:
                    # The Cartesian constraints settings is given as a string containing the
                    # indices of the atoms to be constrained separated by whitespaces
                    constraint_atoms = settings["cartesian_constraints"]
                    if len(constraint_atoms) > 0:
                        input.write("Constraints\n")
                        for a in constraint_atoms:
                            input.write("{{C {} C}}\n".format(a))
                        input.write("end\n")
                input.write("end\n\n")

            # Settings for SCF
            self_consistence_criterion_set = sn.self_consistence_criterion in settings
            max_scf_iterations_set = sn.max_scf_iterations in settings
            if self_consistence_criterion_set or max_scf_iterations_set:
                input.write("%SCF\n")
                if self_consistence_criterion_set:
                    input.write("TolE {}\n".format(settings[sn.self_consistence_criterion]))
                if max_scf_iterations_set:
                    input.write("MaxIter {}\n".format(int(settings[sn.max_scf_iterations])))
                input.write("end\n\n")

            # Structure file, charge and multiplicity
            input.write(
                "*xyzfile {} {} {}\n".format(
                    structure.get_charge(),
                    structure.get_multiplicity(),
                    self.input_structure,
                )
            )

    def parse_output_file(self) -> Tuple[Any, float, list]:
        import scine_utilities as utils

        successful = False
        atomic_charges: List[float] = []
        optimized_energy = None
        with open(self.output_file, "r") as out:
            lines = out.readlines()
            for line_index, line in enumerate(lines):
                # Check for convergence
                if ("OPTIMIZATION" in line) and ("CONVERGED" in line):
                    successful = True
                # Parse final energy
                elif "FINAL SINGLE POINT ENERGY" in line:
                    optimized_energy = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", line)[0]
                # Parse Hirshfeld charges
                elif "HIRSHFELD ANALYSIS" in line:
                    # Clear atomic charges list
                    atomic_charges = []
                    current_line_index = line_index
                    # Go forward until charges start
                    while "CHARGE" not in lines[current_line_index]:
                        current_line_index += 1
                    current_line_index += 1
                    # Add the charges to the list until empty line
                    while lines[current_line_index].strip():
                        charge = lines[current_line_index].split()[2]
                        atomic_charges.append(float(charge))
                        current_line_index += 1
        if successful:
            if optimized_energy is None:
                raise RuntimeError("Optimization converged but no energy found in output file.")
            optimized_structure, _ = utils.io.read(self.output_structure)
            return optimized_structure, float(optimized_energy), atomic_charges
        else:
            raise RuntimeError

    def execute_program(self):
        env = os.environ.copy()
        exe = env["ORCA_BINARY_PATH"]
        with open(self.output_file, 'w') as f:
            subprocess.run(
                #                ["{}".format(exe), "{}".format(self.input_file)],
                "{}".format(exe) + " " + "{}".format(self.input_file),
                env=env,
                stdout=f,
                shell=True
            )

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_utilities as utils

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
        # TODO: These labels are not necessarily correct; during the optimization, a
        # complex coul be created
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

        # Do calculation
        with calculation_context(self):
            # Write xyz file
            utils.io.write(self.input_structure, structure.get_atoms())
            # Write input file
            self.write_input_file(
                structure,
                model,
                job,
                config["daemon"]["uuid"],
                dict(calculation.get_settings()),
            )
            # Execute ORCA
            self.execute_program()
            # Parse output file
            (
                optimized_structure,
                parsed_energy,
                atomic_charges,
            ) = self.parse_output_file()

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
                "Geometry Optimization Job Error: Orca Geometry Optimization job failed with an unspecified error."
            )

        if not os.path.exists(os.path.join(self.work_dir, "success")):
            calculation.set_comment(error_msg)
            calculation.set_status(db.Status.FAILED)
            # Clean up .tmp files
            tmp_files = glob.glob(os.path.join(self.work_dir, "*.tmp"))
            for f in tmp_files:  # type: ignore
                os.remove(str(f))
            return False

        # A sanity check for the optimized structure
        if len(optimized_structure) != natoms:
            calculation.set_comment(
                "Geometry Optimization Job Error: Optimized structure has incorrect number of atoms"
            )
            calculation.set_status(db.Status.FAILED)
            return False
        # A sanity check for the atomic charges
        if "calculate_hirshfeld_charges" in calculation.get_settings():
            if calculation.get_settings()["calculate_hirshfeld_charges"]:
                if len(atomic_charges) != natoms:
                    calculation.set_comment("Geometry Optimization Job Error: Incorrect number of atomic charges.")
                    calculation.set_status(db.Status.FAILED)
                    return False

        # Update model
        model.program = "orca"
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
        if len(atomic_charges) > 0:
            self.store_property(
                properties,
                "hirshfeld_charges",
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
        return ["database", "utils", "orca"]
