# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List
import os
import subprocess
from scine_puffin.config import Configuration
from .templates.job import Job, calculation_context, job_configuration_wrapper
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class GaussianChargeModel5(Job):
    """
    A job calculating CM5 (Charge Model 5) atomic partial charges based
    on the Hirshfeld population analysis for a given structure
    with the Gaussian program.

    **Order Name**
      ``gaussian_charge_model_5``

    **Optional Settings**
      There are no optional settings.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - The Gaussian program has to be available.

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The ``cm5_charges`` calculated for the given structure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_file = "gaussian_calc.inp"
        self.output_file = "gaussian_calc.out"

    def write_input_file(self, structure, model, job, id: str):
        import scine_utilities as utils

        with open(self.input_file, "w") as input:
            # Number of cores
            input.write("%NProcShared={}\n".format(job.cores))
            # Memory requirements
            input.write("%Mem={}MB\n".format(int(job.memory * 1024)))
            # Method
            input.write("# {}/{} Pop=Hirshfeld\n\n".format(model.method, model.basis_set))
            # Title with puffin id
            input.write("Gaussian calculation created by {}\n\n".format(id))
            # Charge and multiplicity
            input.write("{} {}\n".format(structure.get_charge(), structure.get_multiplicity()))
            # Write atoms
            for atom in structure.get_atoms():
                pos = atom.position * utils.ANGSTROM_PER_BOHR
                input.write("{}    {:.8f}    {:.8f}    {:.8f}\n".format(atom.element, pos[0], pos[1], pos[2]))
            # One last empty line
            input.write("\n")

    def parse_output_file(self, natoms: int) -> list:
        cm5_charges = []
        keyword = "CM5 charges"
        with open(self.output_file, "r") as output:
            lines = output.readlines()
            # Find line index to begin
            for index, line in enumerate(lines):
                if keyword in line:
                    line_begin = index + 2
                    break
            try:
                # The following line will raise an exception if line_begin was never defined above
                for i in range(line_begin, line_begin + natoms):
                    line = lines[i]
                    cm5_charges.append(float(line.split()[7]))
            except UnboundLocalError as e:
                raise RuntimeError("CM5 charges were not present in the Gaussian output file.") from e

        return cm5_charges

    def execute_program(self):
        env = os.environ.copy()
        exe = env["GAUSSIAN_BINARY_PATH"]
        env.update({"GAUSS_SCRDIR": self.work_dir})
        subprocess.run(
            "{} < {} > {}".format(exe, self.input_file, self.output_file),
            env=env,
            shell=True,
        )

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

        # Do calculation
        with calculation_context(self):
            # Write input file
            self.write_input_file(structure, model, job, config["daemon"]["uuid"])
            # Execute Gaussian
            self.execute_program()
            # Parse output file
            cm5_charges = self.parse_output_file(natoms)

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
                calculation.set_comment("Gaussian Charge Model job failed with an unspecified error.")
            calculation.set_status(db.Status.FAILED)
            return False

        # A sanity check
        if len(cm5_charges) != natoms:
            calculation.set_comment(
                "Calculation failed because the number of parsed CM5 charges does not equal the number of atoms."
            )
            calculation.set_status(db.Status.FAILED)
            return False

        # Update model
        model.program = "gaussian"
        model.version = config.programs()[model.program]["version"]
        calculation.set_model(model)

        # Store CM5 Charges
        self.store_property(
            properties,
            "cm5_charges",
            "VectorProperty",
            cm5_charges,
            model,
            calculation,
            structure,
        )

        calculation.set_executor(config["daemon"]["uuid"])
        calculation.set_status(db.Status.COMPLETE)
        return True

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "utils", "gaussian"]
