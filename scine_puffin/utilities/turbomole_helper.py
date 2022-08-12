# -*- coding: utf-8 -*-
"""turbomole_helper.py: Collection of common procedures to be carried out with turbomole"""
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os
import re
import sys
from subprocess import Popen, PIPE, run
from typing import List, Tuple

from scine_puffin.jobs.templates.job import TurbomoleJob


class TurbomoleHelper:
    def __init__(self):
        self.define_input = "tm.input"
        self.input_structure = "system.xyz"
        self.coord_file = "coord"
        self.define_output_file = "define.out"
        self.control_file = "control"

        self.energy_file = "energy"
        self.proper_file = "proper.out"
        self.proper_input_file = "proper.inp"
        self.available_d3_params = ["D3", "D3BJ"]
        self.available_spin_modes = ["any", "restricted", "unrestricted"]

        self.turbomole_job = TurbomoleJob()

    def check_settings_availability(self, job, settings: dict):
        import scine_utilities.settings_names as sn

        # All available settings that are implemented for Turbomole
        available_settings = [
            "cartesian_constraints",
            sn.max_scf_iterations,
            "transform_coordinates",
            sn.scf_damping,
            sn.self_consistence_criterion,
            "scf_orbitalshift",
            "calculate_loewdin_charges",
            sn.spin_mode,
        ]

        available_settings_structure_optimization = [
            "convergence_delta_value",
            "convergence_max_iterations",
        ]

        settings_to_check = available_settings

        if job.order == "turbomole_geometry_optimization":
            settings_to_check.extend(available_settings_structure_optimization)

        for key in settings.keys():
            if key not in settings_to_check:
                raise NotImplementedError("Error: The key '{}' was not recognized.".format(key))

    # Executes any Turbomole pre- or postprocessing tool
    def execute(self, args, input_file=None, error_test=True, stdout_tofile=True):

        if isinstance(args, str):
            args = args.split()

        # stdout
        if stdout_tofile:
            out_file = os.path.basename(args[0]) + ".out"
            out = open(out_file, "w")
        else:
            out = PIPE

        if input_file:
            with open(input_file, "r") as f:
                data = f.read()
                stdin = data.encode("utf-8")
        else:
            stdin = None

        message = 'Turbomole command "' + os.path.basename(args[0]) + '" execution failed'

        with Popen(args, stdin=PIPE, stderr=out, stdout=out) as proc:
            res = proc.communicate(input=stdin)
        if error_test:
            with open(out_file) as file:
                lines = file.readlines()
                for line in lines:
                    if "ended abnormally" in line:
                        raise RuntimeError(message)

        if not stdout_tofile:
            return res[0].decode("utf-8", errors='replace')

    # Converts xyz file to Turbomole coord file

    def write_coord_file(self, settings: dict):

        import scine_utilities as utils

        # Read in xyz file and transform coordinates from Angstrom to Bohr
        xyz, _ = utils.io.read(self.input_structure)

        coord_in_bohr = []
        for i in range(len(xyz)):
            coord_in_bohr.append(
                "{} {} {} {}".format(
                    xyz.positions[i][0],
                    xyz.positions[i][1],
                    xyz.positions[i][2],
                    xyz.elements[i],
                )
            )

        with open(self.coord_file, "w") as coord:
            coord.write("$coord\n")
            # The Cartesian constraints settings is given as a list
            # containing the indices of the atoms to be constrained
            constraints_set = "cartesian_constraints" in settings
            for j, line in enumerate(coord_in_bohr):
                if constraints_set:
                    constraint_atoms = settings["cartesian_constraints"]
                    if (j + 1) in constraint_atoms:
                        coord.write(line + " f" + "\n")
                    else:
                        coord.write(line + "\n")
                else:
                    coord.write(line + "\n")
            coord.write("$end\n")

    # Generates input file for the preprocessing tool define
    def prepare_define_session(self, structure, model, settings: dict, job):
        import scine_utilities.settings_names as sn

        with open(self.define_input, "w") as define_input:
            define_input.write("\n\na {}\n".format(self.coord_file))

            cartesian_constraints_set = "cartesian_constraints" in settings
            transform_coordinates_set = False

            if job.order == "turbomole_single_point" or "turbomole_hessian":
                transform_coordinates_set = True

            if "transform_coordinates" in settings:
                transform_coordinates_set = settings["transform_coordinates"]

            spin_mode = "any"
            spin_mode_is_set = False
            if sn.spin_mode in settings:
                spin_mode = settings[sn.spin_mode]
                spin_mode_is_set = True

            if spin_mode not in self.available_spin_modes:
                raise NotImplementedError("Invalid spin mode!")

            if spin_mode == "restricted" and (structure.get_multiplicity() != 1):
                raise RuntimeError("Restricted spin mode and chosen multiplicity do not match.")

            # Cartesian or internal coordinates
            if cartesian_constraints_set or transform_coordinates_set:
                define_input.write("*\nno\n")
            else:
                define_input.write("ired\n*\n")

            define_input.write("\nb all {} \n\n*\neht\n\n".format(model.basis_set))
            # If spin mode is not set or set to "restricted", let turbomole handle
            # spin mode automatically (default is RHF)
            if (spin_mode_is_set and spin_mode == "restricted") or (not spin_mode_is_set):
                define_input.write("{}\n\n\n\n".format(structure.get_charge()))
            # Assign unrestricted spin mode
            elif spin_mode == "unrestricted":
                number_of_unpaired_electrons = structure.get_multiplicity() - 1
                if number_of_unpaired_electrons == 0:
                    define_input.write("{}\nno\ns\n*\n\n".format(structure.get_charge()))
                else:
                    define_input.write(
                        "{}\nno\nu {}\n*\n\n".format(structure.get_charge(), number_of_unpaired_electrons)
                    )
            # Enable RI per default TODO: Make this a setting?
            define_input.write("ri\non\n\n")
            # Method
            if model.method_family.casefold() == "dft":
                define_input.write("dft\non\nfunc {}\n\n".format(model.method.split("-")[0]))

            # Dispersion Correction
            method_string = model.method.split("-")
            if len(method_string) > 1:
                vdw_type = method_string[1].upper()

                if vdw_type in self.available_d3_params and vdw_type == "D3":
                    define_input.write("dsp\non\n\n")
                elif vdw_type in self.available_d3_params and vdw_type == "D3BJ":
                    define_input.write("dsp\nbj\n\n")
                else:
                    raise NotImplementedError("Invalid dispersion correction!")

            # Max number of SCF iterations
            if sn.max_scf_iterations in settings:
                define_input.write("scf\niter\n{}\n".format(int(settings[sn.max_scf_iterations])))
            define_input.write("\n*")

    # Runs the Turbomole preprocessing tool define
    def initialize(self, model, settings: dict):

        import math
        import scine_utilities.settings_names as sn

        self.execute(
            os.path.join(self.turbomole_job.turboexe, "define"),
            input_file=self.define_input,
            error_test=True,
            stdout_tofile=True,
        )

        with open(self.define_output_file) as file:
            if "define ended normally" not in file.read():
                raise RuntimeError("Define ended abnormally!")

        # Add damping for SCF if requested
        # TODO: SCF damping setting should allow for the choice of custom damping parameters
        if sn.scf_damping in settings:
            if settings[sn.scf_damping]:
                run(
                    r"sed -i '/$scfdamp*/c\$scfdamp   start=8.500  step=0.10  min=0.50' {}".format(self.control_file),
                    shell=True,
                )

        if "scf_orbitalshift" in settings:
            run(
                r"sed -i '/$scforbitalshift */c\$scforbitalshift  closedshell={}' {}".format(
                    float(settings["scf_orbitalshift"]), self.control_file
                ),
                shell=True,
            )
        # SCF convergence criterion
        if sn.self_consistence_criterion in settings:
            convergence_threshold = int(round(-math.log10(settings[sn.self_consistence_criterion])))
            run(
                r"sed -i '/$scfconv*/c\$scfconv {}' {}".format(convergence_threshold, self.control_file),
                shell=True,
            )

        # A sanity check
        # Checks if DFT functional and basis set were assigned correctly, sdg
        # reads the respective data groups from control file
        args = [os.path.join(self.turbomole_job.turboexe, "sdg"), "dft", "atoms"]
        self.execute(args, error_test=False)
        with open("sdg.out") as f:
            if model.basis_set not in f.read():
                raise NotImplementedError(
                    "Basis set not assigned correctly. Please check spelling."
                    + "\n"
                    + "Turbomole is case-sensitive with regard to the basis set input."
                )
            f.seek(0)
            if model.method.split("-")[0] not in f.read():
                raise NotImplementedError(
                    "DFT functional is not assigned correctly. Please check spelling."
                    + "\n"
                    + "Turbomole accepts all functionals in lower case letters only."
                )

    # Parse Loewdin charges
    def get_loewdin_charges(self, natoms, calculation_settings) -> Tuple[bool, List[float]]:

        # Execute Turbomole postprocessing tool proper
        with open(self.proper_input_file, "a") as proper_file:
            proper_file.write("pop\nloewdin\nend\nend\n")
        self.execute(
            "{}".format(os.path.join(self.turbomole_job.turboexe, "proper")),
            input_file=self.proper_input_file,
        )
        loewdin_charges = []
        # Read atomic charges from proper output
        with open(self.proper_file, "r") as file:
            lines = file.readlines()
            for n, line in enumerate(lines):
                if "atom      charge" in line:
                    charge_lines = lines[n + 1: n + natoms + 1]
            for n, line in enumerate(charge_lines):
                charge_list = re.split(r"\s+", charge_lines[n])
                charge_list = list(filter(None, charge_list))
                if "f" in charge_list:
                    loewdin_charges.append(float(charge_list[2]))
                else:
                    loewdin_charges.append(float(charge_list[1]))

        atomic_charges_set = False
        if "calculate_loewdin_charges" in calculation_settings:
            if calculation_settings["calculate_loewdin_charges"]:
                atomic_charges_set = True
                # a sanity check
                if len(loewdin_charges) != natoms:
                    raise IndexError()

        return atomic_charges_set, loewdin_charges

    # Parse energy file
    def parse_energy_file(self) -> float:
        try:
            with open(self.energy_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("$"):
                        pass
                    elif line.startswith("$end"):
                        break
                    else:
                        energy = line.split()[1]
            return float(energy)
        except FileNotFoundError as e:
            raise RuntimeError("Energy file is not accessible because the job failed.") from e

    def evaluate_spin_mode(self, calculation_settings) -> str:
        import scine_utilities.settings_names as sn

        with open(self.control_file) as f:
            if "uhf" and "uhfmo" in f.read():
                if calculation_settings.get(sn.spin_mode) == "restricted":
                    sys.stderr.write(
                        "Requested restricted calculation was converted to an unrestricted calculation "
                        "with multiplicity != 1. Please enforce unrestricted singlet for this case."
                    )
                return "unrestricted"
            else:
                return "restricted"
