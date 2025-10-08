# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from typing import cast, Any, Dict, List, Tuple, Union, TYPE_CHECKING
import numpy as np
import copy

from scine_puffin.config import Configuration
from scine_puffin.utilities import scine_helper, swoose_helper
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from ..utilities.task_to_readuct_call import SubTaskToReaductCall
from .templates.scine_react_job import ReactJob

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")
if module_exists("scine_readuct") or TYPE_CHECKING:
    import scine_readuct as readuct
else:
    readuct = MissingDependency("scine_readuct")


class ScineKingfisher(ReactJob):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Kingfisher Job"
        self.exploration_key = "kf"
        kf_defaults = {
            "solvent_types": 1,
            "solvent_ratio": [1],
            "seed": 42,
            "nt_qm_region_radius": 5.0,
            "tsopt_qm_region_radius": 2.5,
            "tsopt_qm_region_hbond_based": False,
            "tsopt_qm_region_scaling_factor": 2.60,  # corresponds to radius of 2.5A derived from OH
            "tsextract_qm_region_radius": 1.75,
            "tsextract_qm_region_hbond_based": False,
            "tsextract_qm_region_scaling_factor": 2.0,  # corresponds to radius of 1.98A derived from OH
            "preopt_runs": 2,
            # need to be here because of the way the settings are merged
            "include_acceptors": False,
            "acceptors": [],
            "nt_associations": [],
            "nt_dissociations": [],
        }
        nt_defaults = {
            # # # Settings for the nt task
            "convergence_max_iterations": 600,
            "nt_total_force_norm": 0.1,  # 0.1 is the default
            "sd_factor": 1.0,
            "nt_use_micro_cycles": True,
            "nt_fixed_number_of_micro_cycles": True,
            "nt_number_of_micro_cycles": 20,  # 20 is the default
            "nt_filter_passes": 10,
            "nt_associations": [],
            "nt_dissociations": [],
            "output": ["qmmm_nt2"],
            "stop_on_error": False,
        }
        mm_opt_defaults = {
            "geoopt_coordinate_system": "cartesian",
            "convergence_requirement": 2,
            "convergence_delta_value": 1e-5,
            "convergence_step_max_coefficient": 1.0e-2,
            "convergence_step_rms": 5.0e-3,
            "convergence_gradient_max_coefficient": 1.0e-3,
            "convergence_gradient_rms": 5.0e-4,
            "convergence_max_iterations": 4000,
            "output": ["mm_opt"]
        }
        tsopt_defaults = {
            "geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "output": ["qmmm_ts"],
            "convergence_max_iterations": 200,
            "mm_optimizer": "bfgs",
            "qmmm_opt_env_start": False,
            "qmmm_opt_max_qm_microiterations": 19,
            "qmmm_opt_max_env_microiterations": 1000,
            "qmmm_opt_max_macroiterations": 50,
            "convergence_delta_value": 1e-6,
            "convergence_step_max_coefficient": 2.0e-3,
            "convergence_step_rms": 1.0e-3,
            "convergence_gradient_max_coefficient": 2.0e-4,
            "convergence_gradient_rms": 1.0e-4,
        }
        irc_defaults = {
            "output": ["qmmm_irc_forward", "qmmm_irc_backward"],
            "convergence_max_iterations": 500,
            "convergence_delta_value": 1e-6,
            "convergence_step_max_coefficient": 2.0e-3,
            "convergence_step_rms": 1.0e-3,
            "convergence_gradient_max_coefficient": 2.0e-4,
            "convergence_gradient_rms": 1.0e-4,
            "sd_factor": 2.5,
            "sd_use_trust_radius": True,
            "sd_trust_radius": 0.1,
            "stop_on_error": False,
        }
        ircopt_defaults = {
            "stop_on_error": True,
            "geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "qmmm_opt_env_start": False,
            "convergence_delta_value": 1e-6,
            "convergence_step_max_coefficient": 2.0e-3,
            "convergence_step_rms": 1.0e-3,
            "convergence_gradient_max_coefficient": 2.0e-4,
            "convergence_gradient_rms": 1.0e-4,
            "convergence_max_iterations": 5000,
            "qmmm_opt_env_switch_off": 15,
            "qmmm_opt_max_macroiterations": 100
        }
        qmmm_preopt_defaults = {
            "geoopt_coordinate_system": "cartesian",
            "qmmm_opt_env_start": False,
            # Factor 10 looser than normal
            "convergence_delta_value": 1e-6,
            "convergence_step_max_coefficient": 1.0e-2,
            "convergence_step_rms": 5.0e-3,
            "convergence_gradient_max_coefficient": 1.0e-3,
            "convergence_gradient_rms": 5.0e-4,
            "convergence_max_iterations": 300,
            "qmmm_opt_env_switch_off": 30,
            "qmmm_opt_max_macroiterations": 50,
            "stop_on_error": False,
            "output": ["opt"]
        }
        opt_defaults = {
            "convergence_max_iterations": 500,
            "geoopt_coordinate_system": "cartesianWithoutRotTrans",
        }
        self.settings = {
            **self.settings,
            "kf": kf_defaults,
            "kfp": {},
            "mmopt": mm_opt_defaults,
            "preopt": qmmm_preopt_defaults,
            "nt": nt_defaults,
            "tsopt": tsopt_defaults,
            "irc": irc_defaults,
            "ircopt": ircopt_defaults,
            self.opt_key: opt_defaults
        }
        # # # No spin propensity check here
        self.settings[self.propensity_key]['check'] = 0
        # # # Expect charge separation
        self.settings[self.single_point_key]['expect_charge_separation'] = True
        self.settings[self.single_point_key]['charge_separation_threshold'] = 0.5
        # # # Allow flask decomposition
        self.settings[self.job_key]['allow_exhaustive_product_decomposition'] = True
        # # # Is a QM/MM calculation
        self._is_qmmm_calculation = True

    @requires("database")
    def write_parameter_file(self, parameter_file_name: str, structure: db.Structure) -> None:
        try:
            parameters = db.StringProperty(structure.get_property('sfam_parameters'))
        except RuntimeError as e:
            raise RuntimeError('SFAM-parameters are missing as properties of the structure ' + structure.id().string() +
                               ' in QM/MM.') from e
        parameters.link(self._properties)
        with open(parameter_file_name, 'w') as p_file:
            p_file.write(parameters.get_data())

    @requires("database")
    def _get_charge_and_multiplicity_of_rc(self, end_index: int) -> Tuple[int, int]:

        from scine_utilities.settings_names import molecular_charge, spin_multiplicity

        start_structure_ids = self._calculation.get_structures()
        start_structures = [db.Structure(sid, self._structures) for sid in start_structure_ids[:end_index]]

        default_charge = 0
        multi_diff = 0
        multi_sum = 0
        for start_structure in start_structures:
            default_charge += start_structure.get_charge()
            multi_diff -= start_structure.get_multiplicity()
            multi_sum += start_structure.get_multiplicity()
        # NOTE: Logic adapted from 'determine_pes_of_rc' in scine_react_job_template
        min_mult = abs(multi_diff) + 1  # max spin recombination
        max_mult = multi_sum - 1  # no spin recombination between molecules
        default_mult = min_mult if self.settings[self.rc_key]["minimal_spin_multiplicity"] else max_mult

        # pick values from settings, otherwise defaults
        charge = self.settings[self.rc_key].get(molecular_charge, default_charge)
        multiplicity = self.settings[self.rc_key].get(spin_multiplicity, default_mult)

        return charge, multiplicity

    @staticmethod
    @requires("utilities")
    def create_microdroplet(solute: utils.AtomCollection,
                            solute_charge: float,
                            solvents: List[utils.AtomCollection],
                            solvent_charges: List[float],
                            seed: int = 42,
                            solvents_ratio: Union[None, List[int]] = None,
                            **custom_settings) -> Tuple[utils.AtomCollection, int, str]:
        """
        Create a microdroplet around a solute with a given number of shells of solvents.

        Parameters
        ----------
        solute : utils.AtomCollection
            The solute.
        solute_charge : float
            The charge of the solute.
        solvents : List[utils.AtomCollection]
            A list of solvents.
        solvent_charges : List[float]
            A list of charges for the solvents.
        seed : int, optional
            A random seed for the placing algorithm, by default 42.
        solvents_ratio : List[int], optional
            A list of the ratios of the solvents, by default [1]

        Returns
        -------
        Tuple[utils.AtomCollection, int, str]
            The microdroplet, the total charge of the microdroplet, and the path to the XYZ file.
        """

        name = 'solute'
        shells = custom_settings['shells'] if "shells" in custom_settings.keys() else 2
        # # # Overwrite from kfp settings
        placement_settings = utils.solvation.placement_settings()
        for key, value in custom_settings.items():
            if hasattr(placement_settings, key):
                setattr(placement_settings, key, value)
            else:
                print(key + " not an attribute of the placement settings.")

        if solvents_ratio is None:
            solvents_ratio = [1]

        if len(solvents) == 1:
            solvent_vector = utils.solvation.solvate_shells(solute, solute.size(), solvents[0], shells, seed,
                                                            placement_settings)
            n_solvents = np.sum([len(shell) for shell in solvent_vector])
            total_solvent_charge = n_solvents * solvent_charges[0]
        else:
            solvent_vector, solvent_indices = utils.solvation.solvate_shells_mix(solute, solute.size(), solvents,
                                                                                 solvents_ratio, shells, seed,
                                                                                 placement_settings)
            total_solvent_charge = 0.0
            for solvent_index in utils.solvation.merge_solvent_shell_indices_vector(solvent_indices):
                total_solvent_charge += solvent_charges[solvent_index]

        cluster = solute + utils.solvation.merge_solvent_shell_vector(solvent_vector)
        cluster_charge = solute_charge + total_solvent_charge
        xyz_path = name + "_nShells" + str(shells) + "_seed" + str(seed) + "_clusterSize" + str(cluster.size()) + ".xyz"

        utils.io.write(xyz_path, cluster)

        return cluster, cluster_charge, xyz_path

    @staticmethod
    @requires("utilities")
    def select_mode(hessian: np.ndarray, structure: utils.AtomCollection, relevant_atoms: List[int]) -> int:
        """
        Selects the index of the eigenvector of the non-mass weighted Hessian matrix
        with the highest score based on negative eigenvalues and contribution of relevant atoms.
        This is required as the TS optimizer is utilizing a non-mass weighted Hessian matrix as well.

        Parameters
        ----------
        hessian : np.ndarray
            The Hessian matrix.
        structure : utils.AtomCollection
            The atom collection representing the molecular structure.
        relevant_atoms : List[int]
            The list of relevant atom indices.

        Returns
        -------
        int
            The index of the selected mode.
        """
        modes_container = utils.normal_modes.calculate(hessian, structure.elements, structure.positions,
                                                       normalize=True, mass_weighted=False)
        wavenumbers = modes_container.get_wave_numbers()
        contribution_of_atoms = list()
        im_wavenumbers = list()
        sqrt_masses = np.array([[utils.ElementInfo.mass(structure.elements[i])**0.5]
                               * 3 for i in range(len(structure))])

        # Loop over modes with imaginary frequencies
        for mode_index in range(len(wavenumbers)):
            if wavenumbers[mode_index] > 0.0:
                break
            # NOTE: Modes in mode container are not mass weighted
            tmp_mode = modes_container.get_mode(mode_index) * sqrt_masses
            # renormalize
            tmp_mode_norm = tmp_mode / np.linalg.norm(tmp_mode)
            tmp_contribution = np.float64(0.0)
            # Loop over contribution of relevant atoms
            for atom_index in relevant_atoms:
                # Squared norm to obtain relative contribution to all of mode
                tmp_contribution += np.linalg.norm(tmp_mode_norm[atom_index])**2
            im_wavenumbers.append(wavenumbers[mode_index])
            contribution_of_atoms.append(tmp_contribution)

        im_wavenumbers_array = np.array(im_wavenumbers)
        im_wavenumbers_array *= 1.0 / np.min(im_wavenumbers_array)
        contribution_of_atoms_array = np.array(contribution_of_atoms)
        # As in ReaDuct
        mode_score = im_wavenumbers_array * 0.5 + contribution_of_atoms_array * 0.5
        selected_mode = np.argmax(mode_score)

        return int(selected_mode)

    @staticmethod
    @requires("utilities")
    def get_contribution_per_atom_of_mode(normal_mode: np.ndarray, elements: utils.ElementTypeCollection) -> np.ndarray:
        """
        Calculate the relative contribution of each atom to a normal mode given in Cartesian coordinates
        in mass-weighted coordinates.
        The normal mode is transferred to mass-weighted coordinates and then re-normalized.
        The squared norm of each atom's contribution to the mode is calculated and returned.

        Parameters
        ----------
        normal_mode : np.ndarray
            The normal mode in Cartesian coordinates.
        elements : utils.ElementTypeCollection
            The elements of the atoms in the normal mode.

        Returns
        -------
        np.ndarray
            The relative contribution of each atom to the normal mode.
        """

        contribution_per_atom = list()
        sqrt_masses = np.array([[utils.ElementInfo.mass(element)**0.5] * 3 for element in elements])
        tmp_mode = normal_mode * sqrt_masses
        # renormalize
        tmp_mode_norm = tmp_mode / np.linalg.norm(tmp_mode)
        # Loop over rows of mode matrix
        for i in range(len(normal_mode)):
            # Squared norm corresponds to relative contribution
            contribution_per_atom.append(np.linalg.norm(tmp_mode_norm[i])**2)

        return np.asarray(contribution_per_atom)

    @job_configuration_wrapper
    @requires("readuct")
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        # Everything that calls SCINE is enclosed in a try/except block
        with breakable(calculation_context(self)):
            # Sort settings
            qmmm_model = self._calculation.get_model()
            print("QMMM model:\n", qmmm_model)
            qmmm_settings_manager = scine_helper.SettingsManager(qmmm_model.method_family,
                                                                 qmmm_model.program)
            # This overwrites any default settings by user settings
            qmmm_settings_manager.separate_settings(self._calculation.get_settings())
            self.sort_settings(qmmm_settings_manager.task_settings)
            print(qmmm_settings_manager.task_settings)

            # NOTE: First structures in list are reactants, then solvents
            final_parameters: Union[swoose_helper.SFAMParameters, None] = None
            first_solvent_index = len(self._calculation.get_structures()) - self.settings["kf"]["solvent_types"]
            solvents: List[utils.AtomCollection] = []
            solvent_charges: List[float] = []
            for i, structure_id in enumerate(self._calculation.get_structures()):
                structure = db.Structure(structure_id, self._structures)
                if i < first_solvent_index:
                    name = "reactant_{:02d}".format(i)
                else:
                    name = "solvent_{:02d}".format(i - first_solvent_index)
                    solvents.append(structure.get_atoms())
                    solvent_charges.append(structure.get_charge())
                # Write parameter and combine
                self.write_parameter_file(name + ".sfam", structure)
                tmp_parameters = swoose_helper.SFAMParameters()
                swoose_helper.parse_parameter(name + ".sfam", tmp_parameters)
                if final_parameters is None:
                    final_parameters = tmp_parameters
                else:
                    final_parameters += tmp_parameters
                # Write xyz
                xyz_name = name + ".xyz"
                utils.io.write(xyz_name, structure.get_atoms())

            with open("Parameters.dat", "w") as out:
                out.write(str(final_parameters))
            # Parameter file for mm calculator and qm/mm calculator
            self.settings['kf']['mm_parameter_file'] = "Parameters.dat"

            """ Setup Microdroplet """
            # Set ref structure
            self.ref_structure = self.check_structures(self._calculation.get_structures()[:first_solvent_index])
            # Build reactive complex
            reactive_complex_atoms = self.build_reactive_complex(end_index=first_solvent_index)
            utils.io.write("reactive_complex.xyz", reactive_complex_atoms)

            reactive_complex_charge, reactive_complex_multiplicity = self._get_charge_and_multiplicity_of_rc(
                end_index=first_solvent_index)
            # Create microdroplet around reactive complex
            microdroplet, microdroplet_charge, microdroplet_file = self.create_microdroplet(
                reactive_complex_atoms, reactive_complex_charge,
                solvents, solvent_charges,
                seed=self.settings["kf"]["seed"],
                solvents_ratio=self.settings["kf"]["solvent_ratio"],
                **self.settings['kfp'])
            # # # From derive connectivity
            connectivity_file = "Connectivity.dat"
            swoose_helper.write_bond_detected_connectivity_file(microdroplet, connectivity_file)
            self.settings['kf']['mm_connectivity_file'] = connectivity_file

            # # # Run MM optimization
            mm_name = "mm_start"
            mm_settings_manager = scine_helper.SettingsManager(qmmm_model.method_family.split("/")[1],
                                                               qmmm_model.program.split("/")[1])
            # This overwrites any default settings by user settings
            mm_settings_manager.separate_settings(self._calculation.get_settings())
            mm_settings_manager.calculator_settings.update(self.settings["kf"])
            self.systems[mm_name] = utils.core.load_system_into_calculator(microdroplet_file,
                                                                           qmmm_model.method_family.split("/")[1],
                                                                           **mm_settings_manager.calculator_settings)
            self.settings["mmopt"]['geoopt_constrained_atoms'] = list(range(reactive_complex_atoms.size()))

            # # # # MM Optimization, print settings
            print("# " * 3 + "MM Calculator Settings " + "# " * 3)
            print(mm_settings_manager.calculator_settings.as_dict(), "\n")
            print("# " * 3 + "MM Optimization Settings " + "# " * 3)
            print(self.settings["mmopt"], "\n")
            self.systems, _ = readuct.run_opt_task(self.systems, [mm_name], **self.settings["mmopt"])

            # # # Prepare QMMM calculator
            qmmm_settings_manager.calculator_settings['molecular_charge'] = microdroplet_charge
            qmmm_settings_manager.calculator_settings['spin_multiplicity'] = reactive_complex_multiplicity
            qmmm_settings_manager.calculator_settings['qm_atoms'] = [i for i in range(0, reactive_complex_atoms.size())]
            qmmm_settings_manager.calculator_settings['electrostatic_embedding'] = True
            qmmm_settings_manager.calculator_settings['method'] = qmmm_model.method
            qmmm_settings_manager.calculator_settings.update(self.settings["kf"])

            qmmm_name = "qmmm_opt"
            self.systems[qmmm_name] = utils.core.load_system_into_calculator(
                self.settings['mmopt']['output'][0] + "/mm_opt.xyz",
                qmmm_model.method_family, **qmmm_settings_manager.calculator_settings)
            qm_region_settings: Dict[str, Any] = {}
            qm_region_settings['qm_region_center_atoms'] = list(
                set(self.settings['kf']['nt_dissociations'] +
                    self.settings['kf']['nt_associations'] + self.settings['kf']['acceptors']))

            qm_region_settings['radius'] = self.settings['kf']['nt_qm_region_radius']
            qm_region_nt2, _ = swoose_helper.get_simple_qm_region(self.get_system(qmmm_name).structure, qmmm_name,
                                                                  **qm_region_settings)
            self.get_system(qmmm_name).settings["qm_atoms"] = list(
                set(cast(List[int], self.get_system(qmmm_name).settings["qm_atoms"]) + qm_region_nt2))
            print(len(self.get_system(qmmm_name).settings["qm_atoms"]),  # type: ignore
                  self.get_system(qmmm_name).settings["qm_atoms"])

            # # # Start QM/MM Pre-Optimization

            loose_out = "qmmm_pre_opt"

            for i in range(self.settings['kf']['preopt_runs']):
                if len(loose_out.split("_")) == 3:
                    loose_out += "_" + str(i + 1)
                else:
                    loose_out_list = loose_out.split("_")
                    loose_out_list[-1] = str(i + 1)
                    loose_out = "_".join(loose_out_list)
                self.settings['preopt']['output'] = [loose_out]
                # Constrain solute atoms here, just relax close solvents
                self.settings['preopt']['geoopt_constrained_atoms'] = [
                    i for i in range(0, reactive_complex_atoms.size())]

                print("# " * 3 + "QM/MM Calculator Settings " + "# " * 3)
                print(self.get_system(qmmm_name).settings.as_dict(), "\n")
                print("# " * 3 + "QM/MM Loose Optimization Settings " + "# " * 3)
                print(self.settings['preopt'], "\n")
                self.systems, _ = self.observed_readuct_call(
                    SubTaskToReaductCall.OPT, self.systems, [qmmm_name], **self.settings['preopt'])
                qmmm_name = loose_out
                # Redefine QM Region after first partial optimization
                if i == 0:
                    loose_out_calc = self.get_system(loose_out)
                    loose_out_calc.settings["qm_atoms"] = list(set([
                        i for i in range(0, reactive_complex_atoms.size())] +
                        swoose_helper.get_simple_qm_region(loose_out_calc.structure, qmmm_name,
                                                           **qm_region_settings)[0]))
                    print("QM Atoms after 1st PreOpt:\n",
                          len(cast(List[int], self.get_system(loose_out).settings["qm_atoms"])),
                          self.get_system(loose_out).settings["qm_atoms"])
                    # Start next round with MM Opt first
                    self.settings['preopt']['qmmm_opt_env_start'] = True

            # # # Start QM/MM NT2
            self.settings["nt"]['nt_associations'] = self.settings["kf"]['nt_associations']
            self.settings["nt"]['nt_dissociations'] = self.settings["kf"]['nt_dissociations']
            print("# " * 3 + "QM/MM Calculator Settings " + "# " * 3)
            print(self.get_system(qmmm_name).settings.as_dict(), "\n")
            print("# " * 3 + "QM/MM NT2 Settings " + "# " * 3)
            print(self.settings["nt"], "\n")
            self.systems, success = self.observed_readuct_call(
                SubTaskToReaductCall.NT2, self.systems, [qmmm_name], **self.settings["nt"])
            # Handle failure of NT2
            if not success:
                self.verify_connection()
                calculation.set_comment(calculation.get_comment() + self.name + " QM/MM NT2 Job: No TS guess found.")
                self.capture_raw_output()
                # update model because job will be marked complete
                # use start calculator because nt might have last failed calculation
                scine_helper.update_model(
                    self.get_system(qmmm_name), calculation, self.config
                )
                raise breakable.Break
            self.systems, success = readuct.run_hessian_task(self.systems, self.output("nt"))
            self.throw_if_not_successful(success, self.systems, self.output("nt"), ["energy", "partial_hessian"],
                                         "Partial Hessian calculation failed for the QM/MM NT2 TS guess structure.\n")
            nt_out_name = self.output("nt")[0]

            # # # New QM/MM Region for TSGuess
            nt2_results = self.get_system(nt_out_name).get_results()
            nt2_structure = self.get_system(nt_out_name).structure
            relevant_atoms = list(set(self.settings["nt"]['nt_dissociations'] + self.settings["nt"]['nt_associations']))
            selected_mode_index_nt2 = self.select_mode(nt2_results.partial_hessian, nt2_structure, relevant_atoms)
            print("Selected Mode after Analysis: ", selected_mode_index_nt2)

            # # # QM Region Selection for TS Opt
            # # # # Identify atoms which contribute to the TS Mode
            # # # Not mass-weighted as TS Opt selection can't be with mass weighted hessian
            assert nt2_results.partial_hessian
            nt2_modes_container = utils.normal_modes.calculate(nt2_results.partial_hessian,
                                                               nt2_structure.elements, nt2_structure.positions,
                                                               normalize=True, mass_weighted=False)
            atom_contribution = self.get_contribution_per_atom_of_mode(
                nt2_modes_container.get_mode(selected_mode_index_nt2), nt2_structure.elements)
            atom_contribution_normalized = atom_contribution  # / np.max(atom_contribution)
            print("Atom Contribution in TS Mode:\n", atom_contribution_normalized, np.sum(atom_contribution_normalized))
            # # # NOTE: Threshold might be different here
            min_rel_atom_contribution = np.min([atom_contribution_normalized[i] for i in relevant_atoms])
            relevant_qm_atoms = list(np.where(atom_contribution_normalized >= min_rel_atom_contribution)[0])
            print("Relevant Atoms (mode analysis):", min_rel_atom_contribution)

            swoose_helper.write_bond_detected_connectivity_file(nt2_structure, "Connectivity.nt2.dat")

            # # Only rely on mode analysis, if any atom of the qm atoms
            atoms_for_ts_opt = []
            if any(atom in relevant_qm_atoms for atom in relevant_atoms):
                atoms_for_ts_opt = list(set(relevant_qm_atoms))
            else:
                print("Ignoring Mode Analysis for TS QM Region")
                atoms_for_ts_opt = list(set(relevant_atoms))

            print("Pre TS Opt - Atom Selection without QM Region around Acceptor: ", atoms_for_ts_opt)

            if self.settings['kf']['include_acceptors']:
                qm_region_settings['qm_region_center_atoms'] = list(set(
                    atoms_for_ts_opt + self.settings['kf']['acceptors']))
            else:
                print("No QM Acceptor for QM Region in TS Opt")
                qm_region_settings['qm_region_center_atoms'] = list(set(atoms_for_ts_opt))

            qmmm_tsguess_name = "qmmm_tsguess"
            # # # Select QM Region for TS Opt
            if not self.settings['kf']['tsopt_qm_region_hbond_based']:
                qm_region_settings['radius'] = self.settings['kf']['tsopt_qm_region_radius']
                qm_region_tsopt, _ = swoose_helper.get_simple_qm_region(nt2_structure, qmmm_tsguess_name,
                                                                        **qm_region_settings)
            else:
                qm_region_settings['scaling_factor'] = self.settings['kf']['tsopt_qm_region_scaling_factor']
                qm_region_tsopt, _ = swoose_helper.get_hydrogen_bonded_qm_region(nt2_structure, qmmm_tsguess_name,
                                                                                 **qm_region_settings)

            qmmm_settings_manager.calculator_settings['qm_atoms'] = qm_region_tsopt
            # # # New QM Region! -- Needs MM at the beginning
            mm_name = "mm_tsguess"
            mm_settings_manager.calculator_settings.update(self.settings["kf"])
            print(mm_settings_manager.calculator_settings.as_dict())
            self.systems[mm_name] = utils.core.load_system_into_calculator(
                self.settings["nt"]["output"][0] + "/qmmm_nt2.xyz",
                qmmm_model.method_family.split("/")[1],
                **mm_settings_manager.calculator_settings)
            # # # Run stricter MM optimization
            self.settings["mmopt"]['convergence_requirement'] += 1
            self.settings["mmopt"]['convergence_delta_value'] /= 10.0
            self.settings["mmopt"]['geoopt_constrained_atoms'] = qm_region_tsopt
            self.settings["mmopt"]['output'] = [mm_name]

            # # # # MM Optimization, print settings
            print("# " * 3 + "MM Calculator Settings " + "# " * 3)
            print(mm_settings_manager.calculator_settings.as_dict(), "\n")
            print("# " * 3 + "MM Optimization Settings " + "# " * 3)
            print(self.settings["mmopt"], "\n")
            self.systems, _ = readuct.run_opt_task(
                self.systems, [mm_name], **self.settings["mmopt"])

            # # # MM Opt with frozen core qm region
            self.systems[qmmm_tsguess_name] = utils.core.load_system_into_calculator(
                self.settings["mmopt"]['output'][0] + "/mm_tsguess.xyz",
                qmmm_model.method_family, **qmmm_settings_manager.calculator_settings)
            print("# " * 3 + "QM/MM Calculator Settings " + "# " * 3)
            print(self.get_system(qmmm_tsguess_name).settings.as_dict(), "\n")
            self.systems, success = readuct.run_hessian_task(self.systems, [qmmm_tsguess_name],
                                                             **{"output": [qmmm_tsguess_name]})
            self.throw_if_not_successful(success, self.systems, [qmmm_tsguess_name], ["energy", "partial_hessian"],
                                         "QM/MM TS guess partial Hessian calculation failed.\n")

            # # # # # Mode Selection with smaller QM/MM Region
            qmmm_tsguess_results = self.get_system(qmmm_tsguess_name).get_results()
            qmmm_tsguess_structure = self.get_system(qmmm_tsguess_name).structure
            selected_mode_index: int = self.select_mode(qmmm_tsguess_results.partial_hessian,
                                                        qmmm_tsguess_structure, relevant_atoms)

            print("Selected Mode for TS Opt:", selected_mode_index)
            # NOTE: maybe let ReaDuct handle this (['automatic_mode_selection'] = relevant_atoms)
            self.settings['tsopt']['bofill_follow_mode'] = selected_mode_index

            qm_settings_manager = scine_helper.SettingsManager(qmmm_model.method_family.split("/")[0],
                                                               qmmm_model.program.split("/")[0])
            qm_calculator_settings_from_qmmm = copy.deepcopy(qmmm_settings_manager.calculator_settings)
            del qm_calculator_settings_from_qmmm["program"]
            qm_settings_manager.calculator_settings.update(qm_calculator_settings_from_qmmm)
            print("# " * 3 + "QM Calculator Settings " + "# " * 3)
            print(qm_settings_manager.calculator_settings.as_dict(), "\n")
            # This overwrites any default settings by user settings
            qm_settings_manager.separate_settings(self._calculation.get_settings())
            self.sort_settings(qm_settings_manager.task_settings)
            print(qm_settings_manager.task_settings)
            # NOTE: Keep on going if the TS Opt did not fail
            try:
                self._tsopt_hess_irc_ircopt_postprocessing(qmmm_tsguess_name, qmmm_settings_manager,
                                                           program_helper=None,
                                                           irc_sanity_settings_manager=qm_settings_manager)
            except BaseException as react_error:
                # Raise exception, if something in React template fails and no TS Opt is available
                if not self.get_system(self.output("tsopt")[0]).has_results():
                    print("Kingfisher job failed while determining the QM/MM minimum energy path.")
                    print("No QM/MM TS could be determined, react job failed with error message:")
                    raise react_error
                else:
                    if hasattr(react_error, "args") and len(react_error.args) > 0:
                        error_message = react_error.args[0]
                    else:
                        error_message = "No error message available."
                    print("Kingfisher job failed while determining the QM/MM minimum energy path.")
                    print("A QM/MM TS could be determined, hence we will exract a QM TS guess.")
                    print("The react job failed with error message:")
                    print(error_message)

            # # # Start TS Extraction based on QM Region Selector and Acceptor
            # For TS_GUESS
            ts_opt_name = self.output("tsopt")[0]
            ts_results = self.get_system(ts_opt_name).get_results()
            ts_structure = self.get_system(ts_opt_name).structure
            # # # Mode analysis
            # # # Mass-weighted Hessian for mode analysis of TS as only one mode should be there
            assert ts_results.partial_hessian
            ts_modes_container = utils.normal_modes.calculate(ts_results.partial_hessian,
                                                              ts_structure.elements, ts_structure.positions)
            ts_atom_contribution = self.get_contribution_per_atom_of_mode(
                ts_modes_container.get_mode(0), ts_structure.elements)
            ts_atom_contribution_normalized = ts_atom_contribution

            print("Contribution of Relevant Atoms:\n", [ts_atom_contribution_normalized[i] for i in relevant_atoms])
            min_rel_atom_contribution = np.min([ts_atom_contribution_normalized[i] for i in relevant_atoms])
            ts_relevant_atoms = list(np.where(ts_atom_contribution_normalized >= min_rel_atom_contribution)[0])
            print("Min Contribution", np.min([ts_atom_contribution_normalized[i] for i in relevant_atoms]))
            print("Before QM Region:", ts_relevant_atoms)

            swoose_helper.write_bond_detected_connectivity_file(ts_structure, "Connectivity.qmmm_ts.dat")

            if self.settings['kf']['include_acceptors']:
                qm_region_settings['qm_region_center_atoms'] = list(set(
                    ts_relevant_atoms + self.settings['kf']['acceptors']))
            else:
                print("No QM Acceptor for QM Region in TS Opt")
                qm_region_settings['qm_region_center_atoms'] = list(set(ts_relevant_atoms))

            qmmm_tsopt_name = "qmregion_tsopt"
            # # # Select QM Region for TS Opt
            if not self.settings['kf']['tsextract_qm_region_hbond_based']:
                qm_region_settings['radius'] = self.settings['kf']['tsextract_qm_region_radius']
                qm_region_tsopt, qm_ts_guess = swoose_helper.get_simple_qm_region(ts_structure, qmmm_tsopt_name,
                                                                                  **qm_region_settings)
            else:
                qm_region_settings['scaling_factor'] = self.settings['kf']['tsextract_qm_region_scaling_factor']
                qm_region_tsopt, qm_ts_guess = swoose_helper.get_hydrogen_bonded_qm_region(ts_structure,
                                                                                           qmmm_tsopt_name,
                                                                                           **qm_region_settings)
            print("QM Region for pure QM TS guess:\n", len(qm_region_tsopt), qm_region_tsopt)
            # # # All QM from QMMM TS Opt
            qmmm_settings_manager.calculator_settings['qm_atoms'] = [i for i in range(0, qm_ts_guess.size())]
            # Store a TS Guess with QM Model of QMMM calculation
            self.systems[qmmm_tsopt_name] = utils.core.load_system_into_calculator(
                qmmm_tsopt_name + ".selection.xyz",
                qmmm_model.method_family.split("/")[0],
                **qm_settings_manager.calculator_settings)
            _, qm_ts_guess_struct = self._store_ts_with_propensity_info(
                qmmm_tsopt_name, None, db.Label.TS_GUESS, store_hessian=False)
            self.store_qm_atoms([i for i in range(0, qm_ts_guess.size())], qm_ts_guess_struct)
            # # # Store reactive atoms for follow up TS Opt
            self.store_property(
                self._properties, "reactive_atoms", "VectorProperty", relevant_atoms,
                self._calculation.get_model(), self._calculation, qm_ts_guess_struct,
            )

        return self.postprocess_calculation_context()
