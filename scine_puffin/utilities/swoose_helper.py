from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
import numpy as np
from typing import List, Tuple, TYPE_CHECKING

from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_swoose") or TYPE_CHECKING:
    import scine_swoose as swoose
else:
    swoose = MissingDependency("scine_swoose")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class SFAMParameters:
    """
    Class representing a set of parameters of the SFAM method.
    This class stores the parameters for bonds, angles, dihedrals, impropers, charges, and non-covalent interactions.
    """

    def __init__(self) -> None:
        self.bonds: List[List[str]] = []
        self.angles: List[List[str]] = []
        self.dihedrals: List[List[str]] = []
        self.impropers: List[List[str]] = []
        self.charges: List[List[str]] = []
        self.non_covalent: List[List[str]] = []

    def __repr__(self) -> str:
        ret = "# MM parameters combined from two separate parameter files\n\n"

        ret += "! bonds\n"
        for b in self.bonds:
            ret += "  ".join(b) + "\n"

        ret += "\n! angles\n"
        for a in self.angles:
            ret += "  ".join(a) + "\n"

        ret += "\n! dihedrals\n"
        for d in self.dihedrals:
            ret += "  ".join(d) + "\n"

        ret += "\n! impropers\n"
        for id in self.impropers:
            ret += "  ".join(id) + "\n"

        ret += "\n! charges\n"
        for ch in self.charges:
            ret += "  ".join(ch) + "\n"

        ret += "\n! non-covalent\n"
        for nc in self.non_covalent:
            ret += "  ".join(nc) + "\n"

        ret += "\n! c6 coefficients\n*\n"
        return ret

    def calculate_mean_for_parameter(self, parameter_data_self: List[str], parameter_data_other: List[str],
                                     value_indices: List[int]) -> None:
        for number in value_indices:
            val = float(parameter_data_self[number])
            val += float(parameter_data_other[number])
            val /= 2
            parameter_data_self[number] = str(round(val, 5))

    def __iadd__(self, other) -> SFAMParameters:
        # bonds
        copy_of_self_bonds = self.bonds.copy()
        for m, other_b in enumerate(other.bonds):
            for n, b in enumerate(copy_of_self_bonds):
                if (other_b[0] == b[0] and other_b[1] == b[1]) or (other_b[0] == b[1] and other_b[1] == b[0]):
                    self.calculate_mean_for_parameter(self.bonds[n], other.bonds[m], [2, 3])
                    break
            else:
                self.bonds.append(other_b)

        # angles
        copy_of_self_angles = self.angles.copy()
        for m, other_a in enumerate(other.angles):
            for n, a in enumerate(copy_of_self_angles):
                if other_a[1] == a[1]:
                    if (other_a[0] == a[0] and other_a[2] == a[2]) or (other_a[0] == a[2] and other_a[2] == a[0]):
                        self.calculate_mean_for_parameter(self.angles[n], other.angles[m], [3, 4])
                        break
            else:
                self.angles.append(other_a)

        # dihedrals
        copy_of_self_dihedrals = self.dihedrals.copy()
        for m, other_d in enumerate(other.dihedrals):
            for n, d in enumerate(copy_of_self_dihedrals):
                if (other_d[1] == d[1] and other_d[2] == d[2]) or (other_d[1] == d[2] and other_d[2] == d[1]):
                    if (other_d[5] != d[5]) or (other_d[6] != d[6]):
                        raise RuntimeError("Problem with dihedrals, not agreeing on periodicity and phase shift!")
                    self.calculate_mean_for_parameter(self.dihedrals[n], other.dihedrals[m], [4])
                    break
            else:
                self.dihedrals.append(other_d)

        # improper dihedrals
        copy_of_self_impropers = self.impropers.copy()
        for m, other_id in enumerate(other.impropers):
            for n, id in enumerate(copy_of_self_impropers):
                if other_id[0] == id[0]:
                    if (other_id[1] == id[1]) and (other_id[2] == id[2]) and (other_id[3] == id[3]):
                        self.calculate_mean_for_parameter(self.impropers[n], other.impropers[m], [4, 5])
                        break
            else:
                self.impropers.append(other_id)

        # charges
        copy_of_self_charges = self.charges.copy()
        for _, other_ch in enumerate(other.charges):
            for m, self_ch in enumerate(copy_of_self_charges):
                if not other_ch[0] == self_ch[0]:
                    self.charges.append(other_ch)
                    break
        return self


def fill_container(params: List[str], start: int, container: List[List[str]]) -> None:
    """
    Fill a container with parameters from a parameter file.

    Parameters
    ----------
    params : List[str]
        The parameters from the parameter file.
    start : int
        The index at which to start reading the parameters.
    container : List[List[str]]
        The container to fill with the parameters.
    """
    index = start
    while params[index].strip() != "*" and params[index].strip():
        container.append(params[index].strip().split())
        index += 1


def parse_parameter(parameter_file: str, final_params: SFAMParameters) -> None:
    """
    Parse the parameter file and fill the corresponding containers in the `final_params` SFAMParameters class.

    Parameters
    ----------
    parameter_file : str
        The path to the parameter file.
    final_params : SFAMParameters
        The object containing the containers to be filled with the parsed parameters.
    """
    with open(parameter_file) as f1:
        parameters = f1.readlines()
    for i, line in enumerate(parameters):
        if "bond" in line:
            fill_container(parameters, i + 1, final_params.bonds)
        elif "angle" in line:
            fill_container(parameters, i + 1, final_params.angles)
        elif "dihedral" in line:
            fill_container(parameters, i + 1, final_params.dihedrals)
        elif "improper" in line:
            fill_container(parameters, i + 1, final_params.impropers)
        elif "charges" in line:
            fill_container(parameters, i + 1, final_params.charges)
        elif ("non-covalent" in line) or ("vdw" in line):
            fill_container(parameters, i + 1, final_params.non_covalent)


def combine_parameter_files(para_file_1: str, para_file_2: str, result: str) -> None:
    """
    Combine two parameter files into one, taking the mean of the parameters if they are present in both files.
    The result is written to the given file.

    Parameters
    ----------
    params1 : str
        The first parameter file.
    params2 : str
        The second parameter file.
    result : str
        The name of the file to write the combined parameters.
    """
    first_params = SFAMParameters()
    parse_parameter(para_file_1, first_params)
    other_params = SFAMParameters()
    parse_parameter(para_file_2, other_params)
    first_params += other_params  # combine the two

    with open(result, "w") as out:
        out.write(str(first_params))


@requires("swoose")
def write_bond_detected_connectivity_file(atoms: utils.AtomCollection, connectivity_filename: str) -> None:
    """
    Write a connectivity file based on bond detection.

    Parameters
    ----------
    atoms : utils.AtomCollection
        The atoms for which to write the connectivity file.
    connectivity_filename : str
        The name of the connectivity file to write.
    """
    bonds = utils.BondDetector.detect_bonds(atoms)
    topology = swoose.topology_utilities.generate_lists_of_neighbors(atoms.size(), bonds)
    swoose.utilities.write_connectivity_file(connectivity_filename, topology)


@requires("swoose")
def extract_connected_atoms(atoms: utils.AtomCollection, atom_indices: List[int]) -> List[int]:
    """
    Extract all atoms connected to a given set of atoms.

    Parameters
    ----------
    atoms : utils.AtomCollection
        The atoms of which to extract connected atoms.
    atom_indices : List[int]
        The indices of the atoms for which to extract connected atoms.

    Returns
    -------
    List[int]
        The indices of the connected atoms.
    """

    # Detect bonds and create neighbor list on the fly
    bonds = utils.BondDetector.detect_bonds(atoms)
    neighbour_list = swoose.topology_utilities.generate_lists_of_neighbors(atoms.size(), bonds)

    connected_atoms = list()
    checked_atoms = list()
    atoms_to_check = atom_indices

    while len(atoms_to_check) != 0:
        atom = atoms_to_check[0]
        if atom in checked_atoms:
            continue
        # Append and uniquify
        checked_atoms.append(atom)
        checked_atoms = list(set(checked_atoms))
        # Look up in neighbor, append and uniquify
        # Save for atoms without neighbors
        if len(neighbour_list[atom]) == 0:
            connected_atoms += [atom]
        else:
            connected_atoms += neighbour_list[atom]
        connected_atoms = list(set(connected_atoms))
        atoms_to_check = list(set(connected_atoms) - set(checked_atoms))
        if len(atoms_to_check) == 0:
            atoms_to_check = list(set(atom_indices) - set(checked_atoms))

    return connected_atoms


def get_simple_qm_region(atoms: utils.AtomCollection, name: str,
                         **simple_qm_region_settings) -> Tuple[List[int], utils.AtomCollection]:
    """
    Create a QM region around a set of center atoms.
    Initial Radius is given in Angstrom and then converted to Bohr

    Parameters
    ----------
    atoms : utils.AtomCollection
        Atoms to extract spherical QM regions.
    name : str
        Name of the output file the QM region is written to.

    Returns
    -------
    Tuple[List[int], utils.AtomCollection]
        A tuple containing the indices of the atoms in the QM region and the extracted region as atom collection.
    """
    radius_in_au = simple_qm_region_settings['radius'] * utils.BOHR_PER_ANGSTROM

    atoms_in_sphere = list()
    for center_atom in simple_qm_region_settings['qm_region_center_atoms']:
        tmp_radius = radius_in_au
        atoms_in_sphere += list(np.where(
            np.asarray([np.linalg.norm(i) for i in atoms.positions - atoms.positions[center_atom]]) <= tmp_radius)[0])
    atoms_in_sphere = list(set(atoms_in_sphere))

    connected_atoms_in_sphere = extract_connected_atoms(atoms, atoms_in_sphere)
    connected_atoms_in_sphere = sorted(connected_atoms_in_sphere)

    extracted_region = utils.AtomCollection()

    for atom_index in connected_atoms_in_sphere:
        extracted_region.push_back(utils.Atom(atoms.elements[atom_index], atoms.positions[atom_index]))

    utils.io.write(name + ".selection.xyz", extracted_region)

    return connected_atoms_in_sphere, extracted_region


def get_hydrogen_bonded_qm_region(atoms: utils.AtomCollection, name: str,
                                  **simple_qm_region_settings) -> Tuple[List[int], utils.AtomCollection]:
    """
    Obtain a QM region derived from hydrogen bonds around center atoms.
    For each non-hydrogen center, the threshold radius is calculated as the sum of the covalent radii of the center
    atom and a hydrogen atom, scaled by a given factor.
    For hydrogen centers, the threshold radius is calculated as the sum of the covalent radii of the center atom
    and the covalent radii of all other atoms in the structure, scaled by a given factor.

    Parameters
    ----------
    atoms : utils.AtomCollection
        The atoms to extract the QM region from.
    name : str
        The name of the output file the QM region is written to.

    Returns
    -------
    Tuple[List[int], utils.AtomCollection]
        A tuple containing the indices of the atoms in the QM region and the extracted region as atom collection.
    """

    scaling_factor = simple_qm_region_settings["scaling_factor"]

    # H Bond for all atoms in structure
    h_center_radii = np.array([(utils.ElementInfo.covalent_radius(atoms[i].element) +
                                utils.ElementInfo.covalent_radius(utils.ElementType.H)) *
                               scaling_factor for i in range(0, atoms.size())
                               ])
    atoms_in_sphere = list()
    for center_atom in simple_qm_region_settings['qm_region_center_atoms']:
        # H bonds from heteroatom to hydrogen atom
        if atoms[center_atom].element != utils.ElementType.H and\
           atoms[center_atom].element != utils.ElementType.D and\
           atoms[center_atom].element != utils.ElementType.T:
            # Heteroatom plus hydrogen atom times a scaling factor
            tmp_radius = (utils.ElementInfo.covalent_radius(atoms[center_atom].element) +
                          utils.ElementInfo.covalent_radius(utils.ElementType.H)) *\
                scaling_factor
            tmp_indices = np.where(np.array([
                                   np.linalg.norm(i) for i in atoms.positions - atoms.positions[center_atom]
                                   ]) <= tmp_radius)[0]
        # H bonds from hydrogen atom to all heteroatom
        else:
            tmp_indices = np.where(np.array([
                                   np.linalg.norm(i) for i in atoms.positions - atoms.positions[center_atom]
                                   ]) <= h_center_radii)[0]
        atoms_in_sphere += list(tmp_indices)

    atoms_in_sphere = list(set(atoms_in_sphere))

    connected_atoms_in_sphere = extract_connected_atoms(atoms, atoms_in_sphere)
    connected_atoms_in_sphere = sorted(connected_atoms_in_sphere)

    extracted_region = utils.AtomCollection()

    for atom_index in connected_atoms_in_sphere:
        extracted_region.push_back(utils.Atom(atoms.elements[atom_index], atoms.positions[atom_index]))

    utils.io.write(name + ".selection.xyz", extracted_region)

    return connected_atoms_in_sphere, extracted_region
