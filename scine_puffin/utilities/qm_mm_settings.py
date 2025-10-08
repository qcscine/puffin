# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List

from scine_puffin.utilities.scine_helper import SettingsManager
from scine_puffin.utilities.imports import module_exists, MissingDependency
if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    db = MissingDependency("scine_utilities")


# TODO: Checking for QM/MM or MM calculations like this seems like a bad idea. However, there is no method family
#  QM/MM or MM at the moment.
def contains_mm(model: db.Model) -> bool:
    method_family = model.method_family.lower()
    return "gaff" in method_family or "sfam" in method_family


def is_qm_mm(model: db.Model) -> bool:
    method_family = model.method_family.lower()
    return contains_mm(model) and "/" in method_family


def write_xml_parameter_files(parameter_property_ids: List[db.ID], properties: db.Collection) -> List[str]:
    file_names: List[str] = []
    for i, property_id in enumerate(parameter_property_ids):
        string_property = db.StringProperty(property_id, properties)
        file_name = "xml_parameters_" + str(i) + ".xml"
        with open(file_name, "w") as xmlFile:
            xmlFile.write(string_property.get_data())
        file_names.append(file_name)
    return file_names


def write_atom_types_file(atom_collection: utils.AtomCollection) -> str:
    name = "atom_types.dat"
    atom_types = [tup[1] for tup in atom_collection.residues]
    if "" in atom_types:
        raise ValueError("External atom types requested but none were provided for the structure!")
    with open(name, "w") as atom_types_file:
        atom_types_file.write("\n".join(atom_types))
    return name


def prepare_optional_settings(structure: db.Structure, calculation: db.Calculation,
                              settings_manager: SettingsManager, properties: db.Collection,
                              skip_qm_atoms: bool = False) -> utils.AtomCollection:
    """
    Prepare setting for QM/MM calculations. This includes writing the connectivity file, charge file, etc.
    """
    from scine_puffin.jobs.swoose_qmmm_forces import SwooseQmmmForces

    settings = calculation.get_settings()
    calculator_settings = settings_manager.calculator_settings
    method_family = calculation.get_model().method_family.lower()
    atom_collection = structure.get_atoms()
    if contains_mm(calculation.get_model()):
        atom_collection = annotated_structure_with_residue_labels(structure, properties)
        connectivity_file_name: str = "connectivity.dat"
        SwooseQmmmForces.write_connectivity_file(connectivity_file_name, properties, structure)
        calculator_settings['mm_connectivity_file'] = connectivity_file_name
        print("Writing connectivity file    :", connectivity_file_name)
        if "gaff" in method_family:
            charge_file_name = "atomic_charges.csv"
            SwooseQmmmForces.write_partial_charge_file(charge_file_name, properties, structure)
            calculator_settings['gaff_atomic_charges_file'] = charge_file_name
            print("Gaff point charge file       :", charge_file_name)
            if 'use_xml_parameters' in settings and settings["use_xml_parameters"]:
                print("Parameters from OpenMM XML files are used!")
                property_ids = structure.get_properties("openmm_xml_files")
                calculator_settings['openmm_xml_files'] = write_xml_parameter_files(property_ids, properties)
            if "use_external_atom_types" in settings and settings["use_external_atom_types"]:
                print("Gaff atom types file         :", charge_file_name)
                calculator_settings['gaff_atom_types_file'] = write_atom_types_file(atom_collection)
        if "sfam" in method_family:
            parameter_file_name: str = "sfam-parameters.dat"
            SwooseQmmmForces.write_parameter_file(parameter_file_name, properties, structure)
            calculator_settings['mm_parameter_file'] = parameter_file_name

    if is_qm_mm(calculation.get_model()):
        if not skip_qm_atoms:
            calculator_settings['qm_atoms'] = SwooseQmmmForces.get_qm_atoms(properties, structure)
        if "electrostatic_embedding" in settings:
            calculator_settings['electrostatic_embedding'] = settings['electrostatic_embedding']
            print("Using electrostatic embedding:", calculator_settings['electrostatic_embedding'])
    return atom_collection


def annotated_structure_with_residue_labels(structure: db.Structure, properties: db.Collection) -> utils.AtomCollection:
    """
    Add the residue information to the structure's atom collection.
    """
    residue_label_property_name = "residue_labels"
    atom_type_property_name = "atom_types"
    atom_collection = structure.get_atoms()

    residue_labels = ["" for _ in range(len(atom_collection))]
    atom_types = ["" for _ in range(len(atom_collection))]
    if structure.has_property(residue_label_property_name):
        print("Get residue labels")
        prop = db.StringProperty(structure.get_property(residue_label_property_name), properties)
        print("Interpret residue labels")
        residue_labels = prop.get_data().split()
    if structure.has_property(atom_type_property_name):
        prop = db.StringProperty(structure.get_property(atom_type_property_name), properties)
        atom_types = prop.get_data().split()
        print("Interpret atom types")

    if len(residue_labels) != len(atom_collection) or len(atom_types) != len(atom_collection):
        raise ValueError("There must be as many atom types/residue labels as there are atoms.")
    atom_collection.residues = [(label, atom_type, "A", i) for i, (label, atom_type)
                                in enumerate(zip(residue_labels, atom_types))]
    return atom_collection
