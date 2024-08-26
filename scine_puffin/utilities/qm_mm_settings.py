# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from scine_puffin.utilities.scine_helper import SettingsManager
from scine_puffin.utilities.imports import module_exists, MissingDependency
if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


# TODO: Checking for QM/MM or MM calculations like this seems like a bad idea. However, there is no method family
#  QM/MM or MM at the moment.
def contains_mm(model: db.Model) -> bool:
    method_family = model.method_family.lower()
    return "gaff" in method_family or "sfam" in method_family


def is_qm_mm(model: db.Model) -> bool:
    method_family = model.method_family.lower()
    return contains_mm(model) and "/" in method_family


def prepare_optional_settings(structure: db.Structure, calculation: db.Calculation,
                              settings_manager: SettingsManager, properties: db.Collection,
                              skip_qm_atoms: bool = False) -> None:
    from scine_puffin.jobs.swoose_qmmm_forces import SwooseQmmmForces

    settings = calculation.get_settings()
    calculator_settings = settings_manager.calculator_settings
    method_family = calculation.get_model().method_family.lower()
    if contains_mm(calculation.get_model()):
        connectivity_file_name: str = "connectivity.dat"
        SwooseQmmmForces.write_connectivity_file(connectivity_file_name, properties, structure)
        calculator_settings['mm_connectivity_file'] = connectivity_file_name
        print("Writing connectivity file ", connectivity_file_name)
        if "gaff" in method_family:
            charge_file_name = "atomic_charges.csv"
            SwooseQmmmForces.write_partial_charge_file(charge_file_name, properties, structure)
            calculator_settings['gaff_atomic_charges_file'] = charge_file_name
            print("Gaff point charge file: ", charge_file_name)
        if "sfam" in method_family:
            parameter_file_name: str = "sfam-parameters.dat"
            SwooseQmmmForces.write_connectivity_file(parameter_file_name, properties, structure)
            calculator_settings['mm_parameter_file'] = parameter_file_name

    if is_qm_mm(calculation.get_model()):
        if not skip_qm_atoms:
            calculator_settings['qm_atoms'] = SwooseQmmmForces.get_qm_atoms(properties, structure)
        if "electrostatic_embedding" in settings:
            calculator_settings['electrostatic_embedding'] = settings['electrostatic_embedding']
            print("Using electrostatic embedding: ", calculator_settings['electrostatic_embedding'])
