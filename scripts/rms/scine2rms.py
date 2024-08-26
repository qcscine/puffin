# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

"""
This script creates a RMS input file from a SCINE database.
Requires scine_chemoton.
"""

import scine_utilities as utils
import scine_database as db

from scine_puffin.utilities.rms_input_file_creator import create_rms_yml_file
from scine_chemoton.gears.kinetic_modeling.rms_network_extractor import ReactionNetworkData
from scine_chemoton.gears.kinetic_modeling.kinetic_modeling import KineticModeling
from scine_chemoton.utilities.model_combinations import ModelCombination
from scine_chemoton.gears.kinetic_modeling.atomization import ZeroEnergyReference
from scine_chemoton.utilities.db_object_wrappers.thermodynamic_properties import ReferenceState


def get_options() -> KineticModeling.Options:
    """
    Define the electronic structure models and options for the rate calculations here.
    """
    T = 150 + 273.15  # 150 degree Celsius
    p = utils.MOLAR_GAS_CONSTANT * T / 1e-3  # 1 mol/L

    model = db.Model("gfn2", "gfn2", "")
    model.solvation = "gbsa"
    model.solvent = "toluene"
    model.program = "xtb"
    model.pressure = p
    model.temperature = T

    dft_struc_model = db.Model("dft", "pbe-d3bj", "def2-sv(p)")
    dft_struc_model.solvation = "cosmo"
    dft_struc_model.solvent = "toluene"
    dft_struc_model.program = "turbomole"
    dft_struc_model.pressure = model.pressure
    dft_struc_model.temperature = model.temperature

    model_single_points = db.Model("dft", "pbe0-d3bj", "def2-tzvp")
    model_single_points.solvation = "cosmo"
    model_single_points.solvent = "toluene"
    model_single_points.program = "turbomole"
    model_single_points.pressure = model.pressure
    model_single_points.temperature = model.temperature

    options = KineticModeling.Options
    options.model_combinations = [ModelCombination(model_single_points, dft_struc_model),
                                  ModelCombination(model_single_points, model)]
    options.model_combinations_reactions = [ModelCombination(model_single_points, dft_struc_model),
                                            ModelCombination(model_single_points, model)]
    options.reference_state = ReferenceState(T, p)
    options.max_barrier = 300.0  # kJ/mol
    options.only_electronic = False
    options.min_flux_truncation = 1e-9
    return options


if __name__ == "__main__":
    manager = db.Manager()
    db_name = "my-database"
    db_ip = "my-database-ip"
    db_port = 27017
    credentials = db.Credentials(db_ip, db_port, db_name)
    manager.set_credentials(credentials)
    manager.connect(False, 60, 120)
    reactions = manager.get_collection("reactions")

    kinetic_modeling_options = get_options()
    refs = [ZeroEnergyReference(c.electronic_model) for c in kinetic_modeling_options.model_combinations]
    network_data = ReactionNetworkData(manager, kinetic_modeling_options, refs)

    rms_file_name = "chem.rms"
    solvent = "toluene"
    solvent_viscosity = None
    solvent_aggregate_index = None
    create_rms_yml_file(network_data.aggregate_ids, network_data.enthalpies, network_data.entropies,
                        network_data.prefactors, network_data.exponents, network_data.ea,
                        [db.ID(str_id) for str_id in network_data.reaction_ids], reactions, rms_file_name, solvent,
                        solvent_viscosity, solvent_aggregate_index)
