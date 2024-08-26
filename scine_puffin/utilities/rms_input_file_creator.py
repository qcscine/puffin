# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Dict, Optional, Tuple, Union, Any, TYPE_CHECKING
import yaml
import numpy as np

from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


def create_single_species_entry(name: str, h: float, s: float):
    """
    Get an entry for a species in the RMS format.

    We could extend this function to use the heat capacity of the species or directly insert the NASA polynomials.
    This would be interesting for non-constant temperature simulations.

    Parameters
    ----------
    name : str
        The species name (unique identifier).
    h : float
        The species enthalpy in J/mol.
    s : float
        The species entropy in J/mol/K.

    Returns
    -------
    The species entry.
    """
    species_type_str = "Species"
    # NASA Polynomials: https://reactionmechanismgenerator.github.io/RMG-Py/reference/thermo/nasa.html
    # C_p = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4
    # H/(RT) = a_0 + 1/2 a_1 T + 1/3 a_2 T^2 + 1/4 a_3 T^3 + 1/5 a_4 T^4 + a_5/T
    # S/R = a_0 ln(T) + a_1 T + 1/2 a_2 T^2 + 1/3 a_3 T^3 + 1/4 a_4 T^4 + a_6
    # At the moment we only parse S and H for a given temperature. Therefore, we only set the
    # coefficients a_5 = H/R and a_6 = S/R. This will only work for T=const in the reactor because in that case
    # C_p does not matter and H and S are valid.
    entry = {
        "name": name,
        "radicalelectrons": 0,
        "thermo": {
            "polys": [{
                "Tmax": 5000.0,
                "Tmin": 1.0,
                "coefs": [0.0, 0.0, 0.0, 0.0, 0.0, h / utils.MOLAR_GAS_CONSTANT, s / utils.MOLAR_GAS_CONSTANT],
                "type": "NASApolynomial"
            }],
            "type": "NASA"
        },
        "type": species_type_str
    }
    return entry


def create_rms_phase_entry(aggregate_str_ids: List[str], enthalpies: List[float], entropies: List[float],
                           solvent_name: Optional[str] = None) -> List[Dict]:
    """
    Create all entries for the species in the RMS input dictionary (this is the 'Phase' dictionary in RMS).

    Parameters
    ----------
    aggregate_str_ids : List[str]
        The aggregate IDs as strings.
    enthalpies : List[float]
        The aggregate enthalpies (same ordering as for the aggregate_str_ids) in J/mol.
    entropies : List[float]
        The aggregate entropies (note the ordering) in J/mol/K.
    solvent_name : Optional[str] (default None)
        The name of an additional solvent species that is added to the species if provided.

    Returns
    -------
    The species a list of one dictionary.
    """
    species_list = []
    for str_id, h, s in zip(aggregate_str_ids, enthalpies, entropies):
        species_list.append(create_single_species_entry(str_id, h, s))
    if solvent_name is not None:
        species_list.append(create_single_species_entry(solvent_name, 0.0, 0.0))
    return [{"Species": species_list, "name": "phase"}]


def create_arrhenius_reaction_entry(reactant_names: List[str], product_names: List[str], e_a: float, n: float, a: float,
                                    type_str: str = "ElementaryReaction") -> Dict[str, Any]:
    """
    Create a reaction entry in the RMS format assuming that the rate constant is given by the Arrhenius equation:
    k = a / T^n exp(-e_a/(k_B T)).

    Parameters
    ----------
    reactant_names : List[str]
        Species names of the reactions LHS (the names must correspond to an entry in the RMS phase dictionary).
    product_names : List[str]
        Species names of the reactions RHS (the names must correspond to an entry in the RMS phase dictionary).
    e_a : float
        Activation energy in J/mol.
    n : float
        Temperature exponent.
    a : float
        Arrhenius prefactor.
    type_str : str (default 'ElementaryReaction')
        Type of the reaction entry (see the RMS documentation for other options).

    Returns
    -------
    Dict[str, Any]
        The reaction entry.
    """
    kinetics_type_str: str = "Arrhenius"
    return {
        "kinetics": {
            "A": a,
            "Ea": float(e_a),
            "n": n,
            "type": kinetics_type_str
        },
        "products": [s for s in product_names],  # copying the list like this avoids id counters in the final yaml file
        "reactants": [s for s in reactant_names],
        "type": type_str
    }


def create_rms_reaction_entry(prefactors: List[float], temperature_exponents: List[float],
                              activation_energies: Union[List[float], np.ndarray],
                              reactant_list: List[Tuple[List[str], List[str]]]) -> List[Dict[str, Any]]:
    """
    Create the reaction entries for the RMS input dictionary assuming Arrhenius kinetics and that all reactions are
    Elementary Reactions (according to the RMS definition):
    k = a / T^n exp(-e_a/(k_B T))

    The parameters are given as lists. The ordering in all lists must be the same.

    Parameters
    ----------
    prefactors : List[float]
        Arrhenius prefactors (a in the equation above).
    temperature_exponents : List[float]
        Temperature exonents (n in the equation above).
    activation_energies : Union[List[float], np.ndarray]
        Activation energies (e_a in the equation above).
    reactant_list : List[Tuple[List[str], List[str]]]
        LHS (tuple[0]) and RHS (tuple[1]) of all reactions.

    Returns
    -------
    List[Dict[str, Any]]
        All reaction entries as a list of dictionaries.
    """
    reaction_type_str = "ElementaryReaction"
    reaction_list = []
    for a, n, e_a, reactants in zip(prefactors, temperature_exponents, activation_energies, reactant_list):
        lhs_str_ids = reactants[0]
        rhs_str_ids = reactants[1]
        reaction_list.append(create_arrhenius_reaction_entry(lhs_str_ids, rhs_str_ids, e_a, n, a, reaction_type_str))
    return reaction_list


def create_rms_units_entry(units: Optional[Dict] = None) -> Dict:
    """
    Create the 'units' entry in the RMS input dictionary. See the RMS documentation for supported units.

    Parameters
    ----------
    units : Optional[Dict] (default None)
        The units as a dictionary.

    Returns
    -------
    Dict
        If no units were provided, an empty dictionary is returned.
    """
    if units is None:
        units = {}
    return units


def create_solvent_entry(solvent: str, solvent_viscosity: Optional[float], solvent_id_str: Optional[str]) -> Dict:
    """
    Create the entry for the solvent.

    Parameters
    ----------
    solvent : str
        The solvent name (e.g., water) to extract tabulated viscosity values.
    solvent_viscosity : Optional[float]
        The solvent's viscosity in Pa s.
    solvent_id_str : Optional[str]
        The string id of the solvent compound. Only required if the solvent is a reacting species.

    Returns
    -------
    Dict
        The solvent entry as a dictionary.
    """
    if solvent_viscosity is None:
        solvent_viscosity = get_default_viscosity(solvent)
    solvent_entry = {
        "mu": {"mu": solvent_viscosity, "type": "ConstantViscosity"},
        "name": solvent if solvent_id_str is None else solvent_id_str,
        "type": "Solvent"
    }
    return solvent_entry


def create_rms_yml_file(aggregate_str_ids: List[str],
                        enthalpies: List[float],
                        entropies: List[float],
                        prefactors: List[float],
                        temperature_exponents: List[float],
                        activation_energies: Union[List[float], np.ndarray],
                        reactants: List[Tuple[List[str], List[str]]],
                        file_name: str, solvent_name: Optional[str] = None, solvent_viscosity: Optional[float] = None,
                        solvent_aggregate_index: Optional[int] = None) -> None:
    """
    Write the yml file input for RMS.

    Parameters
    ----------
    aggregate_str_ids : List[str]
        The list of aggregate string ids to be added as RMS species.
    enthalpies : List[float]
        The list of enthalpies for the aggregates (in J/mol).
    entropies : List[float]
        The list of the aggregates entropies (in J/mol/K).
    prefactors : List[float]
        The list of the Arrhenius prefactors.
    temperature_exponents : List[float]
        The list of the temperature exponents.
    activation_energies : Union[List[float], np.ndarray]
        The activation energies in J/mol.
    reactants : List[Tuple[List[str], List[str]]]
        LHS (tuple[0]) and RHS (tuple[1]) of all reactions.
    file_name : str
        The filename for the yml file.
    solvent_name : Optional[str] (default None)
        The solvent name.
    solvent_viscosity : Optional[float] (default None)
        The solvent's viscosity in Pa s.
    solvent_aggregate_index : Optional[int] (default None)
        The index of the solvent in the aggregte id list. This is only required if the solvent is a reacting species.
    """
    solvent_aggregate_id_str = None
    solvent_in_aggregate_list = False
    if solvent_aggregate_index is not None:
        solvent_aggregate_id_str = aggregate_str_ids[solvent_aggregate_index]
        solvent_in_aggregate_list = True
    phase_entry = create_rms_phase_entry(aggregate_str_ids, enthalpies, entropies,
                                         None if solvent_in_aggregate_list else solvent_name)
    reaction_entry = create_rms_reaction_entry(prefactors, temperature_exponents, activation_energies, reactants)
    unit_entry = create_rms_units_entry()
    input_dictionary = {
        "Phases": phase_entry,
        "Reactions": reaction_entry,
        "Units": unit_entry
    }
    if solvent_name is not None:
        input_dictionary["Solvents"] = [create_solvent_entry(solvent_name, solvent_viscosity, solvent_aggregate_id_str)]
    with open(file_name, "w") as outfile:
        yaml.dump(input_dictionary, outfile, default_flow_style=False)


def resolve_rms_solver(solver_name: str, reactor: Any):
    """
    Resolve the solver for the ODE system by string.

    Parameters
    ----------
    solver_name : str
        The solver name.
    reactor : rms.Reactor
        The julia RMS reactor object.

    Returns
    -------
    Returns the selected ODE solver as a julia Differential Equation object.
    """
    # pylint: disable=import-error
    from diffeqpy import de
    # pylint: enable=import-error
    solvers = {
        "CVODE_BDF": de.CVODE_BDF(max_convergence_failures=60),
        "Rosenbrock23": de.Rosenbrock23(),
        "QNDF": de.QNDF(),
        "TRBDF2": de.TRBDF2(),
        "Recommended": reactor.recommendedsolver
    }
    if solver_name not in solvers:
        raise LookupError("Unknown differential equation solver.")
    return solvers[solver_name]


def resolve_rms_phase(phase_name: str, rms_species: Any, rms_reactions: Any, rms_solvent: Any, diffusion_limited: bool,
                      site_density: Optional[float]) -> Any:
    """
    Resolve the RMS phase model by string.

    Parameters
    ----------
    phase_name : str
        The name of the phase. Options are 'ideal_gas' and 'ideal_dilute_solution'.
    rms_species : RMS species object
        The RMS species.
    rms_reactions : RMS Reaction list
        The RMS reactions.
    rms_solvent : RMS solvent object
        The RMS solvetn object.
    diffusion_limited : bool
        If true, diffusion limitations are enforced.
    site_density : Optional[float]
        The site density for surface reactions.

    Returns
    -------
    Any
        Returns the RMS phase object.
    """
    # pylint: disable=import-error
    from julia import ReactionMechanismSimulator as rms
    # pylint: enable=import-error

    if phase_name == "ideal_gas":
        return rms.IdealGas(rms_species, rms_reactions, name="ideal_gas")
    elif phase_name == "ideal_dilute_solution":
        return get_ideal_dilute_solution(rms_species, rms_reactions, rms_solvent, diffusion_limited)
    elif phase_name == "ideal_surface":
        return get_ideal_surface(rms_species, rms_reactions, diffusion_limited, site_density)
    else:
        raise LookupError("Unknown phase name for the kinetic modeling. Options are:"
                          "ideal_gas, ideal_dilute_solution, and ideal_surface.\n"
                          "You chose: " + phase_name)


def get_ideal_dilute_solution(rms_species: Any, rms_reactions: Any, rms_solvent: Any, diffusion_limited: bool) -> Any:
    """
    Getter for the ideal dilute solution phase object of RMS.

    Parameters
    ----------
    rms_species : RMS species object
        The RMS species.
    rms_reactions : RMS Reaction list
        The RMS reactions.
    rms_solvent : RMS solvent object
        The RMS solvetn object.
    diffusion_limited : bool
        If true, diffusion limitations are enforced.

    Returns
    -------
    Any
        The rms.IdealDiluteSolution object.
    """
    # pylint: disable=import-error
    from julia import ReactionMechanismSimulator as rms
    # pylint: enable=import-error
    assert rms_solvent
    return rms.IdealDiluteSolution(rms_species, rms_reactions, rms_solvent, name="ideal_solution",
                                   diffusionlimited=diffusion_limited)


def get_ideal_surface(rms_species, rms_reactions, diffusion_limited: bool, site_density: Optional[float]):
    """
    Getter for the ideal surface RMS phase object.

    Parameters
    ----------
    rms_species : RMS species object
        The RMS species.
    rms_reactions : RMS Reaction list object
        The RMS reactions.
    diffusion_limited : bool
        If true, diffusion limitations are enforced.
    site_density : float
        The site density for surface reactions.

    Returns
    -------
    The rms.IdealSurface object.
    """
    # pylint: disable=import-error
    from julia import ReactionMechanismSimulator as rms
    # pylint: enable=import-error
    if site_density is None:
        raise NotImplementedError("No site density was provided through the settings. Currently there is no sensible"
                                  " default available.\nPlease add an entry 'site_density: some_number' to your"
                                  " settings.")
    return rms.IdealSurface(rms_species, rms_reactions, site_density, name="ideal_surface",
                            diffusionlimited=diffusion_limited)


def get_default_viscosity(solvent_name: str) -> float:
    """
    Getter for tabulated solvent viscosity (in Pa s). Tabulated values are at 25 celsius.

    Source: https://hbcp.chemnetbase.com/faces/contents/InteractiveTable.xhtml
    Accessed on 23.01.2023, 14:00

    Parameters
    ----------
    solvent_name : str
        The solvent's name.

    Returns
    -------
    float
        The solvent's viscosity (in Pa s).
    """
    viscosities = {
        "water": 0.890,
        "h2o": 0.890,
        "acetone": 0.306,
        "benzene": 0.604,
        "dmso": 1.987,
        "ethanol": 1.074,
        "methanol": 0.544,
        "hexane": 0.240,
        "toluene": 0.560,
        "chloroform": 0.537,
        "nitrobenzene": 1.863,
        "aceticacid": 1.056,
        "acetonitrile": 0.369,
        "aniline": 3.85,
        "benzylalcohol": 5.47,
        "bromoform": 1.857,
        "butanol": 2.54,
        "tertbutanol": 4.31,
        "carbontetrachloride": 0.908,
        "cyclohexane": 0.894,
        "cyclohexanone": 2.02,
        "dichlorobenzene": 1.324,  # (ortho) | 1.044(meta)
        "diethylether": 0.224,
        "dioxane": 1.177,
        "dmfa": 0.794,
        "ethylacetate": 0.423,
        "dichloroethane": 0.779,  # (1, 2) | 0.464(1, 1)
        "ethyleneglycol": 16.06,
        "formicacid": 1.607,
        "isopropanol": 2.04,
        "thf": 0.456,
        "ch2cl2": 0.413
    }
    if solvent_name not in viscosities:
        raise LookupError("There is no viscosity tabulated for the given solvent: " + solvent_name
                          + ". Please provide it manually by settings the corresponding settings value.")
    return viscosities[solvent_name] * 1e-3
