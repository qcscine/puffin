# -*- coding: utf-8 -*-
"""masm_helper.py: Collection of common procedures to be carried out with molassembler"""
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING
import math
import sys

from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_molassembler") or TYPE_CHECKING:
    import scine_molassembler as masm
else:
    masm = MissingDependency("scine_molassembler")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


def get_molecules_result(
    atoms: utils.AtomCollection,
    bond_orders: utils.BondOrderCollection,
    connectivity_settings: Dict[str, Union[bool, int]],
    pbc_string: str = "",
    unimportant_atoms: Optional[Union[List[int], Set[int]]] = None,
    modifications: Optional[List[Tuple[int, int, float]]] = None,
) -> masm.interpret.MoleculesResult:
    """
    Generates the molassembler molecules interpretation of an atom
    collection and a bond order collection optionally made subject
    to distance based adaptations.

    Parameters
    ----------
    atoms : utils.AtomCollection
        The atom collection to be interpreted.
    bond_orders : utils.BondOrderCollection
        The bond order collection to be interpreted.
    connectivity_settings : Dict[str, Union[bool, int]]
        Settings describing whether to use the connectivity as predicted based on inter-atomic distances
        by the utils.BondDetector.
    pbc_string : str
        The string specifying periodic boundaries, empty string represents no periodic boundaries.
    unimportant_atoms : Optional[Union[List[int], Set[int]]]
        The indices of atoms for which no stereopermutators shall be determined.
    modifications : Optional[List[Tuple[int, int, float]]]
        Manual bond modifications. They are specified as a list with each element containing the
        indices and the new bond order between those two indices

    Returns
    -------
    masm_results : masm.interpret.MoleculesResult
        The result of the molassembler interpretation.

    """
    ignored_atoms = set() if unimportant_atoms is None else set(unimportant_atoms)

    consider_distance_connectivity = bool(
        connectivity_settings["sub_based_on_distance_connectivity"]
        or connectivity_settings["add_based_on_distance_connectivity"]
    )
    if pbc_string and pbc_string.lower() != "none":
        ps = utils.PeriodicSystem(utils.PeriodicBoundaries(pbc_string), atoms, ignored_atoms)
        if consider_distance_connectivity:
            bo = ps.construct_bond_orders()
            bo.set_to_absolute_values()
            new_bos = _modify_based_on_distances(atoms, bo, bond_orders, connectivity_settings)
            if modifications is not None:
                for mod in modifications:
                    new_bos.set_order(*mod)
            ps.make_bond_orders_across_boundaries_negative(bond_orders)
            data = ps.get_data_for_molassembler_interpretation(new_bos)
        else:
            if modifications is not None:
                for mod in modifications:
                    bond_orders.set_order(*mod)
            ps.make_bond_orders_across_boundaries_negative(bond_orders)
            data = ps.get_data_for_molassembler_interpretation(bond_orders)
        return masm.interpret.molecules(*data, masm.interpret.BondDiscretization.Binary)

    if consider_distance_connectivity:
        distance_bos = utils.SolidStateBondDetector.detect_bonds(atoms, ignored_atoms)
        connectivity_bo_collection = _modify_based_on_distances(
            atoms, distance_bos, bond_orders, connectivity_settings)
    else:
        connectivity_bo_collection = bond_orders

    if modifications is not None:
        for mod in modifications:
            connectivity_bo_collection.set_order(*mod)

    return masm.interpret.molecules(
        atoms,
        connectivity_bo_collection,
        ignored_atoms,
        {},
        masm.interpret.BondDiscretization.Binary,
    )


def _modify_based_on_distances(
    atoms: utils.AtomCollection,
    distance_bos: utils.BondOrderCollection,
    to_add_bos: utils.BondOrderCollection,
    settings: Dict[str, Union[bool, int]],
):
    new_matrix = to_add_bos.matrix
    if settings["add_based_on_distance_connectivity"]:
        # Use maximum of distance and Mayer bond orders
        new_matrix = (to_add_bos.matrix).maximum(distance_bos.matrix)
    if settings["sub_based_on_distance_connectivity"]:
        # Remove bonds that do not exist based on distances
        new_matrix = (new_matrix).multiply(distance_bos.matrix)
    new_bos = utils.BondOrderCollection(len(atoms))
    new_bos.matrix = new_matrix
    return new_bos


def get_cbor_graph_from_molecule(molecule: masm.Molecule):
    """
    Generates the canonical CBOR graph from a single masm.Molecule

    Parameters
    ----------
    molecule : masm.Molecule
        The molecule of which a graph is to be generated

    Returns
    -------
    cbor_string:: str
        The cbor graph string.
    """

    canonical = deepcopy(molecule)
    canonical.canonicalize(masm.AtomEnvironmentComponents.All)
    serialization = masm.JsonSerialization(canonical)
    try:
        serialization.to_molecule()
    except BaseException:
        sys.stderr.write("Non-canonical molecule serialization. Saving non-canonical\n")
        serialization = masm.JsonSerialization(molecule)

    binary = serialization.to_binary(masm.JsonSerialization.BinaryFormat.CBOR)
    cbor_string = masm.JsonSerialization.base_64_encode(binary)

    return cbor_string


def get_cbor_graph(
    atoms: utils.AtomCollection,
    bond_orders: utils.BondOrderCollection,
    connectivity_settings: Dict[str, Union[bool, int]],
    pbc_string: str = "",
    unimportant_atoms: Optional[Union[List[int], Set[int]]] = None,
) -> str:
    """
    Generates the CBOR graph of an atom collection and bond order collection.
    Multiple graphs are concatenated with ";" as a deliminator.

    Parameters
    ----------
    atoms : utils.AtomCollection
        The atom collection to be interpreted.
    bond_orders : utils.BondOrderCollection
        The bond order collection to be interpreted.
    connectivity_settings : Dict[str, Union[bool, int]]
        Settings describing whether to use the connectivity as predicted based on inter-atomic distances
        by the utils.BondDetector.
    pbc_string : str
        The string specifying periodic boundaries, empty string represents no periodic boundaries.
    unimportant_atoms : Optional[Union[List[int], Set[int]]]
        The indices of atoms for which no stereopermutators shall be determined.

    Returns
    -------
    masm_cbor_graphs : str
        The cbor graphs as sorted strings separated by semicolons.
    """

    masm_results = get_molecules_result(atoms, bond_orders, connectivity_settings, pbc_string, unimportant_atoms)
    graph = []
    for m in masm_results.molecules:
        string = get_cbor_graph_from_molecule(m)
        graph.append(string)
    graph.sort()
    return ";".join(graph)


def make_bin_str(int_bounds: Tuple[int, int], dihedral: float, sym: int) -> str:
    """Turn integer bounds and a floating dihedral into a joint string"""
    int_dihedral = int(0.5 + 180 * dihedral / math.pi)
    return "({}, {}, {}, {})".format(int_bounds[0], int_dihedral, int_bounds[1], sym)


def get_decision_list_from_molecule(molecule: masm.Molecule, atoms: utils.AtomCollection) -> str:
    """
    Generates the dihedral decision list for rotatable bonds in a single molecule.

    Parameters
    ----------
    molecule : masm.Molecule
        The molecule.
    atoms : utils.AtomCollection
        The atoms and their positions in the molecule.

    Returns
    -------
    masm_decision_list : str
        The dihedral decision list for rotatable bonds.
    """

    # Infer decision list from positions and store it
    alignment = masm.BondStereopermutator.Alignment.EclipsedAndStaggered
    generator = masm.DirectedConformerGenerator(molecule, alignment)
    relabeler = generator.relabeler()
    # Reorder the molecule-specific atom collection according to the
    # canonicalization reordering
    ordering = molecule.canonicalize(masm.AtomEnvironmentComponents.All)
    ordered_atoms = molecule.apply_canonicalization_map(ordering, atoms)
    positions = ordered_atoms.positions
    dihedrals = relabeler.add(positions)
    structure_bins = []
    symmetries = []
    for j, d in enumerate(dihedrals):
        symmetries.append(relabeler.sequences[j].symmetry_order)
        float_bounds = relabeler.make_bounds(d, 5.0 * math.pi / 180)
        structure_bins.append(relabeler.integer_bounds(float_bounds))

    bin_strs = [make_bin_str(b, d, s) for b, d, s in zip(structure_bins, dihedrals, symmetries)]
    return ":".join(bin_strs)


def get_decision_lists(
    atoms: utils.AtomCollection,
    bond_orders: utils.BondOrderCollection,
    connectivity_settings: Dict[str, Union[bool, int]],
    pbc_string: str = "",
    unimportant_atoms: Optional[Union[Set[int], List[int]]] = None,
) -> List[str]:
    """
    Generates the dihedral decision lists for rotatable bonds in a given system.

    Parameters
    ----------
    atoms : utils.AtomCollection
        The atom collection to be interpreted.
    bond_orders : utils.BondOrderCollection
        The bond order collection to be interpreted.
    connectivity_settings : Dict[str, Union[bool, int]]
        Settings describing whether to use the connectivity as predicted based on inter-atomic distances
        by the utils.BondDetector.
    pbc_string : str
        The string specifying periodic boundaries, empty string represents no periodic boundaries.
    unimportant_atoms : Union[List[int], None]
        The indices of atoms for which no stereopermutators shall be determined.

    Returns
    -------
    masm_decision_lists : List[str]
        The dihedral decision lists for rotatable bonds in all molecules in the given input.
    """
    masm_results = get_molecules_result(atoms, bond_orders, connectivity_settings, pbc_string, unimportant_atoms)
    # Split the atom collection into separate collections for each molecule
    positions = masm_results.component_map.apply(atoms)
    lists = []
    for m, p in zip(masm_results.molecules, positions):
        string = get_decision_list_from_molecule(m, p)
        lists.append(string)
    return lists


def add_masm_info(
    structure: db.Structure,
    bo_collection: utils.BondOrderCollection,
    connectivity_settings: Dict[str, Union[bool, int]],
    unimportant_atoms: Union[List[int], Set[int], None] = None,
) -> None:
    """
    Generates a structure's CBOR graph and decision lists and adds them to the
    database.

    Parameters
    ----------
    structure : db.Structure
        The structure to be analyzed. It has to be linked to a database.
    bo_collection : utils.BondOrderCollection
        The bond order collection to be interpreted.
    connectivity_settings : Dict[str, Union[bool, int]]
        Settings describing whether to use the connectivity as predicted based on inter-atomic distances
        by the utils.BondDetector.
    unimportant_atoms : Union[List[int], None]
        The indices of atoms for which no stereopermutators shall be determined.
    """
    pbc_string = structure.get_model().periodic_boundaries
    atoms = structure.get_atoms()
    try:
        masm_results = get_molecules_result(atoms, bo_collection, connectivity_settings, pbc_string, unimportant_atoms)
    except BaseException as e:
        if structure.get_label() in [db.Label.TS_OPTIMIZED, db.Label.TS_GUESS]:
            print("Molassembler could not generate a graph for TS as it is designed for Minima")
            return
        raise e
    # Split the atom collection into separate collections for each molecule
    positions = masm_results.component_map.apply(atoms)

    properties: List[Dict[str, Any]] = [{"component": i} for i in range(len(masm_results.molecules))]
    atom_map: List[masm.interpret.ComponentMap.ComponentIndexPair] = [masm_results.component_map.apply(i)
                                                                      for i in range(len(atoms))]

    for i, m in enumerate(masm_results.molecules):
        ordering = m.canonicalize(masm.AtomEnvironmentComponents.All)
        # Reorder the molecule-specific atom collection according to the
        # canonicalization reordering
        positions[i] = m.apply_canonicalization_map(ordering, positions[i])
        # Apply the canonicalization reordering to the atom mapping
        atom_map = [(c, ordering[j]) if c == i else (c, j) for c, j in atom_map]  # type: ignore
        # Generate a canonical b64-encoded cbor serialization of the molecule
        serialization = masm.JsonSerialization(m)
        binary = serialization.to_binary(masm.JsonSerialization.BinaryFormat.CBOR)
        b64_str = masm.JsonSerialization.base_64_encode(binary)
        properties[i]["serialization"] = b64_str

    # Infer decision list from positions and store it
    for i, (m, p) in enumerate(zip(masm_results.molecules, positions)):
        alignment = masm.BondStereopermutator.Alignment.EclipsedAndStaggered
        generator = masm.DirectedConformerGenerator(m, alignment)
        relabeler = generator.relabeler()
        dihedrals = relabeler.add(p.positions)
        structure_bins = []
        symmetries = []
        for j, d in enumerate(dihedrals):
            symmetries.append(relabeler.sequences[j].symmetry_order)
            float_bounds = relabeler.make_bounds(d, 5.0 * math.pi / 180)
            structure_bins.append(relabeler.integer_bounds(float_bounds))

        bin_strs = [make_bin_str(b, d, s) for b, d, s in zip(structure_bins, dihedrals, symmetries)]
        properties[i]["decisions_nearest"] = ":".join(bin_strs)

    # Order by the graph serialization to ensure when multiple molecules are
    # present, the composed string is atom order independent
    properties.sort(key=lambda x: x["serialization"])

    decisions_nearest = ";".join(x["decisions_nearest"] for x in properties)
    structure.set_graph("masm_decision_list", decisions_nearest)

    graphs = ";".join(x["serialization"] for x in properties)
    structure.set_graph("masm_cbor_graph", graphs)

    # Apply new ordering to the atom map and store it, too
    new_component_order = [x["component"] for x in properties]
    atom_map = [(new_component_order.index(c), i) for c, i in atom_map]  # type: ignore
    structure.set_graph("masm_idx_map", str(atom_map))
