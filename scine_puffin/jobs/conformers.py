# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_connectivity_job import ConnectivityJob
from ..utilities import masm_helper


class Conformers(ConnectivityJob):
    """
    A job generating all possible conformers (guesses) for a given structure
    with Molassembler. Currently, the structure must be a structure representing
    a single compound and not a non-covalently bonded complex.
    The given model is used as a hint for the quality of the bond orders that
    are to be used when generating the ``molassembler`` molecule.

    **Order Name**
      ``conformers``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      add_based_on_distance_connectivity :: bool
        If ``True``, the structure's connectivity is derived from interatomic
        distances via the utils.BondDetector: The bond orders used for
        interpretation are set to the maximum between those given in the
        ``bond_orders`` property and 1.0, whereever the utils.BondDetector
        detects a bond. (default: True)
      sub_based_on_distance_connectivity :: bool
        If ``True``, the structure's connectivity is derived from interatomic
        distances via the utils.BondDetector: The bond orders used for
        interpretation are removed, whereever the utils.BondDetector does not
        detect a bond. (default: True)
      enforce_bond_order_model :: bool
        If ``True``, only processes ``bond_orders`` that were generated with
        the specified model. If ``False``, eventually falls back to any
        ``bond_orders`` available for the structure. (default: True)
      dihedral_retries :: int
        The number of attempts to generate the dihedral decision
        during conformer generation. (default: 100)

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: molassembler (present by default)
      - SCINE: Utils (present by default)

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Structures
        A set of conformers guesses derived from the graph representation of the
        initial structure. All generated conformers will have a graph
        (``masm_cbor_graph``) and decision list set (``masm_decision_list``).
    """

    def __init__(self):
        super().__init__()
        self.name = "Scine conformer generation"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_database as db
        import scine_molassembler as masm
        import math

        structure = db.Structure(calculation.get_structures()[0], self._structures)
        atoms = structure.get_atoms()

        # Generate database results
        db_results = calculation.get_results()
        db_results.clear()

        with calculation_context(self):
            # Get settings and check for incorrect settings
            self.connectivity_settings_from_only_connectivity_settings()
            # Query bond orders of the specified model
            db_bond_orders = self.query_bond_orders(structure)
            # construct bond orders from db property
            bond_orders = self.bond_orders_from_db_bond_orders(structure, db_bond_orders)
            # Load data into molassembler molecule
            results = masm_helper.get_molecules_result(
                atoms,
                bond_orders,
                self.connectivity_settings,
                calculation.get_model().periodic_boundaries,
            )
            if len(results.molecules) > 1:
                self.raise_named_exception("Too many molecules, expected only one.")

            # Check if the structure has a compound linked
            if structure.has_aggregate():
                compound = db.Compound(structure.get_aggregate(), self._compounds)
            else:
                compound = None

            masm.Options.Thermalization.disable()
            alignment = masm.BondStereopermutator.Alignment.BetweenEclipsedAndStaggered
            generator = masm.DirectedConformerGenerator(results.molecules[0], alignment)

            # objects for store function
            result_model = db.Model("guess", "", "")
            result_model.program = "molassembler"
            charge = structure.get_charge()
            multiplicity = structure.get_multiplicity()
            structures = self._structures

            def store(_, conformation):
                """Enumeration callback storing the conformation in the DB"""
                # Update positions
                atoms.positions = conformation
                # New structure
                new_structure = db.Structure()
                new_structure.link(structures)
                new_id = new_structure.create(atoms, charge, multiplicity, result_model, db.Label.MINIMUM_GUESS)
                if structure.has_graph("masm_cbor_graph"):
                    new_structure.set_graph("masm_cbor_graph", structure.get_graph("masm_cbor_graph"))

                relabeler = generator.relabeler()
                # Generate dihedral angles from atom positions
                dihedrals = relabeler.add(atoms.positions)
                structure_bins = []
                symmetries = []
                for j, d in enumerate(dihedrals):
                    symmetries.append(relabeler.sequences[j].symmetry_order)
                    float_bounds = relabeler.make_bounds(d, 5.0 * math.pi / 180)
                    structure_bins.append(relabeler.integer_bounds(float_bounds))

                bin_strs = [masm_helper.make_bin_str(b, d, s) for b, d, s in zip(structure_bins, dihedrals, symmetries)]
                decision_list_str = ":".join(bin_strs)

                new_structure.set_graph("masm_decision_list", decision_list_str)
                new_structure.set_comment("Conformer guess generated from " + str(structure.id()))
                db_results.add_structure(new_id)
                # If the structure has a compound add the new conformer
                if compound is not None:
                    compound.add_structure(new_id)
                    new_structure.set_aggregate(compound.id())

            enumeration_settings = generator.EnumerationSettings()
            # Stopgap until refinement with conflicting dihedral terms is more reliable
            enumeration_settings.dihedral_retries = self.connectivity_settings["dihedral_retries"]
            enumeration_settings.fitting = masm.BondStereopermutator.FittingMode.Nearest
            generator.enumerate(callback=store, seed=42, settings=enumeration_settings)

            calculation.set_results(db_results)
        return self.postprocess_calculation_context()

    @staticmethod
    def required_programs():
        return ["database", "molassembler", "utils"]
