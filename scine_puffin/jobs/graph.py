# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_connectivity_job import ConnectivityJob


class Graph(ConnectivityJob):
    """
    A job generating the molassembler graph and decision lists of a structure.

    **Order Name**
      ``graph``

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
        the specified model. If ``False``, eventually falls back to any ``bond_orders``
        available for the structure. (default: True)

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: molassembler (present by default)
      - SCINE: Utils (present by default)

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        - None

      Other
        Graph representations of the structure will be added to the structures
        ``graphs`` field. The added representations are: A representation of the
        graph ``masm_cbor_graph``, and the decision representations of the existing
        stereopermutators using a nearest neighour fit ``masm_decision_list``
        Any previous graph representations of the structure will be overwritten.
    """

    def __init__(self):
        super().__init__()
        self.name = "Scine Graph Job"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_database as db

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)

        with calculation_context(self):
            # Get settings and check for incorrect settings
            self.connectivity_settings_from_only_connectivity_settings()
            # Query bond orders of the specified model
            db_bond_orders = self.query_bond_orders(structure)
            # construct bond orders from db property
            bond_orders = self.bond_orders_from_db_bond_orders(structure, db_bond_orders)
            # construct graph from bond orders
            self.add_graph(structure, bond_orders)

            # After the calculation, verify that the connection to the database still exists
            self.verify_connection()

            calculation.set_model(
                db_bond_orders.get_model()
            )  # updates model based on model of bond order job
        return self.postprocess_calculation_context()

    @staticmethod
    def required_programs():
        return ["database", "molassembler", "utils"]
