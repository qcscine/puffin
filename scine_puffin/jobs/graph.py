# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, List

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_connectivity_job import ConnectivityJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class Graph(ConnectivityJob):
    __doc__ = ("""
    A job generating the molassembler graph and decision lists of a structure.

    **Order Name**
      ``graph``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:
    """ + "\n"
               + ConnectivityJob.optional_settings_doc() + "\n"
               + ConnectivityJob.general_calculator_settings_docstring() + "\n"
               + ConnectivityJob.generated_data_docstring() + "\n"
               + """
    If successful the following data will be generated and added to the
    database:

    Properties
      - None

    Other
      Graph representations of the structure will be added to the structures
      ``graphs`` field. The added representations are: A representation of the
      graph ``masm_cbor_graph``, and the decision representations of the existing
      stereopermutators using a nearest neighbor fit ``masm_decision_list``
      Any previous graph representations of the structure will be overwritten.
    """
               + ConnectivityJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Graph Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

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
    def required_programs() -> List[str]:
        return ["database", "molassembler", "utils"]
