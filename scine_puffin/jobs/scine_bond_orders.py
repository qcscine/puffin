# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_connectivity_job import ConnectivityJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class ScineBondOrders(ConnectivityJob):
    __doc__ = ("""
    A job calculating bond orders.

    **Order Name**
      ``scine_bond_orders``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The ``bond_orders`` (``SparseMatrixProperty``) are added.
        Optionally the ``electronic_energy`` associated with the structure if it
        is present in the results of provided by the calculator interface.
      Other
        If a graph is requested, graph representations of the structure will be
        added to the structures ``graphs`` field. The added representations are:
        A representation of the graph ``masm_cbor_graph``, and the decision
        representations of the existing stereopermutators using a nearest
        neighbour fit ``masm_decision_list``.
        Any previous graph representations of the structure will be overwritten.
    """ + "\n"
               + ConnectivityJob.optional_settings_doc() + "\n"
               + ConnectivityJob.general_calculator_settings_docstring() + "\n"
               + ConnectivityJob.generated_data_docstring() + "\n"
               + """
    If successful the following data will be generated and added to the
    database:

      Properties
        The ``bond_orders`` (``SparseMatrixProperty``) are added.
        Optionally the ``electronic_energy`` associated with the structure if it
        is present in the results of provided by the calculator interface.
      Other
        If a graph is requested, graph representations of the structure will be
        added to the structures ``graphs`` field. The added representations are:
        A representation of the graph ``masm_cbor_graph``, and the decision
        representations of the existing stereopermutators using a nearest
        neighbour fit ``masm_decision_list``.
        Any previous graph representations of the structure will be overwritten.
    """
               + ConnectivityJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Bond Order Job"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_readuct as readuct

        # Get structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)
        settings_manager.separate_settings(calculation.get_settings())

        # Get connectivity settings and remove them from settings passed to readuct
        self.extract_connectivity_settings_from_dict(settings_manager.task_settings)
        add_graph = settings_manager.task_settings.pop("add_graph", True)

        # actual calculation
        success = False  # success might not be set if something throws in context -> ensure it exists in scope
        with calculation_context(self):
            # Distance based bond orders
            if self.connectivity_settings["only_distance_connectivity"]:
                bond_orders = self.distance_bond_orders(structure)
            # Bond order calculation with readuct
            else:
                task_settings = utils.ValueCollection(settings_manager.task_settings)
                systems, keys = settings_manager.prepare_readuct_task(
                    structure,
                    calculation,
                    task_settings,
                    config["resources"],
                )
                if program_helper is not None:
                    program_helper.calculation_preprocessing(
                        self.get_calc(keys[0], systems), task_settings
                    )
                systems, success = readuct.run_single_point_task(
                    systems, keys, require_bond_orders=True, **task_settings
                )

                self.throw_if_not_successful(success, systems, keys)
                bond_orders = self.get_calc(keys[0], systems).get_results().bond_orders  # type: ignore
                if bond_orders is None:
                    self.raise_named_exception("No bond orders found in results.")
                    raise RuntimeError("Unreachable")  # for mypy

            # Graph generation
            if add_graph:
                self.add_graph(structure, bond_orders)

            # There are no results if the bond orders are purely distance based
            if self.connectivity_settings["only_distance_connectivity"]:
                self.verify_connection()
                self.capture_raw_output()
            else:
                self.sp_postprocessing(
                    success, systems, keys, structure, program_helper
                )

            # Store bond orders
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                bond_orders.matrix,
                calculation.get_model(),
                calculation,
                structure,
            )

        return self.postprocess_calculation_context()
