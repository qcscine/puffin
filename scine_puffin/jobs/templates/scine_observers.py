# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING, List

from numpy.random import default_rng

from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class Observer(ABC):
    """
    Abstract base class that defines an observer pattern in a Puffin job
    in order to observe each calculation.
    """

    @abstractmethod
    def gather(self, cycle: int, atoms, results, tag: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self, db_manager: db.Manager, charge: int, multiplicity: int) -> None:
        raise NotImplementedError


class StoreEverythingObserver(Observer):
    """
    Observer implementation that stores every structure
    and calculated properties in the database.
    """

    def __init__(self, calculation_id: db.ID, model: db.Model) -> None:
        super().__init__()
        self.data: List[Dict[str, Any]] = []
        self.white_list = ['energy', 'gradients']
        self.calculation_id = calculation_id
        self.model = model

    def gather(self, cycle: int, atoms: utils.AtomCollection, results: utils.Results, tag: str) -> None:

        tmp = {
            'atoms': atoms,
            'tag': tag,
        }
        for result_str in dir(results):
            if not result_str.startswith('__'):
                if getattr(results, result_str) is not None:
                    tmp[result_str] = getattr(results, result_str)
        self.data.append(tmp)

    @staticmethod
    @requires('database')
    def tag_to_label(tag: str) -> db.Label:
        mapping = {
            'geometry_optimization': db.Label.MINIMUM_GUESS,
            'ts_optimization': db.Label.TS_GUESS,
            'irc_forward': db.Label.ELEMENTARY_STEP_OPTIMIZED,
            'irc_backward': db.Label.ELEMENTARY_STEP_OPTIMIZED,
            'afir_scan': db.Label.REACTIVE_COMPLEX_SCANNED,
            'nt1_scan': db.Label.REACTIVE_COMPLEX_SCANNED,
            'nt2_scan': db.Label.REACTIVE_COMPLEX_SCANNED,
        }
        return mapping[tag]

    @requires('database')
    def finalize(self, db_manager: db.Manager, charge: int, multiplicity: int) -> None:
        has_white_list: bool = (len(self.white_list) > 0)
        structures = db_manager.get_collection('structures')
        properties = db_manager.get_collection('properties')

        for result in self.data:
            structure = db.Structure.make(result['atoms'], charge, multiplicity, structures)
            label = StoreEverythingObserver.tag_to_label(result['tag'])
            structure.set_label(label)
            for property_name in result:
                if property_name in ['atoms', 'tag', 'successful_calculation']:
                    continue
                if (has_white_list and property_name in self.white_list) or not has_white_list:
                    db_name = property_name
                    if property_name == 'energy':
                        db_name = 'electronic_energy'
                        new_prop: db.Property = db.NumberProperty.make(db_name, self.model, result[property_name],
                                                                       properties)

                    elif property_name == 'gradients':
                        new_prop = db.DenseMatrixProperty.make(
                            db_name, self.model, result[property_name], properties)
                    else:
                        continue
                    new_prop.set_structure(structure.get_id())
                    new_prop.set_calculation(self.calculation_id)
                    structure.add_property(db_name, new_prop.get_id())


class StoreWithFrequencyObserver(StoreEverythingObserver):
    """
    Observer implementation that stores every nth structure
    and calculated properties in the database.
    """

    def __init__(self, calculation_id: db.ID, model: db.Model, frequency: float) -> None:
        super().__init__(calculation_id, model)
        self.frequency = frequency

    def gather(self, cycle: int, atoms: utils.AtomCollection, results: utils.Results, tag: str) -> None:
        if self.frequency == 0:
            return
        if cycle % self.frequency == 0:
            super().gather(cycle, atoms, results, tag)


class StoreWithFractionObserver(StoreEverythingObserver):
    """
    Observer implementation that stores a given fraction of structures
    and their properties in the database.
    Which structures are stored is determined at random.
    """

    def __init__(self, calculation_id: db.ID, model: db.Model, fraction: float) -> None:
        super().__init__(calculation_id, model)
        self.fraction = fraction
        self.rng = default_rng()

    def gather(self, cycle: int, atoms: utils.AtomCollection, results: utils.Results, tag: str) -> None:
        if self.rng.random() < self.fraction:
            super().gather(cycle, atoms, results, tag)
