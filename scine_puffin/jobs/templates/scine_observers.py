# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC, abstractmethod


class Observer(ABC):

    @abstractmethod
    def gather(self, cycle: int, atoms, results, tag: str):
        raise NotImplementedError

    @abstractmethod
    def finalize(self, db_manager, charge: int, multiplicity: int):
        raise NotImplementedError


class StoreEverythingObserver(Observer):
    def __init__(self, calculation_id, model):
        self.data = []
        self.white_list = ['energy', 'gradients']
        self.calculation_id = calculation_id
        self.model = model

    def gather(self, _: int, atoms, results, tag: str):

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
    def tag_to_label(tag: str):
        import scine_database as db
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

    def finalize(self, db_manager, charge: int, multiplicity: int):
        import scine_database as db
        has_white_list: bool = (len(self.white_list) > 0)
        structures = db_manager.get_collection('structures')
        properties = db_manager.get_collection('properties')

        for result in self.data:
            structure = db.Structure.make(result['atoms'], charge, multiplicity, structures)
            label = StoreEverythingObserver.tag_to_label(result['tag'])
            structure.set_label(label)
            for property_name in result:
                if property_name in ['atoms', 'tag', 'successfull_calculation']:
                    continue
                if (has_white_list and property_name in self.white_list) or not has_white_list:
                    db_name = property_name
                    if property_name == 'energy':
                        db_name = 'electronic_energy'
                        new_prop = db.NumberProperty.make(db_name, self.model, result[property_name], properties)

                    elif property_name == 'gradients':
                        new_prop = db.DenseMatrixProperty.make(
                            db_name, self.model, result[property_name], properties)
                    else:
                        continue
                    new_prop.set_structure(structure.get_id())
                    new_prop.set_calculation(self.calculation_id)
                    structure.add_property(db_name, new_prop.get_id())
