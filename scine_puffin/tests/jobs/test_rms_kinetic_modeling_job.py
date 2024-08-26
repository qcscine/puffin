#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union

from ..testcases import (
    JobTestCase,
    skip_without
)
from ..db_setup import (
    add_calculation,
    add_compound_and_structure,
    add_reaction,
    add_flask_and_structure
)


class RMSKineticModelingTest(JobTestCase):

    @skip_without('database', "julia", "diffeqpy")
    def test_concentrations_ideal_gas(self):
        from scine_puffin.jobs.rms_kinetic_modeling import RmsKineticModeling
        import scine_database as db

        n_compounds = 5
        all_compounds = [add_compound_and_structure(self.manager) for _ in range(n_compounds - 1)]
        flask = add_flask_and_structure(self.manager)
        all_compounds.append(flask)
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        c_ids = [c.id() for c in all_compounds]
        all_reaction_ids = [
            add_reaction(self.manager, [c_ids[0]], [c_ids[2]]).id(),
            add_reaction(self.manager, [c_ids[1]], [c_ids[3]]).id(),
            add_reaction(self.manager, [c_ids[2], c_ids[3]], [c_ids[0], c_ids[1]]).id(),
            add_reaction(self.manager, [c_ids[0], c_ids[1]], [c_ids[4], c_ids[4]]).id()
        ]
        ea = [0.00, 0.00, 0.00, 0.00]
        a = [0.1, 0.05, 0.02, 0.02]
        entropies = [0.5, 0.5, 0.5, 0.5, 0.5]
        enthalpies = [-2e+2, -1.98e+2, -1.97e+2, -1.99e+2, -2.1e+2]
        n = [0, 0, 0, 0]
        start_concentrations = [0.5, 0.4, 0.0, 0.0, 0.0]
        reference_data = [0.20530677, 0.15515132, 0.20505846, 0.15521392, 0.17926953]
        reference_max = [0.5, 0.4, 0.20505868, 0.15745688, 0.19571166]
        reference_flux = [0.31232202, 0.26808923, 0.20617824, 0.16194545, 0.21228757]

        model = db.Model('FAKE', '', '')
        job = db.Job('rms_kinetic_modeling')
        settings = {
            "solver": "CVODE_BDF",
            "ea": ea,
            "arrhenius_prefactors": a,
            "arrhenius_temperature_exponents": n,
            "start_concentrations": start_concentrations,
            "reaction_ids": [str(oid) for oid in all_reaction_ids],
            "aggregate_ids": [str(oid) for oid in c_ids],
            "aggregate_types": [db.CompoundOrFlask.COMPOUND for _ in range(4)] + [db.CompoundOrFlask.FLASK],
            "entropies": entropies,
            "enthalpies": enthalpies,
            "energy_model_program": "DUMMY",
            "phase_type": "ideal_gas",
            "max_time": 36000.0,
            "absolute_tolerance": 1e-20,
            "relative_tolerance": 1e-6,
            "reactor_pressure": 1E+5,
        }
        calculation = add_calculation(self.manager, model, job, all_structure_ids, settings)
        # Run calculation/job
        config = self.get_configuration()
        job = RmsKineticModeling()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        for c in all_compounds:
            assert not c.explore()
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        for i, s_id in enumerate(all_structure_ids):
            structure = db.Structure(s_id, structures)
            assert structure.has_property("final_concentration")
            assert structure.has_property("max_concentration")
            assert structure.has_property("concentration_flux")
            final_concentration = db.NumberProperty(structure.get_properties("final_concentration")[0], properties)
            max_concentration = db.NumberProperty(structure.get_properties("max_concentration")[0], properties)
            concentration_flux = db.NumberProperty(structure.get_properties("concentration_flux")[0], properties)
            self.assertAlmostEqual(final_concentration.get_data(), reference_data[i], delta=1e-2)
            self.assertAlmostEqual(max_concentration.get_data(), reference_max[i], delta=1e-2)
            self.assertAlmostEqual(concentration_flux.get_data(), reference_flux[i], delta=1e-1)

    @skip_without('database', "julia", "diffeqpy")
    def test_concentrations_ideal_dilute_solution(self):
        from scine_puffin.jobs.rms_kinetic_modeling import RmsKineticModeling
        import scine_database as db

        n_compounds = 5
        all_compounds = [add_compound_and_structure(self.manager) for _ in range(n_compounds - 1)]
        flask = add_flask_and_structure(self.manager)
        all_compounds.append(flask)
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        c_ids = [c.id() for c in all_compounds]
        all_reaction_ids = [
            add_reaction(self.manager, [c_ids[0]], [c_ids[2]]).id(),
            add_reaction(self.manager, [c_ids[1]], [c_ids[3]]).id(),
            add_reaction(self.manager, [c_ids[2], c_ids[3]], [c_ids[0], c_ids[1]]).id(),
            add_reaction(self.manager, [c_ids[0], c_ids[1]], [c_ids[4], c_ids[4]]).id()
        ]
        ea = [0.00, 0.00, 0.00, 3.00]
        a = [1.1e+3, 1.05e+3, 1.02e+4, 1.02]
        entropies = [0.5, 0.5, 0.5, 0.5, 0.5]
        enthalpies = [-2e+2, -1.98e+2, -1.97e+2, -1.99e+2, -2.1e+2]
        n = [0, 0, 0, 0]
        start_concentrations = [0.5, 0.4, 0.0, 0.0, 0.0]
        reference_data = [0.20531956, 0.15521179, 0.20514741, 0.15525519, 0.17906605, 14.3]
        reference_max = [0.5, 0.4, 0.22863946, 0.22209669, 0.17906605, 14.3]
        reference_flux = [0.38552628, 0.37854594, 0.29599325, 0.28901291, 0.17906606]

        model = db.Model('FAKE', '', '')
        t = 430.15
        model.temperature = t
        model.solvent = "water"
        job = db.Job('rms_kinetic_modeling')
        settings = {
            "solver": "CVODE_BDF",
            "ea": ea,
            "arrhenius_prefactors": a,
            "arrhenius_temperature_exponents": n,
            "start_concentrations": start_concentrations,
            "reaction_ids": [str(oid) for oid in all_reaction_ids],
            "aggregate_ids": [str(oid) for oid in c_ids],
            "aggregate_types": [db.CompoundOrFlask.COMPOUND for _ in range(4)] + [db.CompoundOrFlask.FLASK],
            "entropies": entropies,
            "enthalpies": enthalpies,
            "energy_model_program": "DUMMY",
            "phase_type": "ideal_dilute_solution",
            "max_time": 36000.0,
            "absolute_tolerance": 1e-20,
            "relative_tolerance": 1e-9,
            "reactor_pressure": 1E+5,
            "reactor_solvent": "water",
            "diffusion_limited": False,
        }
        calculation = add_calculation(self.manager, model, job, all_structure_ids, settings)
        # Run calculation/job
        config = self.get_configuration()
        job = RmsKineticModeling()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        for c in all_compounds:
            assert not c.explore()
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        for prop in properties.iterate_all_properties():
            prop.link(properties)
            assert abs(float(prop.get_model().temperature) - t) < 1e-9

        for i, s_id in enumerate(all_structure_ids):
            structure = db.Structure(s_id, structures)
            assert structure.has_property("final_concentration")
            assert structure.has_property("max_concentration")
            assert structure.has_property("concentration_flux")
            final_concentration = db.NumberProperty(structure.get_properties("final_concentration")[0], properties)
            max_concentration = db.NumberProperty(structure.get_properties("max_concentration")[0], properties)
            concentration_flux = db.NumberProperty(structure.get_properties("concentration_flux")[0], properties)
            self.assertAlmostEqual(final_concentration.get_data(), reference_data[i], delta=1e-2)
            self.assertAlmostEqual(max_concentration.get_data(), reference_max[i], delta=1e-2)
            self.assertAlmostEqual(concentration_flux.get_data(), reference_flux[i], delta=1e-1)

    @skip_without('database', "julia", "diffeqpy")
    def test_sensitivity_analysis(self):
        from scine_puffin.jobs.rms_kinetic_modeling import RmsKineticModeling
        import scine_database as db

        n_compounds = 15
        all_compounds = [add_compound_and_structure(self.manager) for _ in range(n_compounds)]
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        c_ids = [c.id() for c in all_compounds]
        all_reaction_ids = [
            add_reaction(self.manager, [c_ids[8]], [c_ids[11], c_ids[12]]).id(),
            add_reaction(self.manager, [c_ids[5]], [c_ids[8]]).id(),
            add_reaction(self.manager, [c_ids[1], c_ids[1]], [c_ids[5]]).id(),
            add_reaction(self.manager, [c_ids[9]], [c_ids[13], c_ids[14], c_ids[1]]).id(),
            add_reaction(self.manager, [c_ids[4]], [c_ids[2]]).id(),
            add_reaction(self.manager, [c_ids[4]], [c_ids[9]]).id(),
            add_reaction(self.manager, [c_ids[0], c_ids[1]], [c_ids[2]]).id(),
            add_reaction(self.manager, [c_ids[0], c_ids[1]], [c_ids[4]]).id(),
            add_reaction(self.manager, [c_ids[5]], [c_ids[7]]).id(),
            add_reaction(self.manager, [c_ids[6]], [c_ids[10], c_ids[1]]).id(),
            add_reaction(self.manager, [c_ids[5]], [c_ids[6]]).id(),
            add_reaction(self.manager, [c_ids[2]], [c_ids[9]]).id(),
            add_reaction(self.manager, [c_ids[4]], [c_ids[3], c_ids[1]]).id(),
            add_reaction(self.manager, [c_ids[3], c_ids[1]], [c_ids[2]]).id()
        ]
        ea = [0.0,
              141258.005874334,
              22350.1045931669,
              0.0,
              37769.2845750187,
              87707.0192024107,
              24118.3178788118,
              23942.6512074803,
              149593.511487877,
              0.0,
              191188.014352586,
              132868.656300405,
              0.0,
              32749.0495510235]
        a = [8817012463061.74 for _ in all_reaction_ids]
        n = [0 for _ in all_reaction_ids]
        entropies = [483.928799667987,
                     294.419261898161,
                     652.415361315592,
                     477.014094417863,
                     675.345450235711,
                     515.52359453181,
                     422.822929205339,
                     417.145780635592,
                     507.650951023459,
                     718.709871540267,
                     280.378644164628,
                     305.918813708162,
                     277.578619144421,
                     225.609287171713,
                     404.146435972299]
        enthalpies = [-1161186535.6783,
                      -506407081.521706,
                      -1667622787.30424,
                      -1161198092.3675,
                      -1667613260.10378,
                      -1012822836.15114,
                      -1012816820.21241,
                      -1012951947.21609,
                      -1012883102.19312,
                      -1667578869.90783,
                      -506390045.122877,
                      -509583214.885545,
                      -503295550.040687,
                      -303469253.644786,
                      -857659812.764682]
        start_concentrations = [1.0, 1.0] + [0.0 for _ in range(len(all_compounds) - 2)]

        # reference from working run.
        reference_final = [7.48691760e-02, 8.98107191e-01, 2.21462075e-03, 8.25022273e-01, 2.43288613e-03,
                           4.82799549e-02, 2.35394815e-14, 2.90416198e-05, 1.17193433e-09, 2.98648216e-05,
                           2.58125417e-10, 2.98720961e-04, 2.98720961e-04, 9.54311790e-02, 9.54311790e-02]
        reference_max = [1.00000000e+00, 1.00000000e+00, 1.37526691e-02, 9.12090645e-01, 2.78100405e-02,
                         4.83372810e-02, 2.35394815e-14, 2.90416198e-05, 1.17193433e-09, 2.98823158e-05,
                         2.58125417e-10, 2.98720961e-04, 2.98720961e-04, 9.54311790e-02, 9.54311790e-02]
        # reference_flux = [2.29282693e+02, 5.42155028e+02, 2.29083511e+02, 2.32554989e+02,
        #                   2.32849987e+02, 4.00383137e+01, 8.35968464e-10, 2.90416198e-05,
        #                   5.87767688e-04, 3.36835658e-01, 5.77819508e-10, 2.89045556e-04,
        #                   2.89045556e-04, 2.41374632e-01, 2.41374632e-01]
        reference_flux = [3.02126416e+02, 7.58527988e+02, 2.64552233e+02, 2.33897598e+02,
                          2.71567599e+02, 1.11175726e+02, 1.05605466e-09, 2.90416198e-05,
                          6.02153590e-04, 2.48637533e-01, 7.97905708e-10, 3.03431457e-04,
                          3.03431457e-04, 1.53176486e-01, 1.53176486e-01]

        reference_c_max_sens = [1.83714115e-01, 1.96540904e-01, 3.48053816e-02, 1.92587203e-01, 5.71615968e-02,
                                8.24115883e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.37171546e-04,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.70804545e-02, 8.70804545e-02]
        reference_c_max_ea_sens = [0.00000000e+00, 0.00000000e+00, 1.71595760e-03, 1.37171547e-04, 0.00000000e+00,
                                   1.37171571e-04, 1.13038940e-02, 1.21804830e-02, 0.00000000e+00, 0.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 7.49768373e-04, 1.08392802e-02]
        reference_c_final_ea_sens = [0.00000000e+00, 0.00000000e+00, 1.66605840e-10, 1.66397229e-10, 0.00000000e+00,
                                     8.47292592e-09, 1.66605840e-10, 1.66605840e-10, 0.00000000e+00, 0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00, 1.66466396e-10, 1.72988956e-10]
        reference_c_flux_ea_sens = [0.00000000e+00, 0.00000000e+00, 4.659885053e-06, 8.02638585e-03, 0.00000000e+00,
                                    6.81609605e-03, 1.87939504e-05, 3.30799624e-05, 0.00000000e+00, 0.00000000e+00,
                                    0.00000000e+00, 0.00000000e+00, 4.94748506e-06, 3.84500387e-05]
        reference_var_final = [1.77284235e-03, 6.28070022e-03, 2.45146925e-06, 3.28166800e-03,
                               2.95284267e-06, 1.51282096e-03, 1.91576991e-27, 5.49343670e-10,
                               2.74160649e-17, 4.54564355e-10, 4.33879991e-20, 5.81191146e-08,
                               5.81191146e-08, 8.41080667e-04, 8.41080667e-04]
        reference_var_max = [0.00000000e+00, 0.00000000e+00, 8.90760701e-05, 2.47119289e-03,
                             2.66124536e-04, 1.52723745e-03, 1.91576991e-27, 5.49343670e-10,
                             2.74160649e-17, 4.55322785e-10, 4.33879991e-20, 5.81191146e-08,
                             5.81191146e-08, 8.41080667e-04, 8.41080667e-04]
        reference_var_flux = [2847.6573183771347, 17956.701741841487, 2183.194068349496, 1706.3903416116216,
                              2300.5616220011425, 385.15557455260193, 5.93208300e-18, 5.49343669e-10,
                              6.42188978e-06, 1.75102380e-03, 5.04266808e-18, 5.28459041e-06,
                              5.28459041e-06, 0.0002534526815016325, 0.0002534526815016325]

        model = db.Model('FAKE', '', '')
        t = 430.15
        model.temperature = t
        model.solvent = "water"
        job = db.Job('rms_kinetic_modeling')
        settings = {
            "solver": "CVODE_BDF",
            "ea": ea,
            "arrhenius_prefactors": a,
            "arrhenius_temperature_exponents": n,
            "start_concentrations": start_concentrations,
            "reaction_ids": [str(oid) for oid in all_reaction_ids],
            "aggregate_ids": [str(oid) for oid in c_ids],
            "aggregate_types": [db.CompoundOrFlask.COMPOUND for _ in all_compounds],
            "entropies": entropies,
            "enthalpies": enthalpies,
            "energy_model_program": "DUMMY",
            "phase_type": "ideal_dilute_solution",
            "max_time": 100.0,
            "absolute_tolerance": 1e-20,
            "relative_tolerance": 1e-9,
            "reactor_pressure": 1E+5,
            "reactor_solvent": "water",
            "diffusion_limited": False,
            "sensitivity_analysis": "morris",
            "ea_lower_uncertainty": [1e+4 for _ in all_reaction_ids],
            "ea_upper_uncertainty": [1e+4 for _ in all_reaction_ids],
            "enthalpy_lower_uncertainty": [5e+3 for _ in c_ids],
            "enthalpy_upper_uncertainty": [5e+3 for _ in c_ids],
            "sample_size": 2,
            "local_sensitivities": True,
            "save_oaat_var": True,
            "enforce_mass_balance": False,
            "screen_global_sens_size": 0
        }
        calculation = add_calculation(self.manager, model, job, all_structure_ids, settings)
        # Run calculation/job
        config = self.get_configuration()
        config["resources"]["cores"] = 2
        job = RmsKineticModeling()
        job.force_parallel = True
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        for c in all_compounds:
            assert not c.explore()
        all_structure_ids = [c.get_centroid() for c in all_compounds]
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        reactions = self.manager.get_collection("reactions")
        compounds = self.manager.get_collection("compounds")
        flasks = self.manager.get_collection("flasks")
        for prop in properties.iterate_all_properties():
            prop.link(properties)
            assert abs(float(prop.get_model().temperature) - t) < 1e-9

        for i, s_id in enumerate(all_structure_ids):
            structure = db.Structure(s_id, structures)
            assert structure.has_property("final_concentration")
            assert structure.has_property("max_concentration")
            assert structure.has_property("concentration_flux")
            assert structure.has_property("max_free_energy_sensitivity_oaat_flux")
            assert structure.has_property("max_free_energy_sensitivity_oaat_max")
            assert structure.has_property("var_final_c")
            assert structure.has_property("var_max_c")
            assert structure.has_property("var_flux_c")
            assert structure.has_property("max_free_energy_sensitivity_morris_mu_c_max")
            assert structure.has_property("max_free_energy_sensitivity_morris_mu_star_c_max")
            assert structure.has_property("max_free_energy_sensitivity_morris_sigma_c_max")
            assert structure.has_property("max_free_energy_sensitivity_morris_mu_c_final")
            assert structure.has_property("max_free_energy_sensitivity_morris_mu_star_c_final")
            assert structure.has_property("max_free_energy_sensitivity_morris_sigma_c_final")
            assert structure.has_property("morris_mean_c_max")
            assert structure.has_property("morris_mean_c_final")
            assert structure.has_property("morris_mean_c_flux")
            assert structure.has_property("morris_var_c_max")
            assert structure.has_property("morris_var_c_final")
            assert structure.has_property("morris_var_c_flux")
            final_concentration = db.NumberProperty(structure.get_properties("final_concentration")[0], properties)
            max_concentration = db.NumberProperty(structure.get_properties("max_concentration")[0], properties)
            concentration_flux = db.NumberProperty(structure.get_properties("concentration_flux")[0], properties)
            c_max_sensitivity = db.NumberProperty(structure.get_properties("max_free_energy_sensitivity_oaat_max")[0],
                                                  properties)
            var_final = db.NumberProperty(structure.get_properties("var_final_c")[0], properties)
            var_max = db.NumberProperty(structure.get_properties("var_max_c")[0], properties)
            var_flux = db.NumberProperty(structure.get_properties("var_flux_c")[0], properties)
            self.assertAlmostEqual(final_concentration.get_data(), reference_final[i], delta=1e-2)
            self.assertAlmostEqual(max_concentration.get_data(), reference_max[i], delta=1e-3)
            self.assertAlmostEqual(concentration_flux.get_data(), reference_flux[i],
                                   delta=1e-2 * max(1.0, reference_flux[i]))
            self.assertAlmostEqual(c_max_sensitivity.get_data(), reference_c_max_sens[i], delta=1e-3)
            self.assertAlmostEqual(var_final.get_data(), reference_var_final[i], delta=1e-4)
            self.assertAlmostEqual(var_max.get_data(), reference_var_max[i], delta=1e-4)
            self.assertAlmostEqual(var_flux.get_data(), reference_var_flux[i],
                                   delta=1e-2 * max(1.0, reference_var_flux[i]))

        for r_str_id, ref_max_sens, ref_final_sens, ref_flux_sens in zip(
                settings["reaction_ids"], reference_c_max_ea_sens, reference_c_final_ea_sens, reference_c_flux_ea_sens):
            reaction = db.Reaction(db.ID(r_str_id), reactions)
            a_id = reaction.get_reactants(db.Side.BOTH)[0][0]
            a: Union[db.Compound, db.Flask] = db.Compound(a_id, compounds)
            if not a.exists():
                a = db.Flask(a_id, flasks)
            centroid = db.Structure(a.get_centroid(), structures)
            for ref, label in zip([ref_max_sens, ref_final_sens, ref_flux_sens], ["max", "final", "flux"]):
                prop_label = r_str_id + "_reaction_barrier_sensitivity_oaat_" + label
                assert centroid.has_property(prop_label)
                ea_max_sens_prop = db.NumberProperty(centroid.get_properties(prop_label)[0], properties)
                self.assertAlmostEqual(ea_max_sens_prop.get_data(), ref, delta=1e-2)

        settings["sensitivity_analysis"] = "sobol"
        calculation = add_calculation(self.manager, model, db.Job("rms_kinetic_modeling"), all_structure_ids, settings)
        # Run calculation/job
        job = RmsKineticModeling()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        for i, s_id in enumerate(all_structure_ids):
            structure = db.Structure(s_id, structures)
            assert structure.has_property("max_free_energy_sensitivity_sobol_s1_c_max")
            assert structure.has_property("max_free_energy_sensitivity_sobol_st_c_max")
            assert structure.has_property("max_free_energy_sensitivity_sobol_s1_c_final")
            assert structure.has_property("max_free_energy_sensitivity_sobol_st_c_final")
            assert structure.has_property("sobol_mean_c_max")
            assert structure.has_property("sobol_mean_c_final")
            assert structure.has_property("sobol_mean_c_flux")
            assert structure.has_property("sobol_var_c_max")
            assert structure.has_property("sobol_var_c_final")
            assert structure.has_property("sobol_var_c_flux")
