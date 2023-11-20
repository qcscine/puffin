#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import json

import numpy as np

from ..testcases import (
    JobTestCase,
    skip_without
)

from ..db_setup import (
    add_calculation,
    add_compound_and_structure,
    add_structure,
)

from ..resources import resource_path


class ScineReactComplexNtJobTest(JobTestCase):

    @skip_without('database', 'readuct', 'molassembler')
    def test_energy_and_structure(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt import ScineReactComplexNt
        from scine_puffin.utilities.masm_helper import get_molecules_result
        import scine_database as db
        import scine_utilities as utils

        reactant_one_path = os.path.join(resource_path(), "proline_acid.xyz")
        reactant_two_path = os.path.join(resource_path(), "propanal.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path)
        reactant_one_guess = db.Structure(compound_one.get_centroid())
        reactant_two_guess = db.Structure(compound_two.get_centroid())
        reactant_one_guess.link(self.manager.get_collection('structures'))
        reactant_two_guess.link(self.manager.get_collection('structures'))
        graph_one = json.load(open(os.path.join(resource_path(), "proline_acid.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "propanal.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        model = db.Model('dftb3', 'dftb3', '')
        job = db.Job('scine_react_complex_nt')
        settings = {
            "nt_convergence_max_iterations": 600,
            "nt_nt_total_force_norm": 0.05,
            "nt_sd_factor": 1.0,
            "nt_nt_use_micro_cycles": True,
            "nt_nt_fixed_number_of_micro_cycles": True,
            "nt_nt_number_of_micro_cycles": 30,
            "nt_nt_filter_passes": 10,
            "tsopt_convergence_max_iterations": 800,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 1e-06,
            "tsopt_optimizer": "bofill",
            "tsopt_bofill_trust_radius": 0.3,
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "irc_convergence_max_iterations": 75,
            "irc_sd_factor": 2.0,
            "irc_irc_initial_step_size": 0.3,
            "irc_stop_on_error": False,
            "irc_convergence_step_max_coefficient": 0.002,
            "irc_convergence_step_rms": 0.001,
            "irc_convergence_gradient_max_coefficient": 0.0002,
            "irc_convergence_gradient_rms": 0.0001,
            "irc_convergence_delta_value": 1e-06,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 800,
            "ircopt_convergence_step_max_coefficient": 0.002,
            "ircopt_convergence_step_rms": 0.001,
            "ircopt_convergence_gradient_max_coefficient": 0.0002,
            "ircopt_convergence_gradient_rms": 0.0001,
            "ircopt_convergence_requirement": 3,
            "ircopt_convergence_delta_value": 1e-06,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.2,
            "opt_convergence_max_iterations": 800,
            "opt_convergence_step_max_coefficient": 0.002,
            "opt_convergence_step_rms": 0.001,
            "opt_convergence_gradient_max_coefficient": 0.0002,
            "opt_convergence_gradient_rms": 0.0001,
            "opt_convergence_requirement": 3,
            "opt_convergence_delta_value": 1e-06,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_use_trust_radius": True,
            "opt_bfgs_trust_radius": 0.4,
            "imaginary_wavenumber_threshold": -30,
            "nt_nt_lhs_list": [
                3,
                16,
            ],
            "nt_nt_rhs_list": [
                0,
                3
            ],
            "rc_x_alignment_0": [
                -0.405162890256947,
                0.382026105406949,
                0.830601641670804,
                -0.382026105406949,
                0.754648889886101,
                -0.533442693999341,
                -0.830601641670804,
                -0.533442693999341,
                -0.159811780143048
            ],
            "rc_x_alignment_1": [
                0.881555658365378,
                -0.439248859713409,
                -0.172974161204658,
                -0.311954059087669,
                -0.817035288655183,
                0.484910303160151,
                -0.354322291456116,
                -0.373515429845424,
                -0.857287546535394
            ],
            "rc_x_rotation": 0.0,
            "rc_x_spread": 3.48715302740618,
            "rc_displacement": 0.0,
            "rc_minimal_spin_multiplicity": True,
            "spin_propensity_check": 3
        }

        calculation = add_calculation(self.manager, model, job, [reactant_one_guess.id(), reactant_two_guess.id()],
                                      settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        assert len(results.property_ids) == 11
        assert len(results.structure_ids) == 3 + 2  # re-optimized reactants (x2) + complex + TS + product
        assert len(results.elementary_step_ids) == 2
        new_elementary_step_one = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert new_elementary_step_one.get_type() == db.ElementaryStepType.BARRIERLESS
        new_elementary_step_two = db.ElementaryStep(results.elementary_step_ids[1], elementary_steps)
        assert new_elementary_step_two.get_type() == db.ElementaryStepType.REGULAR
        assert new_elementary_step_one.get_reactants(db.Side.RHS)[1][0] == \
            new_elementary_step_two.get_reactants(db.Side.LHS)[0][0]
        s_complex = db.Structure(new_elementary_step_two.get_reactants(db.Side.LHS)[0][0])
        s_complex.link(structures)
        assert s_complex.get_label() == db.Label.COMPLEX_OPTIMIZED
        assert len(new_elementary_step_two.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step_two.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        bonds = utils.BondOrderCollection()
        bonds.matrix = db.SparseMatrixProperty(product.get_property('bond_orders'), properties).get_data()
        assert len(get_molecules_result(product.get_atoms(), bonds, job.connectivity_settings).molecules) == 1
        new_ts = db.Structure(new_elementary_step_two.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -31.659096385274275, delta=1e-1)

    @skip_without('database', 'readuct', 'molassembler', 'ams_wrapper', 'ams')
    def test_surface_reaction(self):
        # import Job
        from scine_puffin.jobs.scine_react_complex_nt import ScineReactComplexNt
        import scine_database as db

        # The parallel numerical Hessian via the PipeInterface of AMS has a high numerical error.
        # Therefore, we enforce the use of the serial Hessian
        omp = os.getenv("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"

        structures = self.manager.get_collection("structures")
        properties = self.manager.get_collection("properties")
        elementary_steps = self.manager.get_collection("elementary_steps")
        model = db.Model('reaxff', 'reaxff', '')
        model.program = "ams"
        model.spin_mode = "none"
        model.electronic_temperature = "none"

        reactant_one_path = os.path.join(resource_path(), "h2.xyz")
        reactant_two_path = os.path.join(resource_path(), "au.xyz")
        complex_path = os.path.join(resource_path(), "au_complex.xyz")
        compound_one = add_compound_and_structure(self.manager, reactant_one_path, multiplicity=3)
        compound_two = add_compound_and_structure(self.manager, reactant_two_path,
                                                  label=db.Label.USER_SURFACE_OPTIMIZED)
        reactant_one_guess = db.Structure(compound_one.get_centroid(), structures)
        reactant_one_guess.set_model(model)
        reactant_two_guess = db.Structure(compound_two.get_centroid(), structures)
        model.periodic_boundaries = "16.721503,16.721503,40.959149,90.000000,90.000000,120.000000,xyz"
        reactant_two_guess.set_model(model)
        complex_structure = add_structure(self.manager, complex_path, db.Label.SURFACE_ADSORPTION_GUESS, 0, 3)
        complex_structure.set_model(model)

        indices = np.array([float(i) for i in range(27)])
        complex_slab_dict = \
            "{'@module': 'pymatgen.core.surface', '@class': 'Slab', 'charge': 0, 'lattice': {'matrix': [[" \
            "8.848638315750037, 0.0, 5.418228295097955e-16], [-4.424319157875018, 7.663145570339881, " \
            "5.418228295097955e-16], [0.0, 0.0, 21.674648219236918]], 'pbc': (True, True, True), " \
            "'a': 8.848638315750037, 'b': 8.848638315750037, 'c': 21.674648219236918, 'alpha': 90.0, 'beta': 90.0, " \
            "'gamma': 119.99999999999999, 'volume': 1469.7232924497353}, 'sites': [{'species': [{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.2778860961723223, 0.27785011439004653, 0.611311989614102], " \
            "'xyz': [1.2296159738909629, 2.1292058733065145, 13.249972327087475], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.27788788562832945, 0.6111879661349863, " \
            "0.6113018003926083], 'xyz': [-0.24516123538008705, 4.6836223553323615, 13.24975147929597], " \
            "'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.2778838122721543, 0.9445145401412581, 0.6113030433366179], 'xyz': [-1.7199404262404057, " \
            "7.237952414405092, 13.249778419670134], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', " \
            "'occu': 1}], 'abc': [0.611216010475045, 0.27784888416258796, 0.6112780782104336], " \
            "'xyz': [4.1791372682946255, 2.1291964458944146, 13.24923730934234], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.6112177834682357, 0.6111867201793348, " \
            "0.6112678888794907], 'xyz': [2.7043599829366114, 4.68361280739283, 13.249016459178563], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.6112137096644943, " \
            "0.9445132926045237, 0.6112691379790377], 'xyz': [1.2295807951111568, 7.237942854349492, " \
            "13.249043532971836], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.9445309020867628, 0.27784014704928345, 0.6112634112253129], 'xyz': [7.12855884519792, " \
            "2.1291294921232975, 13.248919407599413], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.9445326829170073, 0.611177990996807, 0.6112532213800518], " \
            "'xyz': [5.653781594098872, 4.68354591439641, 13.24869854628797], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.9445285635060592, 0.94450454403208, " \
            "0.6112544929457119], 'xyz': [4.179002088498917, 7.237875812765322, 13.24872610702634], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.05547002745654646, " \
            "0.16660427084788956, 0.3887505010887012], 'xyz': [-0.24627625696841252, 1.2767127801477107, " \
            "8.426030356149676], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.05546700620246067, 0.49993273725341525, 0.3887425908135827], 'xyz': [-1.7210545107361441, " \
            "3.831057340951401, 8.425858903719167], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.05547014457126487, 0.8332732966402286, 0.38873720403725776], " \
            "'xyz': [-3.195831763437548, 6.385494572031078, 8.425742147237287], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.38878491622941225, 0.1665955429292308, " \
            "0.38873584369565045], 'xyz': [2.7031452541348595, 1.2766458968365024, 8.425712662311492], 'label': " \
            "'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.38878189968552024, " \
            "0.49992400188350966, 0.3887279232503738], 'xyz': [1.2283670750126234, 3.8309904005402036, " \
            "8.42554098944638], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.3887850322697857, 0.8332645576033345, 0.388722539452408], 'xyz': [-0.24641021265014373, " \
            "6.385427603519213, 8.425424297519388], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.7221148122596339, 0.16659429520467012, 0.38870193527793867], " \
            "'xyz': [5.6526664642645335, 1.2766363353415624, 8.424977709285917], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.7221117995823851, 0.4999227580196199, " \
            "0.3886940123612214], 'xyz': [4.177888302275983, 3.8309808686301463, 8.4248059828532], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.7221149470996455, " \
            "0.8332633260862631, 0.3886886262722038], 'xyz': [2.7031110921236228, 6.385418166224623, " \
            "8.424689241268467], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.16668888044495933, 0.05556470280564529, 0.5000187643769566], 'xyz': [1.2291336351900959, " \
            "0.42580040617233267, 10.837730820888046], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.1666892016854119, 0.38889995593508075, 0.5000225659234466], " \
            "'xyz': [-0.24564506868500663, 2.980196974629289, 10.837813218070908], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.1666891714443799, 0.722234185396945, " \
            "0.5000123683272975], 'xyz': [-1.7204223536605938, 5.534585698572632, 10.837592188761692], 'label': " \
            "'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.4999999739438076, " \
            "0.0555540754598818, 0.5000004103713492], 'xyz': [4.178529966958006, 0.4257189672747207, " \
            "10.837333004273093], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.5000003511660567, 0.388889347819449, 0.5000041891925947], 'xyz': [2.7037516733653333, " \
            "2.9801156830949758, 10.837414908894273], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.5000003183460272, 0.7222235749835915, 0.49999399112578524], " \
            "'xyz': [1.2289743757349842, 5.534504389430542, 10.837193869383661], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.8333111067953511, 0.055543465441786885, " \
            "0.49998204163747373], 'xyz': [7.1279265702805485, 0.42563766116155544, 10.83693486842811], " \
            "'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.8333114961523033, 0.38887873511461624, 0.4999858116310773], 'xyz': [5.653148395950459, " \
            "2.9800343563929474, 10.837016581713256], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.833311468474635, 0.7222129708839298, 0.4999756172013136], " \
            "'xyz': [4.178371105750982, 5.534423128671192, 10.836795621034332], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'H', 'occu': 1}], 'abc': [0.35096447466090563, 0.3462402223346323, " \
            "0.7601942160953102], 'xyz': [1.5736804490495533, 2.653289226057133, 16.47694221216442], 'label': 'H', " \
            "'properties': {}}, {'species': [{'element': 'H', 'occu': 1}], 'abc': [0.3551009194623986, " \
            "0.3492683100920289, 0.7951234817366677], 'xyz': [1.5968851263342523, 2.6764939033418274, " \
            "17.23402175749712], 'label': 'H', 'properties': {}}], 'oriented_unit_cell': {'@module': " \
            "'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {'matrix': [[2.085644, " \
            "-2.085644, -0.0], [-0.0, 2.085644, -2.085644], [4.171288, 4.171288, 4.1712880000000006]], 'pbc': (True, " \
            "True, True), 'a': 2.9495460310820714, 'b': 2.9495460310820714, 'c': 7.224882749002367, 'alpha': 90.0, " \
            "'beta': 90.0, 'gamma': 120.00000000000001, 'volume': 54.434193348844616}, 'sites': [{'species': [{" \
            "'element': 'Au', 'occu': 1}], 'abc': [2.7563307014654614e-32, 5.8228074529959955e-33, " \
            "1.1753544774588717e-16], 'xyz': [4.902742027570462e-16, 4.902742027570461e-16, 4.902742027570463e-16], " \
            "'label': 'Au', 'properties': {'bulk_wyckoff': 'a', 'bulk_equivalent': 0}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.33333333333333337, 0.6666666666666666, 0.3333333333333334], " \
            "'xyz': [2.0856440000000003, 2.0856440000000003, 8.30010874845281e-16], 'label': 'Au', 'properties': {" \
            "'bulk_wyckoff': 'a', 'bulk_equivalent': 0}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.6666666666666664, 0.3333333333333333, 0.6666666666666667], 'xyz': [4.171288, " \
            "2.0856440000000007, 2.0856440000000007], 'label': 'Au', 'properties': {'bulk_wyckoff': 'a', " \
            "'bulk_equivalent': 0}}]}, 'miller_index': (1, 1, 1), 'shift': 0.16666666666666677, 'scale_factor': [[1, " \
            "-1, 0], [1, 0, -1], [1, 1, 1]], 'reconstruction': None, 'energy': None} "

        reactant_slab_dict = \
            "{'@module': 'pymatgen.core.surface', '@class': 'Slab', 'charge': 0, 'lattice': {'matrix': [[" \
            "8.848638093246214, 0.0, 5.418228158853657e-16], [-4.424319046623108, 7.663145377645916, " \
            "5.418228158853657e-16], [0.0, 0.0, 21.6746482470071]], 'pbc': (True, True, True), " \
            "'a': 8.848638093246214, 'b': 8.848638093246214, 'c': 21.6746482470071, 'alpha': 90.0, 'beta': 90.0, " \
            "'gamma': 120.00000000000001, 'volume': 1469.723220418804}, 'sites': [{'species': [{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.3333333333333331, 0.3333333333333331, 0.6111111111111112], " \
            "'xyz': [1.4747730155410343, 2.554381792548637, 13.245618373171006], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.33333333333333304, 0.6666666666666664, " \
            "0.6111111111111112], 'xyz': [-1.962495598385766e-15, 5.108763585097275, 13.245618373171006], " \
            "'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.3333333333333331, 0.9999999999999998, 0.6111111111111113], 'xyz': [-1.4747730155410377, " \
            "7.663145377645915, 13.245618373171009], 'label': 'Au', 'properties': {}}, {'species': " \
            "[{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.6666666666666664, 0.3333333333333331, 0.6111111111111112], " \
            "'xyz': [4.424319046623105, 2.554381792548637, 13.245618373171006], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.6666666666666663, 0.6666666666666664, " \
            "0.6111111111111113], 'xyz': [2.9495460310820687, 5.108763585097275, 13.245618373171009], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.6666666666666664, " \
            "0.9999999999999998, 0.6111111111111112], 'xyz': [1.4747730155410332, 7.663145377645915, " \
            "13.245618373171007], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.9999999999999996, 0.3333333333333331, 0.6111111111111113], 'xyz': [7.373865077705175, " \
            "2.554381792548637, 13.245618373171009], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', " \
            "'occu': 1}], 'abc': [0.9999999999999996, 0.6666666666666664, 0.6111111111111112], " \
            "'xyz': [5.899092062164139, 5.108763585097275, 13.245618373171007], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.9999999999999996, 0.9999999999999998, " \
            "0.6111111111111112], 'xyz': [4.424319046623103, 7.663145377645915, 13.245618373171007], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.11111111111111106, " \
            "0.22222222222222224, 0.38888888888888895], 'xyz': [-6.973483372021525e-16, 1.702921195032426, " \
            "8.429029873836095], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.11111111111111109, 0.5555555555555555, 0.38888888888888895], 'xyz': [-1.474773015541036, " \
            "4.257302987581064, 8.429029873836095], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.11111111111111112, 0.8888888888888887, 0.388888888888889], " \
            "'xyz': [-2.9495460310820714, 6.811684780129703, 8.429029873836097], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.4444444444444444, 0.22222222222222224, " \
            "0.38888888888888895], 'xyz': [2.9495460310820705, 1.702921195032426, 8.429029873836095], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.44444444444444436, " \
            "0.5555555555555555, 0.388888888888889], 'xyz': [1.4747730155410346, 4.257302987581064, " \
            "8.429029873836097], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.44444444444444436, 0.8888888888888887, 0.3888888888888889], 'xyz': [-9.188187543387658e-16, " \
            "6.811684780129703, 8.429029873836095], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', " \
            "'occu': 1}], 'abc': [0.7777777777777778, 0.22222222222222224, 0.388888888888889], " \
            "'xyz': [5.899092062164143, 1.702921195032426, 8.429029873836097], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.7777777777777776, 0.5555555555555555, " \
            "0.3888888888888889], 'xyz': [4.424319046623105, 4.257302987581064, 8.429029873836095], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.7777777777777778, " \
            "0.8888888888888887, 0.3888888888888889], 'xyz': [2.9495460310820714, 6.811684780129703, " \
            "8.429029873836095], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.2222222222222221, 0.11111111111111109, 0.5000000000000001], 'xyz': [1.4747730155410346, " \
            "0.8514605975162128, 10.837324123503551], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.22222222222222213, 0.44444444444444436, 0.5000000000000001], " \
            "'xyz': [-9.034985870194455e-16, 3.4058423900648513, 10.837324123503553], 'label': 'Au', 'properties': {" \
            "}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.22222222222222207, 0.7777777777777777, " \
            "0.5000000000000001], 'xyz': [-1.4747730155410372, 5.96022418261349, 10.837324123503553], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.5555555555555555, " \
            "0.11111111111111109, 0.5000000000000001], 'xyz': [4.424319046623106, 0.8514605975162128, " \
            "10.837324123503553], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.5555555555555554, 0.44444444444444436, 0.5000000000000001], 'xyz': [2.9495460310820696, " \
            "3.4058423900648513, 10.837324123503553], 'label': 'Au', 'properties': {}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.5555555555555554, 0.7777777777777777, 0.5000000000000001], " \
            "'xyz': [1.474773015541034, 5.96022418261349, 10.837324123503553], 'label': 'Au', 'properties': {}}, " \
            "{'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.8888888888888885, 0.11111111111111109, " \
            "0.5000000000000001], 'xyz': [7.373865077705174, 0.8514605975162128, 10.837324123503553], 'label': 'Au', " \
            "'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], 'abc': [0.8888888888888886, " \
            "0.44444444444444436, 0.5000000000000001], 'xyz': [5.89909206216414, 3.4058423900648513, " \
            "10.837324123503553], 'label': 'Au', 'properties': {}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.8888888888888886, 0.7777777777777777, 0.5000000000000001], 'xyz': [4.424319046623104, " \
            "5.96022418261349, 10.837324123503553], 'label': 'Au', 'properties': {}}], 'oriented_unit_cell': {" \
            "'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {'matrix': [[" \
            "2.085644, -2.085644, -0.0], [-0.0, 2.085644, -2.085644], [4.171288, 4.171288, 4.1712880000000006]], " \
            "'pbc': (True, True, True), 'a': 2.9495460310820714, 'b': 2.9495460310820714, 'c': 7.224882749002367, " \
            "'alpha': 90.0, 'beta': 90.0, 'gamma': 120.00000000000001, 'volume': 54.434193348844616}, 'sites': [{" \
            "'species': [{'element': 'Au', 'occu': 1}], 'abc': [2.7563307014654614e-32, 5.8228074529959955e-33, " \
            "1.1753544774588717e-16], 'xyz': [4.902742027570462e-16, 4.902742027570461e-16, 4.902742027570463e-16], " \
            "'label': 'Au', 'properties': {'bulk_wyckoff': 'a', 'bulk_equivalent': 0}}, {'species': [{'element': " \
            "'Au', 'occu': 1}], 'abc': [0.33333333333333337, 0.6666666666666666, 0.3333333333333334], " \
            "'xyz': [2.0856440000000003, 2.0856440000000003, 8.30010874845281e-16], 'label': 'Au', 'properties': {" \
            "'bulk_wyckoff': 'a', 'bulk_equivalent': 0}}, {'species': [{'element': 'Au', 'occu': 1}], " \
            "'abc': [0.6666666666666664, 0.3333333333333333, 0.6666666666666667], 'xyz': [4.171288, " \
            "2.0856440000000007, 2.0856440000000007], 'label': 'Au', 'properties': {'bulk_wyckoff': 'a', " \
            "'bulk_equivalent': 0}}]}, 'miller_index': (1, 1, 1), 'shift': 0.16666666666666677, 'scale_factor': [[1, " \
            "-1, 0], [1, 0, -1], [1, 1, 1]], 'reconstruction': None, 'energy': None} "

        # properties
        # adsorption type
        adsorption_property = db.BoolProperty.make("true_adsorption", model, True, properties)
        complex_structure.set_property("true_adsorption", adsorption_property.id())
        # surface indices
        surface_indices_property = db.VectorProperty.make("surface_indices", model, indices, properties)
        reactant_two_guess.set_property("surface_atom_indices", surface_indices_property.id())
        complex_structure.set_property("surface_atom_indices", surface_indices_property.id())
        # slab dicts
        reactant_slab_property = db.StringProperty.make("slab_dict", model, reactant_slab_dict, properties)
        complex_slab_property = db.StringProperty.make("slab_dict", model, complex_slab_dict, properties)
        reactant_two_guess.set_property("slab_dict", reactant_slab_property.id())
        complex_structure.set_property("slab_dict", complex_slab_property.id())

        graph_one = json.load(open(os.path.join(resource_path(), "h2.json"), "r"))
        graph_two = json.load(open(os.path.join(resource_path(), "au.json"), "r"))
        reactant_one_guess.set_graph("masm_cbor_graph", graph_one["masm_cbor_graph"])
        reactant_one_guess.set_graph("masm_idx_map", graph_one["masm_idx_map"])
        reactant_one_guess.set_graph("masm_decision_list", graph_one["masm_decision_list"])
        reactant_two_guess.set_graph("masm_cbor_graph", graph_two["masm_cbor_graph"])
        reactant_two_guess.set_graph("masm_idx_map", graph_two["masm_idx_map"])
        reactant_two_guess.set_graph("masm_decision_list", graph_two["masm_decision_list"])

        db_job = db.Job('scine_react_complex_nt')

        settings = {
            "external_program_nprocs": 1,
            "spin_propensity_check": 0,
            "nt_convergence_max_iterations": 600,
            "nt_nt_total_force_norm": 0.1,
            "nt_sd_factor": 1.0,
            "nt_nt_use_micro_cycles": True,
            "nt_nt_fixed_number_of_micro_cycles": True,
            "nt_nt_number_of_micro_cycles": 10,
            "nt_nt_filter_passes": 10,
            "tsopt_convergence_max_iterations": 2000,
            "tsopt_convergence_step_max_coefficient": 0.002,
            "tsopt_convergence_step_rms": 0.001,
            "tsopt_convergence_gradient_max_coefficient": 0.0002,
            "tsopt_convergence_gradient_rms": 0.0001,
            "tsopt_convergence_requirement": 3,
            "tsopt_convergence_delta_value": 0.000001,
            "tsopt_optimizer": "dimer",
            "tsopt_dimer_calculate_hessian_once": True,
            "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "tsopt_dimer_trust_radius": 0.2,
            "irc_convergence_max_iterations": 100,
            "irc_sd_factor": 2.0,
            "irc_irc_initial_step_size": 0.3,
            "irc_stop_on_error": False,
            "irc_convergence_step_max_coefficient": 0.002,
            "irc_convergence_step_rms": 0.001,
            "irc_convergence_gradient_max_coefficient": 0.0002,
            "irc_convergence_gradient_rms": 0.0001,
            "irc_convergence_delta_value": 0.000001,
            "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_convergence_max_iterations": 2000,
            "ircopt_convergence_step_max_coefficient": 0.002,
            "ircopt_convergence_step_rms": 0.001,
            "ircopt_convergence_gradient_max_coefficient": 0.0002,
            "ircopt_convergence_gradient_rms": 0.0001,
            "ircopt_convergence_requirement": 3,
            "ircopt_convergence_delta_value": 0.000001,
            "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "ircopt_bfgs_use_trust_radius": True,
            "ircopt_bfgs_trust_radius": 0.2,
            "opt_convergence_max_iterations": 2000,
            "opt_convergence_step_max_coefficient": 0.002,
            "opt_convergence_step_rms": 0.001,
            "opt_convergence_gradient_max_coefficient": 0.0002,
            "opt_convergence_gradient_rms": 0.0001,
            "opt_convergence_requirement": 3,
            "opt_convergence_delta_value": 0.000001,
            "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "opt_bfgs_use_trust_radius": True,
            "opt_bfgs_trust_radius": 0.4,
            "rcopt_convergence_max_iterations": 2000,
            "rcopt_convergence_step_max_coefficient": 0.002,
            "rcopt_convergence_step_rms": 0.001,
            "rcopt_convergence_gradient_max_coefficient": 0.0002,
            "rcopt_convergence_gradient_rms": 0.0001,
            "rcopt_convergence_requirement": 3,
            "rcopt_convergence_delta_value": 0.000001,
            "rcopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
            "rcopt_bfgs_use_trust_radius": True,
            "rcopt_bfgs_trust_radius": 0.4,
            "method_parameters": "AuCSOH.ff",
            "only_distance_connectivity": True,
            "nt_nt_rhs_list": [27],
            "nt_nt_lhs_list": [0],
            "nt_nt_movable_side": "rhs",
            "nt_nt_attractive": True,
            "rc_minimal_spin_multiplicity": False
        }

        calculation = add_calculation(self.manager, model, db_job,
                                      [reactant_two_guess.id(), reactant_one_guess.id(), complex_structure.id()],
                                      settings)

        # Run calculation/job
        config = self.get_configuration()
        job = ScineReactComplexNt()
        job.prepare(config["daemon"]["job_dir"], calculation.id())
        self.run_job(job, calculation, config)

        # Check results
        assert calculation.get_status() == db.Status.COMPLETE
        results = calculation.get_results()
        print(calculation.get_comment())
        assert len(results.property_ids) > 5
        assert len(results.structure_ids) == 2  # TS + product
        assert len(results.elementary_step_ids) == 1
        new_elementary_step = db.ElementaryStep(results.elementary_step_ids[0], elementary_steps)
        assert new_elementary_step.get_type() == db.ElementaryStepType.REGULAR
        assert len(new_elementary_step.get_reactants(db.Side.LHS)[0]) == 2
        assert len(new_elementary_step.get_reactants(db.Side.RHS)[1]) == 1
        product = db.Structure(new_elementary_step.get_reactants(db.Side.RHS)[1][0], structures)
        assert product.get_label() == db.Label.SURFACE_OPTIMIZED
        assert product.has_property('bond_orders')
        assert product.has_graph('masm_cbor_graph')
        assert all(product.has_properties(p) for p in ['bond_orders', 'surface_atom_indices', 'slab_dict'])
        indices_prop = db.VectorProperty(product.get_property('surface_atom_indices'), properties)
        assert np.allclose(indices_prop.get_data(), indices)

        new_ts = db.Structure(new_elementary_step.get_transition_state(), structures)
        assert new_ts.has_property('electronic_energy')
        energy_props = new_ts.get_properties("electronic_energy")
        assert energy_props[0] in results.property_ids
        energy = db.NumberProperty(energy_props[0], properties)
        self.assertAlmostEqual(energy.get_data(), -3.587814458488312, delta=1e-1)
        os.environ["OMP_NUM_THREADS"] = omp
