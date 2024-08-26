# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Optional, Any, Tuple, List
import numpy as np

from .rms_kinetic_model import RMSKineticModel


class RMSKineticModelingSensitivityAnalysis:
    """
    Provides a Wrapper around SALib for sensitivity analysis of the RMS kinetic modeling output.

    SALib samples the parameters (activation energies and enthalpies) within the uncertainty around input value.
    The input parameters are sampled uniformly distribution of the parameters.

    To ensure that we can use multiprocessing, we cannot work with any Julia objects in the main thread. Julia objects
    may only be constructed after starting the parallel loop.

    Parameters
    ----------
    rms_kinetic_model : RMSKineticModel
        The microkinetic model.
    n_cores : int
        Number of cores to run the sensitivity analysis on. Note that if n > 1, RMS must not have been instantiated
        before in the python main process because Julia runs into trouble otherwise.
    sample_size : int
        Number of samples for Morris (5 - 25) or Sobol ( > 500) analysis.
    distribution_shape : str (default 'unif')
        Shape of the parameter distribution to be assumed. Options are uniform distribution between error bounds
        ('unif') and truncated normal distributions ('truncnorm'). The normal distributions is only truncated to
        ensure non-negative reaction barriers. The standard deviation for the normal distribution is taken as the
        standard deviation of the error bounds and the normal distribution's mean is taken as the baseline
        parameter in the microkinetic model. Note that a non-uniform distribution may lead to rather strange
        parameter sampling if Morris sampling is used because it constructs discrete parameter levels within the
        distribution.

    TODO: The number of levels for the Morris sampling should be an input argument.
    """

    def __init__(self, rms_kinetic_model: RMSKineticModel, n_cores: int, sample_size: int,
                 distribution_shape: str = 'unif') -> None:
        self.rms_model = rms_kinetic_model
        self.n_cores = n_cores
        self.sample_size = sample_size
        self._problem: Optional[Any] = None
        self._morris_max_mu: Optional[np.ndarray] = None
        self._morris_max_mu_star: Optional[np.ndarray] = None
        self._morris_max_sigma: Optional[np.ndarray] = None
        self._sobol_max_total: Optional[np.ndarray] = None
        self._sobol_max_s1: Optional[np.ndarray] = None
        self._full_to_reduced_parameter_mapping: Optional[List[Tuple[int, int]]] = None
        self._reduced_to_full_parameter_mapping: Optional[List[Tuple[int, int]]] = None
        self.sensitivity_times: Optional[List[float]] = None
        distribution_options = ['unif', 'truncnorm']
        if distribution_shape not in distribution_options:
            raise RuntimeError(f"The distribution shape for the kinetic modeling parameter must be in"
                               f" {distribution_options}.")
        self._distribution_shape = distribution_shape
        self.include_fluxes: bool = True

    def _define_sampling_problem(self):
        """
        Create problem specification object for SALib.
        """
        # pylint: disable=import-error
        from SALib import ProblemSpec
        # pylint: enable=import-error
        n_aggregates = self.rms_model.get_n_aggregates(with_solvent=False)
        n_rxn = self.rms_model.get_n_reactions()
        # Bounds (lower, upper) for uniform distributions.
        # Bounds (lower, upper, mean, std-dev) for truncated normal distributions.
        bounds = self.get_parameter_bounds()
        if self._distribution_shape == 'truncnorm':
            bounds = self.get_parameter_mean_and_std_dev_bounds()
        distributions = [self._distribution_shape for _ in bounds]
        n_outputs = self.get_n_total_output()
        outputs = ['c_max_' + str(i) for i in range(n_aggregates)] + ['c_' + str(i) for i in range(n_aggregates)]
        if self.include_fluxes:
            outputs += ['c_flux_' + str(i) for i in range(n_aggregates)]
        if len(outputs) < n_outputs:
            n_sets = int((n_outputs - len(outputs)) / n_aggregates)
            for i_set in range(n_sets):
                outputs += ['ct' + str(i_set) + "_" + str(i) for i in range(n_aggregates)]
        assert len(outputs) == n_outputs
        f2r_mapping = self.get_reduced_parameter_mapping()
        full_names = ['h' + str(i) for i in range(n_aggregates)] + ['ea' + str(i) for i in range(n_rxn)]
        problem = ProblemSpec({
            'names': [full_names[full_index] for full_index, _ in f2r_mapping],
            'bounds': bounds,
            'outputs': outputs,
            'dists': distributions
        })
        return problem

    def analyse_runs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate mean and variance of the outputs of the sensitivity analysis runs.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            Returns a list of tuples (tuple[0] -> mean ; tuple[1] -> variance) of the different sensitivity analysis
            outputs (these can be aggregate-wise maximum concentrations, fluxes or just concentrations at specific time
            points).
        """
        if self.get_analysis().results is None:
            raise RuntimeError("Run the sensitivity analysis first, please.")
        all_results = self.get_analysis().results  # rows -> runs | cols -> outputs
        n_aggregates = self.rms_model.get_n_aggregates(with_solvent=False)
        n_outputs = all_results.shape[1]
        n_sets = int(n_outputs / n_aggregates)
        assert n_sets * n_aggregates == n_outputs
        separated_output = [all_results[:, int(i * n_aggregates): int((i + 1) * n_aggregates)] for i in range(n_sets)]
        return [(np.mean(out, axis=0), np.var(out, axis=0)) for out in separated_output]

    def morris_sensitivities(self):
        """
        Run Morris sensitivity analysis. The number of model evaluations is M = N(p + 1), where N is the number of
        samples for each parameter and p is the number of parameters (number of reactions + number of aggregates).
        The number of samples (N) is typically between 5 and 25.

        Returns
        -------
        Returns the sensitivity measures for the maximum and final concentrations for each parameter as a dictionary.

        The measures are:
            max_mu: Maximum value of the Morris sensitivity measure mu for the parameter and maximum/final
            concentrations.
            max_mu_star: Maximum value of the Morris sensitivity measure mu* for the parameter and maximum/final
            concentrations.
            max_sigma: Maximum value of the Morris sensitivity measure sigma for the parameter and maximum/final
            concentrations.
        """
        problem = self._define_sampling_problem()
        # pylint: disable=no-member
        if self._distribution_shape == "unif" and self.get_n_parameters() < 1e+3:
            # We use the trajectory selection by Ruano et al. https://doi.org/10.1016/j.envsoft.2012.03.008
            problem.sample_morris(min(500, self.sample_size * 10), optimal_trajectories=max(2, self.sample_size),
                                  local_optimization=True)
        else:
            problem.sample_morris(self.sample_size)
        # pylint: enable=no-member
        print("Morris' method, number of model evaluations:", len(problem.samples))
        # Multiprocessing is somewhat difficult with Julia. Since the julia-code is compiled specifically for the
        # process we have to make sure that Julia is only ever imported in a sub-process and never in the main process.
        self.evaluate_salib_parallel(problem, self.salib_wrapped_kinetic_modeling, nprocs=self.n_cores)
        print("Model evaluations done! Analyzing output.")
        # pylint: disable=no-member
        outputs = problem.analyze_morris().to_df()
        self._problem = problem
        # pylint: enable=no-member
        self._morris_max_mu = self._result_wise_abs_max(outputs, 'mu')
        self._morris_max_mu_star = self._result_wise_abs_max(outputs, 'mu_star')
        self._morris_max_sigma = self._result_wise_abs_max(outputs, 'sigma')
        return self._morris_max_mu, self._morris_max_mu_star, self._morris_max_sigma, outputs

    def sobol_sensitivities(self):
        """
        Run Sobol sensitivity analysis (with Saltelli samping). The number of model evaluations is M = N(p + 2), where
        N is the number of samples for each parameter and p is the number of parameters (number of reactions + number
        of aggregates). Sample size (N) should be 500 or larger (depending on the number of model parameters).
        """
        self._problem = self._define_sampling_problem()
        # pylint: disable=no-member
        self._problem.sample_saltelli(self.sample_size, calc_second_order=False)
        # pylint: enable=no-member
        self._problem.evaluate(self.salib_wrapped_kinetic_modeling, nprocs=self.n_cores)
        print("Model evaluations done! Analyzing output.")
        # pylint: disable=no-member
        outputs = self._problem.analyze_sobol(calc_second_order=False).to_df()
        # pylint: enable=no-member
        self._sobol_max_total = self._result_wise_abs_max(outputs, 'ST')
        self._sobol_max_s1 = self._result_wise_abs_max(outputs, 'S1')
        return self._sobol_max_total, self._sobol_max_s1, outputs

    @staticmethod
    def evaluate_salib_parallel(problem, func, nprocs=None):
        """Evaluate model locally in parallel.
        All detected processors will be used if `nprocs` is None.

        This is a reduced version of SALib's evaluate_parallel function. We need this reimplementation to
        better handle the parallelization itself. The size of the class and all parameters given the the
        individual threads must not become too large. Otherwise, we may be unable to pickle it.

        Parameters
        ----------
        problem : ProblemSpec
            The SALib problem spec.
        func : function,
            The evaluation function.
        nprocs : int,
            The number of processes.
        """
        if problem._samples is None:
            raise RuntimeError("Sampling not yet conducted")

        max_procs = cpu_count()
        if nprocs is None:
            nprocs = max_procs
        else:
            nprocs = min(max_procs, nprocs)

        # Split into chunks. The chunk sizes should not become too large to avoid requiring too much memory for pickle
        # to handle.
        if problem._samples.shape[0] > nprocs * 1e+2:
            n_chunks = int(problem._samples.shape[0] / 100)
        else:
            n_chunks = nprocs
        chunks = np.array_split(problem._samples, n_chunks, axis=0)

        with Pool(nprocs) as pool:
            res = list(pool.imap(func, chunks))

        problem.results = problem._collect_results(res)

        return

    def get_n_parameters(self) -> int:
        """
        Getter for the number of microkinetic model parameters.
        """
        return self.rms_model.get_n_parameters()

    def get_reduced_parameter_mapping(self):
        """
        Getter for the mapping between full parameter list and prescreened parameter list.
        """
        if self._full_to_reduced_parameter_mapping is None:
            return [(i, i) for i in range(self.get_n_parameters())]
        return self._full_to_reduced_parameter_mapping

    def set_prescreening_condition(self, vertex_flux: np.ndarray, edge_flux: np.ndarray, vertex_t: float,
                                   edge_t: float):
        """
        Set a prescreeining condition to reduce the parameters sampled during sensitivity analysis.
        Note: At the moment this will only affect the one-at-a-time analysis.
        """
        reduced_parameter_indices: List[Tuple[int, int]] = []
        reduced_index = 0
        n_enthalpies = len(self.rms_model.uq_h_lower)
        n_ea = len(self.rms_model.uq_ea_lower)
        for i in range(n_enthalpies):
            if abs(vertex_flux[i]) > vertex_t:
                reduced_parameter_indices.append((i, reduced_index))
                reduced_index += 1
        for i in range(n_ea):
            if abs(edge_flux[i]) > edge_t:
                reduced_parameter_indices.append((i + n_enthalpies, reduced_index))
                reduced_index += 1
        self._full_to_reduced_parameter_mapping = reduced_parameter_indices

    def get_parameter_bounds(self) -> List[List[float]]:
        """
        Create the parameter bound list to represent a uniform parameter distribution around the baseline.
        """
        ea_bounds = [[max(0.0, ea - lower), ea + max(upper, 1.0)] for lower, upper, ea in
                     zip(self.rms_model.uq_ea_lower, self.rms_model.uq_ea_upper, self.rms_model.ea)]
        h_bounds = [[h - max(lower, 1.0), h + max(upper, 1.0)] for lower, upper, h in
                    zip(self.rms_model.uq_h_lower, self.rms_model.uq_h_upper, self.rms_model.h)]
        f2r_mapping = self.get_reduced_parameter_mapping()
        p_bounds: List[List[float]] = []
        full_p_bounds = h_bounds + ea_bounds
        for full_i, _ in f2r_mapping:
            p_bounds.append(full_p_bounds[full_i])
        return p_bounds

    def get_parameter_mean_and_std_dev_bounds(self) -> List[List[float]]:
        """
        Convert the error bound list to a list of mean, standard deviation, and parameter range to represent a truncated
        normal distribution around the baseline.
        """
        ea_mean_std = [[ea, max(abs(lower + upper) / 2, 0.5)] for lower, upper, ea in
                       zip(self.rms_model.uq_ea_upper, self.rms_model.uq_ea_upper, self.rms_model.ea)]
        h_mean_std = [[h, max(abs(lower + upper) / 2, 0.5)] for lower, upper, h in
                      zip(self.rms_model.uq_h_lower, self.rms_model.uq_h_upper, self.rms_model.h)]
        ea_lower_shifts = [ea - max(0.0, ea - 10 * lower) for lower, ea in
                           zip(self.rms_model.uq_ea_lower, self.rms_model.ea)]
        ea_bounds = [[max(0.0, ea - shift), ea + max(shift, 1.0)] for shift, ea in
                     zip(ea_lower_shifts, self.rms_model.ea)]
        h_bounds = [[h - max(10 * lower, 1.0), h + max(10 * upper, 1.0)] for lower, upper, h in
                    zip(self.rms_model.uq_h_lower, self.rms_model.uq_h_upper, self.rms_model.h)]
        full_bounds = [a + b for a, b in zip(h_bounds + ea_bounds, h_mean_std + ea_mean_std)]

        # map to reduced parameter set.
        p_bounds: List[List[float]] = []
        f2r_mapping = self.get_reduced_parameter_mapping()
        for full_i, _ in f2r_mapping:
            p_bounds.append(full_bounds[full_i])
        return p_bounds

    def get_local_sensitivity_samples(self) -> Tuple[np.ndarray, List[int]]:
        """
        Getter for the local sensitivity samples (parameter combinations). Parameters are distorted by their error
        bounds one-at-a-time from the baseline.
        """
        parameter_bounds = self.get_parameter_bounds()
        f2r_mapping = self.get_reduced_parameter_mapping()
        full_parameters = self.rms_model.get_all_parameters()
        reduced_parameters = self._full_to_reduced_parameters(full_parameters, f2r_mapping, len(full_parameters))
        samples = []
        parameter_indices = []
        n_agg = self.rms_model.get_n_aggregates(with_solvent=False)
        assert len(f2r_mapping) == len(parameter_bounds)
        for (full_i, reduced_i), (lower, upper) in zip(f2r_mapping, parameter_bounds):
            p = deepcopy(reduced_parameters)
            p[reduced_i] = upper
            samples.append(p)
            parameter_indices.append(full_i)
            if full_i >= n_agg and self.rms_model.ea[full_i - n_agg] < 1.0:  # no point in lowering 0.0 barriers.
                continue
            p = deepcopy(reduced_parameters)
            p[reduced_i] = lower
            samples.append(p)
            parameter_indices.append(full_i)
        return np.array([s for s in samples]), parameter_indices

    def one_at_a_time_differences(self, vertex_fluxes: np.ndarray, edge_fluxes: np.ndarray, vertex_threshold: float,
                                  edge_threshold: float, flux_replace: float, ref_max: np.ndarray,
                                  ref_final: np.ndarray):
        """
        Run one-at-a-time local sensitivity analysis. The parameters are distorted from the base line one at a time by
        the error bounds provided for each parameter. The maximum change in max concentrations, final concentrations and
        concentration flux is provided as a result.

        Parameters
        ----------
        vertex_fluxes : np.ndarray
            The vertex fluxes of the baseline model (the model with all parameters as their default).
        edge_fluxes : np.ndarray
            The edge fluxes of the baseline model.
        vertex_threshold : float
            Vertex fluxes over this values are considered high and reduced to flux_replace. This should remove
            absolutely large but unimportant changes of the fluxes.
        edge_threshold : float
            Edge fluxes over this values are considered high and reduced to flux_replace. This should remove absolutely
            large but unimportant changes of the fluxes.
        flux_replace : float
            The flux replacement value.
        ref_max : np.ndarray
            The maximum concentrations of the baseline model.
        ref_final : np.ndarray
            The final concentrations of the baseline model.
        """
        self.set_prescreening_condition(vertex_fluxes, edge_fluxes, vertex_threshold, edge_threshold)
        samples, parameter_indices = self.get_local_sensitivity_samples()
        n_samples = samples.shape[0]
        n_agg = self.rms_model.get_n_aggregates(with_solvent=False)
        n_outputs = 3 * n_agg
        n_params = self.get_n_parameters()
        self.set_analysis_times([])  # No additional time points
        self.include_fluxes = True
        if self.n_cores > 1:
            chunksizes = [int(n_samples / self.n_cores) for _ in range(self.n_cores)]
            left_over = n_samples - sum(chunksizes)
            chunksizes[0] += left_over
            assert n_samples - sum(chunksizes) == 0
            chunks = []
            i_sample = 0
            for size in chunksizes:
                chunks.append(samples[i_sample: i_sample + size, :])
                i_sample += size
            with Pool(self.n_cores) as pool:
                process_results = pool.imap(self.salib_wrapped_kinetic_modeling, [s for s in chunks])
                results = np.empty((n_samples, n_outputs))
                ind = 0
                for p_result in process_results:
                    n_sam = len(p_result)
                    results[ind: ind + n_sam, :] = p_result
                    ind += n_sam
        else:
            results = self.salib_wrapped_kinetic_modeling(samples)

        ref_flux = deepcopy(vertex_fluxes)
        ref_flux[ref_flux > flux_replace] = flux_replace
        sens_c_max = np.zeros(n_params)
        sens_c_final = np.zeros(n_params)
        sens_c_flux = np.zeros(n_params)
        all_c_max = np.empty((n_samples + 1, n_agg))
        all_c_final = np.empty((n_samples + 1, n_agg))
        all_c_flux = np.empty((n_samples + 1, n_agg))
        # Note that the reference concentrations may still include the solvent species. Therefore, we only copy the
        # first n_agg concentrations into the final array.
        all_c_max[n_samples, :] = ref_max[:n_agg]
        all_c_final[n_samples, :] = ref_final[:n_agg]
        all_c_flux[n_samples, :] = vertex_fluxes[:n_agg]
        all_c_max[:n_samples, :] = results[:, 0:n_agg]
        all_c_final[:n_samples:, :] = results[:, n_agg: 2 * n_agg]
        all_c_flux[:n_samples:, :] = results[:, 2 * n_agg:]

        for p_index, c_max, c_final, c_flux in zip(parameter_indices, all_c_max, all_c_final, all_c_flux):
            c_flux[c_flux > flux_replace] = flux_replace
            sens_c_max[p_index] = max(sens_c_max[p_index], np.max(np.abs(c_max - ref_max[:n_agg])))
            sens_c_final[p_index] = max(sens_c_final[p_index], np.max(np.abs(c_final - ref_final[:n_agg])))
            sens_c_flux[p_index] = max(sens_c_flux[p_index], np.max(np.abs(c_flux - ref_flux[:n_agg])))

        var_final = np.var(all_c_final, axis=0)
        var_max = np.var(all_c_max, axis=0)
        var_flux = np.var(all_c_flux, axis=0)
        return sens_c_max, sens_c_final, sens_c_flux, var_max, var_final, var_flux

    def get_analysis(self):
        """
        Getter for the SALib problem object that contains all raw outputs and results if the analysis was done.
        """
        if self._problem is None:
            raise RuntimeError("The sensitivity analysis must be executed before any results are available.")
        return self._problem

    def _collect_salib_output(self, outputs, metric_key: str) -> List[np.ndarray]:
        """
        Collect the SALib output and separate them into sets containing one value per aggregate.
        """
        n_aggregates = self.rms_model.get_n_aggregates(with_solvent=False)
        # out_array: outputs x params --> take max along the rows for the specific row subblock.
        if metric_key == 'ST':
            out_array = np.array([out[0][metric_key] for out in outputs])
        elif metric_key == 'S1':
            out_array = np.array([out[1][metric_key] for out in outputs])
        else:
            out_array = np.array([out[metric_key] for out in outputs])
        n_sets = int(len(outputs) / n_aggregates)
        assert n_sets * n_aggregates == len(outputs)
        separated_output = [out_array[int(i * n_aggregates): int((i + 1) * n_aggregates), :] for i in range(n_sets)]
        return separated_output

    def _result_wise_abs_max(self, outputs, metric_key: str):
        """
        Extract the sensitivity indices from the output and calculate the absolute maximum value of the index for each
        parameter. Maximum taken over all outputs.
        """
        separated_output: List[np.ndarray] = self._collect_salib_output(outputs, metric_key)
        metric_c_max = separated_output[0]
        metric_c_final = separated_output[1]
        mapping = self.get_reduced_parameter_mapping()
        n_params = self.get_n_parameters()
        return {'c_max': self._reduced_to_full_parameters(np.amax(np.abs(metric_c_max), axis=0), mapping, n_params),
                'c_final': self._reduced_to_full_parameters(np.amax(np.abs(metric_c_final), axis=0), mapping, n_params)}

    @staticmethod
    def _reduced_to_full_parameters(metric: np.ndarray, mapping: List[Tuple[int, int]],
                                    n_total_params: int) -> np.ndarray:
        """
        Parameters
        ----------
        metric : np.ndarray
            The metric in the reduced parameter set.
        mapping : List[Tuple[int, int]]
            Full parameter index - reduced parameter index tuples.
        n_total_params : int
            Total number of parameters.

        Returns
        -------
        np.ndarray
            The metric in the full parameter dimensions.
        """
        assert metric.shape == (len(mapping),)
        full_metric = np.zeros(n_total_params)
        for full_index, reduced_index in mapping:
            full_metric[full_index] = metric[reduced_index]
        return full_metric

    @staticmethod
    def _update_by_reduced_parameters(update: np.ndarray, full_set: np.ndarray,
                                      mapping: List[Tuple[int, int]]) -> np.ndarray:
        """
        Update some of the parameters in the full parameter list.

        Parameters
        ----------
        update : np.ndarray
            The new parameter values.
        full_set : np.ndarray
            The full set of parameters.
        mapping : List[Tuple[int, int]]
            The indices of the update values in the full parameter set.

        Returns
        -------
        np.ndarray
            The updated full parameter set.
        """
        assert len(update) == len(mapping)
        params = deepcopy(full_set)
        for full_index, reduced_index in mapping:
            params[full_index] = update[reduced_index]
        return params

    @staticmethod
    def _full_to_reduced_parameters(metric: np.ndarray, mapping: List[Tuple[int, int]],
                                    n_total_params: int) -> np.ndarray:
        """
        Inverse operation to _reduced_to_full_parameters(...)
        """
        assert metric.shape == (n_total_params, )
        result = np.asarray([metric[full_index] for full_index, _ in mapping])
        return result

    def set_analysis_times(self, sensitivity_times: List[float]):
        """
        Provide a set of time points which should be considered for the sensitivity analysis.
        """
        valid_times = []
        for t in sensitivity_times:
            if 1e-9 < t < self.rms_model.max_time:
                valid_times.append(t)
            else:
                print(f"Additional time points for the sensitivity analysis must be between 1e-9 s and the maximum\n"
                      f"time specified in the input {self.rms_model.max_time} s. Ignoring the time point {t} s.")
        self.sensitivity_times = valid_times

    def get_n_total_output(self):
        """
        Get the total number of outputs from the sensitivity analysis.
        """
        n_times_agg = 2
        if self.include_fluxes:
            n_times_agg += 1
        if self.sensitivity_times is not None:
            n_times_agg += len(self.sensitivity_times)
        return self.rms_model.get_n_aggregates(with_solvent=False) * n_times_agg

    def salib_wrapped_kinetic_modeling(self, params_set: np.ndarray):
        """
        SALib run function for the model evaluation.

        Parameters
        ----------
        params_set : np.ndarray
            An array containing a set of parameters for the run.
        """
        import os
        n_aggregates = self.rms_model.get_n_aggregates(with_solvent=False)
        n_ea = len(self.rms_model.ea)
        n_param_sets = len(params_set)
        n_outputs = self.get_n_total_output()
        all_c = np.zeros((n_param_sets, n_outputs))
        filename = str(os.getpid()) + ".sample.rms"
        mapping = self.get_reduced_parameter_mapping()
        full_parameters = self.rms_model.get_all_parameters()
        for i, reduced_params in enumerate(params_set):
            # We use prescreening for the sensitivity analysis. Therefore, SALib only knows of the non-screened
            # parameters and we need to map back to the full parameter set to run the actual calculations.
            params = self._update_by_reduced_parameters(reduced_params, full_parameters, mapping)
            h = params[:n_aggregates]
            ea = params[n_aggregates:]
            assert ea.shape[0] == n_ea
            assert h.shape[0] == n_aggregates
            ea = self.rms_model.ensure_non_negative_barriers(ea, h, self.rms_model.s)
            simulation, _, volume, _, sol = self.rms_model.run_kinetic_modeling(filename, h=h.tolist(), ea=ea)

            if simulation is None:
                print("Invalid model solution. This is only a reason to worry if you are not screening large spaces\n"
                      "of parameters. Ignoring this result and continuing the calculation.")
                continue
            if self.include_fluxes:
                c_max, c_final, c_flux, _, additional_c = self.rms_model.integrate_results(simulation, volume,
                                                                                           self.sensitivity_times,
                                                                                           sol)
            else:
                c_max, c_final, additional_c = self.rms_model.concentrations(simulation, volume, self.sensitivity_times)
                c_flux = None
            if os.path.exists(filename):
                os.remove(filename)
            else:
                raise RuntimeError("RMS input file was not created correctly or was removed unexpectedly.")
            # Calling get_n_aggregates again ensures that we ignore the concentration of disconnected solvent compounds.
            all_c[i, :n_aggregates] = c_max[:n_aggregates]
            all_c[i, n_aggregates: 2 * n_aggregates] = c_final[:n_aggregates]
            start_col = 2 * n_aggregates
            if c_flux is not None:
                all_c[i, start_col: start_col + n_aggregates] = c_flux[:n_aggregates]
                start_col += n_aggregates
            if additional_c is not None and self.sensitivity_times:
                n_times = len(self.sensitivity_times)
                all_c[i, start_col:] = np.reshape(additional_c[:, :n_aggregates], (1, n_times * n_aggregates))
        return all_c
