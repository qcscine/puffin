# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import numpy as np
import scine_utilities as utils


def single_particle_energy_to_matrix(x: utils.SingleParticleEnergies) -> np.ndarray:
    """
    Converts datatype `utils.SingleParticleEnergies` to `np.array`.
    In the restricted case, there is only one column, in the unrestricted case,
    there are two columns, denoted by `alpha` and `beta`.

    Parameters
    ----------
    x : utils.SingleParticleEnergies
        The single particle energies to be converted to numpy matrices. It
        contains the energies and flags denoting the restricted or unrestricted
        case.
    """
    if x.is_restricted:  # restricted case
        return np.array([x.restricted_energies])
    return np.array([x.alpha, x.beta])


def single_particle_energy_from_matrix(x: np.ndarray) -> utils.SingleParticleEnergies:
    """
    Converts datatype `np.ndarray` to `SingleParticleEnergies`.
    If the first dimension of the numpy array is 1, there is only one column of
    energies, indicating the restricted case.
    Otherwise, for two columns, we're in the unrestricted case.

    Parameters
    ----------
    x : np.ndarray
        The array of dimension $1 \times N$ for the restricted case or $2
        \times N$ for the unrestricted case, where $N$ is the number energies.
    """
    if np.shape(x)[0] == 1:  # restricted case
        a = utils.SingleParticleEnergies.make_restricted()
        a.set_restricted(x[0])
        return a
    else:
        a = utils.SingleParticleEnergies.make_unrestricted()
        a.set_unrestricted(x[0], x[1])
        return a
