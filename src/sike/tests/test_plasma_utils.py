import pytest
import json
from pathlib import Path

from hypothesis import given, strategies as st, settings

from sike.plasma_utils import *
from sike.atomics.atomic_state import State
import numpy as np


def input_states():
    examples_states_filepath = Path(__file__).parent / "data" / "example_states.json"

    with open(examples_states_filepath) as f:
        levels_dict = json.load(f)
        states = [None] * len(levels_dict)
        for i, level_dict in enumerate(levels_dict):
            states[i] = State(**level_dict)
    states[0].ground = True
    states[-1].ground = True
    for i in range(1, len(states) - 1):
        states[i].ground = False
    return states


@given(
    T = st.sampled_from([0.1, 100.0, 1000.0])
)
def test_boltzmann_dist(T):
    """Test the boltzmann_dist function"""
    # No statistical weight dependence
    num_states = 100
    energies = np.linspace(1e-5, 1000, num_states)
    stat_weights = np.ones(num_states)

    dist = boltzmann_dist(T, energies, stat_weights)
    # Check length of output array is correct
    assert len(dist) == num_states

    # Check that densities monotonically decrease
    assert all(dist[:-1] >= dist[1:])

    # Assert that g_normalise does nothing when stat_weights = 1
    dist_gn = boltzmann_dist(T, energies, stat_weights, gnormalise=True)
    assert all(dist_gn == dist)

    # Statistical weight dependence
    num_states = 3
    energies = np.ones(num_states)
    stat_weights = np.array([1, 2, 3])

    # Check densities are in reverse order for these statistical weights
    dist = boltzmann_dist(T, energies, stat_weights)
    assert all(dist[:-1] <= dist[1:])

    # Check nornmalised densities are in correct order
    dist_gn = boltzmann_dist(T, energies, stat_weights, gnormalise=True)
    assert all(dist_gn[:-1] >= dist_gn[1:])


@given(
    T = st.sampled_from([0.0001, 1.0, 5.0, 10.0, 100000.0]),
    n = st.sampled_from([1e18, 1e19, 1e20]),
    nz = st.sampled_from([1e12, 1e16, 1e20])
)
def test_saha_dist(T, n, nz):
    """Test the saha_dist function"""
    dist = saha_dist(T, n, nz, input_states(), num_Z=2)
    assert np.isclose(np.sum(dist), nz)


def test_saha_limits():
    n = 1e20
    nz = 1e12

    # Check un-ionized at low Te
    dist = saha_dist(0.00001, n, nz, input_states(), num_Z=2)
    assert np.isclose(dist[0], nz)
    assert np.isclose(dist[1], 0.0)

    # Check fully ionized at high Te
    dist = saha_dist(1e9, n, nz, input_states(), num_Z=2)
    assert np.isclose(dist[1], nz)
    assert np.isclose(dist[0] / nz, 0.0)


@settings(deadline=None)  # Removes test-timeout if expensive
@given(
    T=st.sampled_from([1.0, 5.0, 10.0, 100.0, 1000.0]),
    n=st.sampled_from([1e18, 1e19, 1e20])
)
def test_maxwellian_dist(T: float, n: float):
    """Test maxwellian function

    :param T: Temperature
    :param n: Density
    """

    _, Egrid = generate_vgrid(nv=1000)

    dist = maxwellian(T, n, Egrid)

    jacobian = 2.0 * np.pi * np.sqrt(Egrid)
    zeroth_moment = np.trapezoid(dist * jacobian, x=Egrid)
    first_moment = np.trapezoid(dist * jacobian * Egrid, x=Egrid)

    # The zeroth moment of the distribution should be equal to the density
    assert np.isclose(zeroth_moment, n, atol=1e-3, rtol=1e-3)

    # The first moment should be proportional to the temperature
    assert np.isclose(first_moment, 1.5 * n * T, atol=1e-3, rtol=1e-3)


@settings(deadline=None)  # Removes test-timeout if expensive
@given(
    T1=st.sampled_from([1.0, 5.0, 10.0, 100.0, 1000.0]),
    T2=st.sampled_from([1.0, 5.0, 10.0, 100.0, 1000.0]),
    n1=st.sampled_from([1e18, 1e19, 1e20]),
    n2=st.sampled_from([1e18, 1e19, 1e20])
)
def test_bimaxwellian_dist(T1: float, T2: float, n1: float, n2: float):
    """Test bimaxwellian function

    :param T1: Temperature of Maxwellian 1
    :param T2: Temperature of Maxwellian 2
    :param n1: Density of Maxwellian 1
    :param n2: Density of Maxwellian 2
    """

    _, Egrid = generate_vgrid(nv=1000)

    dist = bimaxwellian(T1, n1, T2, n2, Egrid)

    jacobian = 2.0 * np.pi * np.sqrt(Egrid)
    zeroth_moment = np.trapezoid(dist * jacobian, x=Egrid)
    first_moment = np.trapezoid(dist * jacobian * Egrid, x=Egrid)

    # The zeroth moment of the distribution should be equal to the sum of the densities
    assert np.isclose(zeroth_moment, n1+n2, atol=1e-3, rtol=1e-3)

    # The first moment should be proportional to the weighted sum of the temperatures
    assert np.isclose(first_moment, 1.5 * (n1*T1 + n2*T2), atol=1e-3, rtol=1e-3)
