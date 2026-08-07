from math import isinf

import numpy as np
import pytest

from atomic_physics import LarmorPrecession


def test_solution_satisfies_initial_condition_and_one_period() -> None:
    initial_moment = np.array([2.0, -0.5, 1.2])
    model = LarmorPrecession([0.0, 0.0, 3.0], initial_moment)

    np.testing.assert_allclose(model.magnetic_moment(0.0), initial_moment)
    np.testing.assert_allclose(
        model.magnetic_moment(model.period),
        initial_moment,
        atol=1e-14,
    )


def test_magnitude_and_axial_projection_are_conserved() -> None:
    model = LarmorPrecession([1.0, -2.0, 0.5], [0.2, 1.1, -0.7])
    times = np.linspace(0.0, 4.0 * model.period, 101)
    moments = model.magnetic_moment(times)

    expected_norm = np.linalg.norm(model.initial_magnetic_moment)
    np.testing.assert_allclose(
        np.linalg.norm(moments, axis=-1),
        expected_norm,
        rtol=1e-14,
        atol=1e-14,
    )

    initial_projection = np.dot(
        model.initial_magnetic_moment,
        model.rotation_axis,
    )
    np.testing.assert_allclose(
        moments @ model.rotation_axis,
        initial_projection,
        rtol=1e-14,
        atol=1e-14,
    )


def test_derivative_matches_omega_cross_mu() -> None:
    model = LarmorPrecession([0.4, 0.7, -1.3], [1.2, -0.3, 0.8])
    time = 0.37
    step = 1e-6
    finite_difference = (
        model.magnetic_moment(time + step)
        - model.magnetic_moment(time - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        finite_difference,
        model.derivative(time),
        rtol=1e-9,
        atol=1e-9,
    )


def test_zero_angular_velocity_keeps_moment_constant() -> None:
    model = LarmorPrecession([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    assert isinf(model.period)
    np.testing.assert_allclose(
        model.magnetic_moment([0.0, 1.0, 5.0]),
        np.tile([1.0, 2.0, 3.0], (3, 1)),
    )


@pytest.mark.parametrize(
    ("angular_velocity", "initial_moment"),
    [
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, np.nan, 3.0]),
    ],
)
def test_invalid_vectors_are_rejected(
    angular_velocity: list[float],
    initial_moment: list[float],
) -> None:
    with pytest.raises(ValueError):
        LarmorPrecession(angular_velocity, initial_moment)
