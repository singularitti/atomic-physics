"""Analytic dynamics for Larmor precession with constant angular velocity."""

from dataclasses import dataclass, field
from math import inf, tau
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _vector3(value: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite three-dimensional vector as an owned float array."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        msg = f"{name} must have shape (3,), received {vector.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return vector.copy()


@dataclass(frozen=True, slots=True)
class LarmorPrecession:
    """Solve ``d mu / dt = omega x mu`` for a constant ``omega``.

    The exact Rodrigues-rotation solution is used instead of numerical time
    stepping, so the magnitude of the magnetic moment and its projection onto
    the precession axis are preserved to floating-point precision.
    """

    angular_velocity: ArrayLike
    initial_magnetic_moment: ArrayLike
    _angular_speed: float = field(init=False, repr=False)
    _rotation_axis: FloatArray = field(init=False, repr=False)
    _parallel_moment: FloatArray = field(init=False, repr=False)
    _perpendicular_moment: FloatArray = field(init=False, repr=False)
    _quarter_turn_moment: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        angular_velocity = _vector3(
            self.angular_velocity,
            name="angular_velocity",
        )
        initial_moment = _vector3(
            self.initial_magnetic_moment,
            name="initial_magnetic_moment",
        )
        angular_speed = float(np.linalg.norm(angular_velocity))

        if angular_speed == 0.0:
            rotation_axis = np.zeros(3, dtype=np.float64)
            parallel_moment = initial_moment.copy()
            perpendicular_moment = np.zeros(3, dtype=np.float64)
            quarter_turn_moment = np.zeros(3, dtype=np.float64)
        else:
            rotation_axis = angular_velocity / angular_speed
            parallel_moment = (
                np.dot(initial_moment, rotation_axis) * rotation_axis
            )
            perpendicular_moment = initial_moment - parallel_moment
            quarter_turn_moment = np.cross(
                rotation_axis,
                perpendicular_moment,
            )

        arrays = (
            angular_velocity,
            initial_moment,
            rotation_axis,
            parallel_moment,
            perpendicular_moment,
            quarter_turn_moment,
        )
        for array in arrays:
            array.setflags(write=False)

        object.__setattr__(self, "angular_velocity", angular_velocity)
        object.__setattr__(self, "initial_magnetic_moment", initial_moment)
        object.__setattr__(self, "_angular_speed", angular_speed)
        object.__setattr__(self, "_rotation_axis", rotation_axis)
        object.__setattr__(self, "_parallel_moment", parallel_moment)
        object.__setattr__(
            self,
            "_perpendicular_moment",
            perpendicular_moment,
        )
        object.__setattr__(self, "_quarter_turn_moment", quarter_turn_moment)

    @property
    def angular_speed(self) -> float:
        """Return ``|omega|`` in radians per unit time."""
        return self._angular_speed

    @property
    def period(self) -> float:
        """Return one precession period, or infinity when ``omega`` is zero."""
        if self.angular_speed == 0.0:
            return inf
        return tau / self.angular_speed

    @property
    def rotation_axis(self) -> FloatArray:
        """Return the unit precession axis (zero when ``omega`` is zero)."""
        return self._rotation_axis.copy()

    def magnetic_moment(self, time: ArrayLike) -> FloatArray:
        """Evaluate the magnetic moment at one time or an array of times.

        A scalar input produces an array with shape ``(3,)``. An input with
        shape ``(...)`` produces magnetic moments with shape ``(..., 3)``.
        """
        times = np.asarray(time, dtype=np.float64)
        if not np.all(np.isfinite(times)):
            msg = "time must contain only finite values"
            raise ValueError(msg)

        output_shape = (*times.shape, 3)
        if self.angular_speed == 0.0:
            return np.broadcast_to(
                self.initial_magnetic_moment,
                output_shape,
            ).copy()

        angles = self.angular_speed * times
        cosines = np.cos(angles)[..., np.newaxis]
        sines = np.sin(angles)[..., np.newaxis]
        return (
            self._parallel_moment
            + cosines * self._perpendicular_moment
            + sines * self._quarter_turn_moment
        )

    def derivative(self, time: ArrayLike) -> FloatArray:
        """Evaluate ``omega x mu(time)``."""
        return np.cross(
            self.angular_velocity,
            self.magnetic_moment(time),
        )
