from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt


Vector3 = tuple[float, float, float]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class RigidTransform:
    parent_frame: str
    child_frame: str
    matrix: Matrix4

    def validate(self, *, label: str = "transform") -> None:
        if not isinstance(self.parent_frame, str) or not self.parent_frame.strip():
            raise ValueError(f"{label} parent_frame must be a non-empty string")
        if not isinstance(self.child_frame, str) or not self.child_frame.strip():
            raise ValueError(f"{label} child_frame must be a non-empty string")
        _validate_matrix(self.matrix, label=label)
        if self.parent_frame == self.child_frame:
            identity = tuple(
                tuple(1.0 if row == column else 0.0 for column in range(4))
                for row in range(4)
            )
            if any(
                abs(self.matrix[row][column] - identity[row][column]) > 1e-8
                for row in range(4)
                for column in range(4)
            ):
                raise ValueError(f"{label} with identical frames must be identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "matrix": [list(row) for row in self.matrix],
        }


def compose(first: RigidTransform, second: RigidTransform) -> RigidTransform:
    first.validate(label="first transform")
    second.validate(label="second transform")
    if first.child_frame != second.parent_frame:
        raise ValueError(
            "cannot compose transforms with disconnected frames: "
            f"{first.child_frame!r} != {second.parent_frame!r}"
        )
    matrix = tuple(
        tuple(
            sum(first.matrix[row][k] * second.matrix[k][column] for k in range(4))
            for column in range(4)
        )
        for row in range(4)
    )
    result = RigidTransform(first.parent_frame, second.child_frame, matrix)  # type: ignore[arg-type]
    result.validate(label="composed transform")
    return result


def inverse(transform: RigidTransform) -> RigidTransform:
    transform.validate()
    rotation = tuple(
        tuple(transform.matrix[column][row] for column in range(3))
        for row in range(3)
    )
    translation = tuple(transform.matrix[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation[row][column] * translation[column] for column in range(3))
        for row in range(3)
    )
    matrix = tuple(
        tuple(rotation[row][column] for column in range(3))
        + (inverse_translation[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    return RigidTransform(transform.child_frame, transform.parent_frame, matrix)  # type: ignore[arg-type]


def rotate_transform_about_parent_axis(
    transform: RigidTransform,
    axis: Vector3,
    angle_rad: float,
) -> RigidTransform:
    transform.validate()
    yaw = _axis_angle_matrix(axis, angle_rad)
    matrix = tuple(
        tuple(
            sum(yaw[row][k] * transform.matrix[k][column] for k in range(3))
            if column < 3
            else transform.matrix[row][3]
            for column in range(4)
        )
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    result = RigidTransform(transform.parent_frame, transform.child_frame, matrix)  # type: ignore[arg-type]
    result.validate(label="yaw-rotated transform")
    return result


def reframe(
    transform: RigidTransform,
    *,
    parent_frame: str | None = None,
    child_frame: str | None = None,
) -> RigidTransform:
    result = RigidTransform(
        parent_frame if parent_frame is not None else transform.parent_frame,
        child_frame if child_frame is not None else transform.child_frame,
        transform.matrix,
    )
    result.validate()
    return result


def _axis_angle_matrix(axis: Vector3, angle_rad: float) -> Matrix4:
    values = tuple(float(value) for value in axis)
    if len(values) != 3 or not all(isfinite(value) for value in values):
        raise ValueError("rotation axis must contain three finite values")
    norm = sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("rotation axis must be normalized")
    if not isfinite(angle_rad):
        raise ValueError("rotation angle must be finite")
    x, y, z = values
    c = cos(angle_rad)
    s = sin(angle_rad)
    one_minus_c = 1.0 - c
    return (
        (
            c + x * x * one_minus_c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
            0.0,
        ),
        (
            y * x * one_minus_c + z * s,
            c + y * y * one_minus_c,
            y * z * one_minus_c - x * s,
            0.0,
        ),
        (
            z * x * one_minus_c - y * s,
            z * y * one_minus_c + x * s,
            c + z * z * one_minus_c,
            0.0,
        ),
        (0.0, 0.0, 0.0, 1.0),
    )


def _validate_matrix(matrix: Matrix4, *, label: str) -> None:
    if not isinstance(matrix, tuple) or len(matrix) != 4:
        raise ValueError(f"{label} matrix must be 4x4")
    if any(not isinstance(row, tuple) or len(row) != 4 for row in matrix):
        raise ValueError(f"{label} matrix must be 4x4")
    if not all(isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{label} matrix must contain only finite values")
    if any(abs(matrix[3][column] - expected) > 1e-9 for column, expected in enumerate((0, 0, 0, 1))):
        raise ValueError(f"{label} matrix must have homogeneous final row [0, 0, 0, 1]")

    for row in range(3):
        for column in range(3):
            dot = sum(matrix[k][row] * matrix[k][column] for k in range(3))
            expected = 1.0 if row == column else 0.0
            if abs(dot - expected) > 1e-6:
                raise ValueError(f"{label} rotation must be orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant - 1.0) > 1e-6:
        raise ValueError(f"{label} rotation determinant must be +1")
