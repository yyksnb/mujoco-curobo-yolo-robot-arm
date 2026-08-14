from __future__ import annotations

from ..contracts.execution import ControlCommand, RuntimeObservation, SceneObjectBinding
from .errors import AdapterUnavailableError


class HardwareExecutorAdapter:
    adapter_id = "hardware_unimplemented"

    def _unavailable(self) -> None:
        raise AdapterUnavailableError(
            "the real-hardware executor adapter interface is reserved but has no control policy"
        )

    def initialize(
        self,
        scene_objects: tuple[SceneObjectBinding, ...],
        target_object_id: str,
        initial_command: ControlCommand,
    ) -> RuntimeObservation:
        self._unavailable()

    def apply_control(self, command: ControlCommand) -> None:
        self._unavailable()

    def advance(self, duration_s: float) -> RuntimeObservation:
        self._unavailable()

    def observe(self) -> RuntimeObservation:
        self._unavailable()

    def close(self) -> None:
        return None
