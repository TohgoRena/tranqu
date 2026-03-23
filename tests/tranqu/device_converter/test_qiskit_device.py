from dataclasses import dataclass

import pytest
from qiskit.transpiler import Target  # type: ignore[import-untyped]

from tranqu.device_converter import QiskitDevice


@dataclass
class FakeRunInput:
    num_qubits: int = 2
    size: int = 1


def test_max_circuits_is_not_supported() -> None:
    device = QiskitDevice("test_device", Target())

    with pytest.raises(
        NotImplementedError,
        match=r"'max_circuits' function is not supported\.",
    ):
        _ = device.max_circuits


def test_run_error_includes_input_summary_and_option_count() -> None:
    device = QiskitDevice("test_device", Target())

    with pytest.raises(NotImplementedError, match="test_device") as exc_info:
        device.run(FakeRunInput(), shots={"value": 1024})

    message = str(exc_info.value)
    assert "FakeRunInput num_qubits=2 size=1" in message
    assert "options: 1 keys" in message
