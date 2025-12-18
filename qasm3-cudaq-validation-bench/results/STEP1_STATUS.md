# Step 1: Minimal QASM3 test suite (fallback_qiskit)

## Core (must-pass)
- bell.qasm
- ghz3.qasm
- h_single.qasm
- x_single.qasm
- z_single.qasm
- s_single.qasm
- t_single.qasm
- rx_pi_over_2_single.qasm
- rz_interference_pi_over_3.qasm

## XFAIL (expected failures / known issues)
- reset_single.qasm: fallback_qiskit does not support `reset`
- two_qregs_mixed_ops.qasm: triggers invalid/self-controlled CX or runtime exception; isolate + debug
- basis_01_swapped_measure.qasm: bitstring ordering / measurement mapping diagnostic (handled in Step 2)
