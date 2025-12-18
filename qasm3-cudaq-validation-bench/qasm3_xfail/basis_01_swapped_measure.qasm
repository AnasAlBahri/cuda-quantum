OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;

x q[1];            // state |01> if q[0] is MSB in your mental model
c[0] = measure q[1];
c[1] = measure q[0];
