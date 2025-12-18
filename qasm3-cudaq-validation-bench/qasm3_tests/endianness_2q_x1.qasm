OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;

x q[1];
c[0] = measure q[0];
c[1] = measure q[1];

