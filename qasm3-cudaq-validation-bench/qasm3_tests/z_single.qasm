OPENQASM 3;
include "stdgates.inc";
qubit[1] q;
bit[1] c;

z q[0];
c[0] = measure q[0];
