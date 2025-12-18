OPENQASM 3;
include "stdgates.inc";
qubit[1] a;
qubit[2] b;
bit[3] c;

x a[0];
h b[0];
cx b[0], b[1];
cx a[0], b[0];

c[0] = measure a[0];
c[1] = measure b[0];
c[2] = measure b[1];
