% inverse
A = [54 3; 10 8]

IofA = pinv(A)

E = A * IofA

% transpose

A_t = A'