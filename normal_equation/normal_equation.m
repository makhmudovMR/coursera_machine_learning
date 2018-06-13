% noraml equation

X = [1 23 100; 1 21 120; 1 18 101]
y = [10; 9; 7]

answer = pinv(X' * X) * X' * y