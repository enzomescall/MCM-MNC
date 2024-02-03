import numpy as np

# 3dim vectotr
extent = np.array([10, 10, 10])

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def force(a,b,c):
    return (a, b, c)

F = np.atleast_2d(force(1,2,3))

for i in range(extent.size):
    print(np.roll(F[i], -1, axis=i))    