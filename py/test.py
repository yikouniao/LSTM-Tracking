import numpy as np
from numpy import array
from numpy import append, reshape

a = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
print(len(a))

b = array([])

# for shit in a:
#     append(b, reshape(shit, (6, 1)))
a = reshape(a, (len(a), 6, 1))
print(a[0])