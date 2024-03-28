from sympy import Piecewise, Abs, Rational, simplify, lambdify, pprint
from sympy.printing.rust import rust_code
from sympy.abc import x
import numpy
import matplotlib.pyplot as plt

B = Rational('1/3')
C = Rational('1/3')

m = Abs(x)
f = ((12 - 9 * B - 6 * C) * (m**3) + (-18 + 12 * B + 6 * C) * (m**2) + (6 - 2 * B)) / 6
g = (
    (-B - 6 * C) * (m**3)
    + (6 * B + 30 * C) * (m**2)
    + (-12 * B - 48 * C) * (m)
    + (8 * B + 24 * C)
) / 6

k = Piecewise((f, m < 1.0), (g, (m >= 1.0) & (m < 2.0)), (0, True))

h = simplify(k)

pprint(rust_code(h))

mitchell = lambdify(x, h, modules='numpy')

# Plot the function
x = numpy.linspace(-3, 3, 1001)
y = mitchell(x)

plt.plot(x, y)
plt.show()
