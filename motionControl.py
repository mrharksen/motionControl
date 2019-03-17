import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

def Trapezoid(f,a,b,m):
    h = (b-a)/m
    Grid = np.arange(a,b+h,h)
    intf = -h*(f(a)+f(b))/2
    for xi in Grid:
        intf += h*f(xi)
    return intf

def adaptiveQuadrature(f, a0, b0, tol0):
    sum = 0
    n = 1
    a = [a0]
    b = [b0]
    tol = [tol0]
    app = [Trapezoid(f,a,b,1)]
    while n > 0:
        c = (a[n-1] + b[n-1])/2
        oldapp = app(n-1)
        app(n) =
    return 5


x = lambda t: 0.5 + 0.3*t + 3.9*t**2 - 4.7*t**3
y = lambda t: 0.5 + 0.3*t + 0.9*t**2 - 2.7*t**3
dx = lambda t: 0.3 + 7.8*t - 14.1*t**2
dy = lambda t: 0.3 + 1.8*t - 8.1*t**2
arc = lambda t: np.sqrt(dx(t)**2 + dy(t)**2)
val = adaptiveQuadrature(arc,0,0.5,0.0001)
print(val)
