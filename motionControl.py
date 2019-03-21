import timeit
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from matplotlib import animation


def Trap(f, a, b):
    return (f(a)+f(b))*(b-a)/2

def Simpson(f,a,b):
    return 5


def lengthen(v):
    m = len(v)
    newv = np.zeros(2*m)
    for i in range(m):
        newv[i]=v[i]
    return newv


def adaptiveQuadrature(f, a0, b0, tol0):
    sum = 0
    n = 1
    a = np.zeros(1000)
    b = np.zeros(1000)
    app = np.zeros(1000)
    tol = np.zeros(1000)
    a[0] = a0
    b[0] = b0
    tol[0] = tol0
    app[0] = Trap(f, a0, b0)
    while n > 0:
        if len(app) < 2*n:
            app = lengthen(app)
            a = lengthen(a)
            b = lengthen(b)
            tol = lengthen(tol)
        c = (a[n-1] + b[n-1])/2
        oldapp = app[n-1]
        app[n-1] = Trap(f, a[n-1], c)
        app[n] = Trap(f, c, b[n-1])
        if np.abs(oldapp-app[n-1]-app[n]) < 3*tol[n-1]:
            sum += app[n-1]+app[n]
            n -= 1
        else:
            b[n] = b[n-1]
            b[n-1] = c
            a[n] = c
            tol[n-1] /=2
            tol[n] = tol[n-1]
            n += 1
    return sum

def adaptiveSimpsonsQuadrature(f, a0, b0, tol0):
    sum = 0
    n = 1
    a = np.zeros(1000)
    b = np.zeros(1000)
    app = np.zeros(1000)
    tol = np.zeros(1000)
    a[0] = a0
    b[0] = b0
    tol[0] = tol0
    app[0] = Trap(f, a0, b0)
    while n > 0:
        if len(app) < 2*n:
            app = lengthen(app)
            a = lengthen(a)
            b = lengthen(b)
            tol = lengthen(tol)
        c = (a[n-1] + b[n-1])/2
        oldapp = app[n-1]
        app[n-1] = Trap(f, a[n-1], c)
        app[n] = Trap(f, c, b[n-1])
        if np.abs(oldapp-app[n-1]-app[n]) < 3*tol[n-1]:
            sum += app[n-1]+app[n]
            n -= 1
        else:
            b[n] = b[n-1]
            b[n-1] = c
            a[n] = c
            tol[n-1] /=2
            tol[n] = tol[n-1]
            n += 1
    return sum


def binary(f, a, b, tol):
    while (b-a)/2 > tol:
        c = (a+b)/2
        if f(c) == 0:
            return c
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return c

def Newton(f,df,x0,tol):
    x = x0
    while np.abs(f(x)) > tol:
        x -= f(x)/df(x)
    return x


def tstar(arc, s, tollength, tolbinary):
    l = adaptiveQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveQuadrature(arc,0,t,tollength)-l*s
    return binary(Arc, 0, 1, tolbinary)

def tstar2(arc, s, tollength, tolNewton):
    l = adaptiveQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveQuadrature(arc,0,t,tollength)-l*s
    return Newton(Arc,arc, s, tolNewton)

def updatePoint(n, x, y, point):
    point.set_data(np.array([x[n], y[n]]))
    return point

def animateCurve(x,y,s):
    fig = plt.figure()
    vx=x(s)
    vy=y(s)
    # create the first plot
    point, = plt.plot([vx[0]], [vy[0]], 'o')
    line, = plt.plot(vx, vy, label='parametric curve')
    plt.legend()

    ani=animation.FuncAnimation(fig, updatePoint, len(s), fargs=(vx, vy, point), blit=True, interval=5)
    plt.show()

    return ani

def timeFunction(func, *args, **kwargs):
    def wrapped():
        return func(*args)
    return timeit.timeit(wrapped, **kwargs)


x = lambda t: 0.5 + 0.3*t + 3.9*t**2 - 4.7*t**3
y = lambda t: 1.5 + 0.3*t + 0.9*t**2 - 2.7*t**3
dx = lambda t: 0.3 + 7.8*t - 14.1*t**2
dy = lambda t: 0.3 + 1.8*t - 8.1*t**2
c = lambda t: t
arc = lambda t: np.sqrt(dx(t)**2 + dy(t)**2)
f = lambda t: t**2
val = adaptiveQuadrature(arc,0,0.2, 0.000001)
print(val)
l = adaptiveQuadrature(arc, 0, 1, 0.00001)
s = 0.7
tstarr = tstar(arc, s, 0.000001, 0.000001)
tstarr2 = tstar2(arc, s, 0.0001, 0.0001)
print(tstarr)
print(tstarr2)
print(s*l)
print(adaptiveQuadrature(arc,0, tstarr, 0.0001))
print(adaptiveQuadrature(arc,0, tstarr2, 0.0001))
animateCurve(x, y, np.linspace(0,1,100))
