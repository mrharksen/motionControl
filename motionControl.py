import timeit
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from matplotlib import animation

'''
In order for this file to run ffmpeg is needed.
It is used to save the animations to an mp4 file.
If you do not want to install ffmpeg you can comment
out all sections where it says save().
'''

'''Returns the trapzoid approximation of the integral of f over [a,b]'''
def Trap(f, a, b):
    return (f(a)+f(b))*(b-a)/2

'''Returns the Simpson approximation of the integral of f over [a,b]'''
def Simpson(f,a,b):
    return (f(a)+f(b)+4*f((a+b)/2))*(b-a)/6

'''Returns the parametirization of the Bezier curve
which is defined by the points p1, p2, p3 and p4'''
def Bezier(p1,p2,p3,p4):
    bx = 3*(p2[0]-p1[0])
    cx = 3*(p3[0]-p2[0])-bx
    dx = p4[0] - p1[0] - bx - cx
    by = 3*(p2[1]-p1[1])
    cy = 3*(p3[1]-p2[1])-by
    dy = p4[1] - p1[1] - by - cy
    a1 = lambda t: p1[0] + bx*t + cx*t**2 + dx*t**3
    a2 = lambda t: p1[1] + by*t + cy*t**2 + dy*t**3
    da1 = lambda t: bx + 2*cx*t + 3*dx*t**2
    da2 = lambda t: by + 2*cy*t + 3*dy*t**2
    return a1, a2, da1, da2

''' Doubles the length of the vector v when needed (while storing the values)'''
def lengthen(v):
    m = len(v)
    newv = np.zeros(2*m)
    for i in range(m):
        newv[i]=v[i]
    return newv

''' Finds the integral of f over [a0,b0]
within a tolerance tol0 via the Trapezoid method'''
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
        if np.abs(oldapp-app[n-1]-app[n]) < 15*tol[n-1]:
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

''' Finds the integral of f over [a0,b0]
within a tolerance tol0 via the Simpson method'''
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
    app[0] = Simpson(f, a0, b0)
    while n > 0:
        if len(app) < 2*n:
            app = lengthen(app)
            a = lengthen(a)
            b = lengthen(b)
            tol = lengthen(tol)
        c = (a[n-1] + b[n-1])/2
        oldapp = app[n-1]
        app[n-1] = Simpson(f, a[n-1], c)
        app[n] = Simpson(f, c, b[n-1])
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

''' Finds the root of f betwen [a,b]
up to a an interval [c,d] of length tol.'''
def binary(f, a, b, tol):
    if np.abs(f(a)) < tol:
        return a
    if np.abs(f(b)) < tol:
        return b
    while (b-a)/2 > tol:
        c = (a+b)/2
        if f(c) == 0:
            return c
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return c

'''Finds the root of f closes to x0 with Newtons method'''
def Newton(f,df,x0,tol):
    x = x0
    while np.abs(f(x)) > tol:
        x -= f(x)/df(x)
    return x

'''Finds the value of tstar via binary method (which is defined in the text)'''
def tstar(arc, s, tollength, tolbinary):
    l = adaptiveQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveQuadrature(arc,0,t,tollength)-l*s
    return binary(Arc, 0, 1, tolbinary)

'''Equipartitions the arc given'''
def equiPartition(arc, n, tol):
    partition = [ tstar(arc, t, tol, tol) for t in np.linspace(0,1,n+1) ]
    return partition

'''Finds the value of tstar via Newton'''
def tstar2(arc, s, tollength, tolNewton):
    l = adaptiveQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveQuadrature(arc,0,t,tollength)-l*s
    return Newton(Arc,arc, s, tolNewton)

'''Equipartitions the arc given'''
def equiPartition2(arc, n, tol):
    partition = [ tstar2(arc, t, tol, tol) for t in np.linspace(0,1,n+1) ]
    return partition

'''Finds the value of tstar via Newton with Simpson AQ integration'''
def tstar3(arc, s, tollength, tolNewton):
    l = adaptiveSimpsonsQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveSimpsonsQuadrature(arc,0,t,tollength)-l*s
    return Newton(Arc,arc, s, tolNewton)

'''Equipartitions the arc given'''
def equiPartition3(arc, n, tol):
    partition = [ tstar3(arc, t, tol, tol) for t in np.linspace(0,1,n+1) ]
    return partition

'''Plots the partition given'''
def plotCurvePartition(x, y, partition):
    t = np.linspace(0, 1, 500)
    vx = [x(s) for s in t]
    vy = [y(s) for s in t]
    xpoint = [x(s) for s in partition]
    ypoint = [y(s) for s in partition]

    plt.figure()
    plt.plot(vx, vy, label='P(t)')
    plt.plot(xpoint, ypoint, 'o', label='Skiptipunktar')
    plt.legend()
    plt.axis("equal")
    plt.title('Jafnskipting sléttuferilsins eftir bogalend')
    plt.xlabel("x-ás")
    plt.ylabel("y-ás")
    plt.show()
    return


'''Updates the point, used in the animation method.'''
def updatePoint(n, x, y, s, point):
    point.set_data(np.array([x(s[n]), y(s[n])]))
    return point,

'''Animates the curve.'''
def animateCurve(x, y, s):
    fig = plt.figure()
    # create the underlying path
    t = np.linspace(0, 1, 500)
    vx, vy = x(t), y(t)
    line, = plt.plot(vx, vy, label='P(t)')
    # plot the first point
    point, = plt.plot([vx[0]], [vy[0]], 'o')

    plt.legend()
    plt.title("Umstikun með einhalla falli af bogalengd")
    plt.xlabel("x-ás")
    plt.ylabel("y-ás")
    plt.axis('equal')

    ani=animation.FuncAnimation(fig, updatePoint, len(s), fargs=(x, y, s, point), blit=True, interval=25)

    return ani

'''Times the length of calling the function ten times.'''
def timeFunction(func, *args, **kwargs):
    def wrapped():
        return func(*args)
    return timeit.timeit(wrapped, **kwargs)

''' Finds the new progress speed for a given function c '''
def parametersFromProgressCurve(arc, c, n):
    return [tstar3(arc,c(t),tol,tol) for t in np.linspace(0,1,n)]


tol = 0.0001
TOL = 0.000001

x = lambda t: 0.5 + 0.3*t + 3.9*t**2 - 4.7*t**3
y = lambda t: 1.5 + 0.3*t + 0.9*t**2 - 2.7*t**3
dx = lambda t: 0.3 + 7.8*t - 14.1*t**2
dy = lambda t: 0.3 + 1.8*t - 8.1*t**2
arc = lambda t: np.sqrt(dx(t)**2 + dy(t)**2)
lTrap = adaptiveQuadrature(arc, 0, 1, TOL)
lSimp = adaptiveSimpsonsQuadrature(arc, 0, 1, TOL)
print("The length of the given curve is:")
print("Evaluated with the trapizoid method: "+str(lTrap))
print("Evaluated with the Simpsins method: "+str(lSimp))


s = 0.7
tstarr = tstar(arc, s, TOL, TOL)
print("The value of t*("+ str(s) + ") is: "+str(tstarr))


plotCurvePartition(x, y, equiPartition(arc, 4, tol))
plotCurvePartition(x, y, equiPartition(arc, 20, tol))


tstarr2 = tstar2(arc, s, TOL, TOL)
print("The value of t*("+ str(s) + ") computed by Newton's method is: "+str(tstarr))

plotCurvePartition(x, y, equiPartition2(arc, 4, tol))
plotCurvePartition(x, y, equiPartition2(arc, 20, tol))


tstarr3 = tstar3(arc, s, TOL, TOL)
print("The value of t*("+ str(s) + ") computed by Newton's method is: "+str(tstarr))

plotCurvePartition(x, y, equiPartition3(arc, 4, tol))
plotCurvePartition(x, y, equiPartition3(arc, 20, tol))

s=0.5
print("Time of computing t* with AQ trapizoid and bisection method:")
print(timeFunction(tstar, arc, s, tol, tol, number=10))

print("Time of computing t*2 with AQ trapizoid and Newtons method:")
print(timeFunction(tstar2, arc, s, tol, tol, number=10))

print("Time of computing t*3 with AQ Simpson and Newtons method:")
print(timeFunction(tstar3, arc, 0.5, tol, tol, number=10))

p1 = (0,0); p2 = (-1,1); p3 = (0,2); p4 = (0,1)
a1, a2, da1, da2 = Bezier(p1,p2,p3,p4)
arcBezier =  lambda t: np.sqrt(da1(t)**2 + da2(t)**2)
lBezSimp = adaptiveSimpsonsQuadrature(arcBezier, 0, 1, TOL)
print("The length of the Bézier curve is: "+ str(lBezSimp))

plotCurvePartition(a1, a2, equiPartition3(arcBezier, 4, tol))
plotCurvePartition(a1, a2, equiPartition3(arcBezier, 20, tol))

ani1 = animateCurve(x, y, np.linspace(0,1,200))
ani1.save("ani1.mp4", writer="ffmpeg", fps=30)

sVec = [tstar3(arc,t,tol,tol) for t in np.linspace(0,1,200)]
ani2 = animateCurve(x, y, sVec)
ani2.save("ani2.mp4", writer="ffmpeg", fps=30)

aniBez1 = animateCurve(a1, a2, np.linspace(0,1,200))
aniBez1.save("aniBez1.mp4", writer="ffmpeg", fps=30)

sVecBezier = [tstar3(arcBezier,t,tol,tol) for t in np.linspace(0,1,200)]
aniBez2 = animateCurve(a1, a2, sVecBezier)
aniBez2.save("aniBez2.mp4", writer="ffmpeg", fps=30)


ct1d3 = lambda t: np.power(t,1/3)
st1d3 = parametersFromProgressCurve(arc,ct1d3,300)
anit1d3 = animateCurve(x,y,st1d3)
anit1d3.save("anit1d3.mp4", writer="ffmpeg", fps=30)

ct2 = lambda t: np.power(t,2)
st2 = parametersFromProgressCurve(arc,ct2,150)
anit2 = animateCurve(x,y,st2)
anit2.save("anit2.mp4", writer="ffmpeg", fps=30)

cSin = lambda t: np.sin(t*np.pi/2)
sSin = parametersFromProgressCurve(arc,cSin,150)
aSin = animateCurve(x,y,sSin)
aSin.save("aniSin.mp4", writer="ffmpeg", fps=30)

cSin2 = lambda t: 0.5 + 0.5*np.sin((2*t-1)*np.pi/2)
sSin2 = parametersFromProgressCurve(arc,cSin2,150)
aSin2 = animateCurve(x,y,sSin2)
aSin2.save("aniSin2.mp4", writer="ffmpeg", fps=30)

Hx = lambda t: 16*(np.sin(2*np.pi*t))**3
Hy = lambda t: 13*np.cos(2*np.pi*t)-5*np.cos(4*np.pi*t)-2*np.cos(6*np.pi*t)-np.cos(8*np.pi*t)
Hdx = lambda t: 48*((np.sin(2*np.pi*t))**2)*np.cos(2*np.pi*t)
Hdy = lambda t: -13*np.sin(2*np.pi*t)+10*np.sin(4*np.pi*t)+6*np.sin(6*np.pi*t)+4*np.cos(8*np.pi*t)
Harc = lambda t: np.sqrt(Hdx(2*np.pi*t)**2 + Hdy(2*np.pi*t)**2)

heart = animateCurve(Hx, Hy, np.linspace(0,1,200))
heart.save("heart.mp4", writer="ffmpeg", fps=30)
