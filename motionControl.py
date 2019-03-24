import timeit
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from matplotlib import animation


def Trap(f, a, b):
    return (f(a)+f(b))*(b-a)/2

def Simpson(f,a,b):
    return (f(a)+f(b)+4*f((a+b)/2))*(b-a)/6

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

def Newton(f,df,x0,tol):
    x = x0
    while np.abs(f(x)) > tol:
        x -= f(x)/df(x)
    return x

def tstar(arc, s, tollength, tolbinary):
    l = adaptiveQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveQuadrature(arc,0,t,tollength)-l*s
    return binary(Arc, 0, 1, tolbinary)

def equiPartition(arc, n, tol):
    partition = [ tstar(arc, t, tol, tol) for t in np.linspace(0,1,n+1) ]
    return partition

def tstar2(arc, s, tollength, tolNewton):
    l = adaptiveQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveQuadrature(arc,0,t,tollength)-l*s
    return Newton(Arc,arc, s, tolNewton)

def equiPartition2(arc, n, tol):
    partition = [ tstar2(arc, t, tol, tol) for t in np.linspace(0,1,n+1) ]
    return partition

def tstar3(arc, s, tollength, tolNewton):
    l = adaptiveSimpsonsQuadrature(arc, 0, 1, tollength)
    Arc = lambda t: adaptiveSimpsonsQuadrature(arc,0,t,tollength)-l*s
    return Newton(Arc,arc, s, tolNewton)

def equiPartition3(arc, n, tol):
    partition = [ tstar3(arc, t, tol, tol) for t in np.linspace(0,1,n+1) ]
    return partition

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
    plt.title('Jafnskipting sléttuferilsins eftir bogalend')
    plt.xlabel("x-ás")
    plt.ylabel("y-ás")
    plt.show()
    return

def updatePoint(n, x, y, point):
    point.set_data(np.array([x[n], y[n]]))
    return point,

def animateCurve(x, y, s):
    fig = plt.figure()
    vx=[x(t) for t in s]
    vy=[y(t) for t in s]
    point, = plt.plot([vx[0]], [vy[0]], 'o')
    line, = plt.plot(vx, vy, label='P(t)')
    plt.legend()
    plt.title("Umstikun með einhalla falli af bogalengd")
    plt.xlabel("x-ás")
    plt.ylabel("y-ás")

    ani=animation.FuncAnimation(fig, updatePoint, len(s), fargs=(vx, vy, point), blit=True, interval=25)
    ani.save('/tmp/animation.gif', writer='imagemagick', fps=30)

    return ani

def timeFunction(func, *args, **kwargs):
    def wrapped():
        return func(*args)
    return timeit.timeit(wrapped, **kwargs)



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

a1=animateCurve(x, y, np.linspace(0,1,200))


p1 = (0,0); p2 = (-1,1); p3 = (0,2); p4 = (0,1)
a1, a2, da1, da2 = Bezier(p1,p2,p3,p4)
arcBezier =  lambda t: np.sqrt(da1(t)**2 + da2(t)**2)
lBezSimp = adaptiveSimpsonsQuadrature(arcBezier, 0, 1, TOL)
print("The length of the Bézier curve is: "+ str(lBezSimp))





#f = lambda t: t**2
#val = adaptiveQuadrature(arc,0,0.2, 0.000001)
#print(val)
#l = adaptiveQuadrature(arc, 0, 1, 0.00001)
#print(l)
#l2 = adaptiveSimpsonsQuadrature(arc, 0, 1, 0.00001)
#print(l2)



#print(s*l)
#print(adaptiveQuadrature(arc,0, tstarr, 0.0001))
#print(adaptiveQuadrature(arc,0, tstarr2, 0.0001))
##a1=animateCurve(x, y, np.linspace(0,1,200))
#s2=[tstar3(arc,t,0.0001,0.0001) for t in np.linspace(0,1,200)]
#a2=animateCurve(x, y, s2)
#s3=[tstar3(arc,np.sin(t*np.pi/2),0.0001,0.0001) for t in np.linspace(0,1,300)]
#a3=animateCurve(x, y, s3)


#plotCurvePartition(a1, a2, partitionBez)







#x = lambda t: 4*np.cos(-t*(2*np.pi)) + np.cos(5*t*(2*np.pi))
#y = lambda t: 4*np.sin(-t*(2*np.pi)) + np.sin(5*t*(2*np.pi))
#dx = lambda t: 8*np.pi*np.sin(-t*2*np.pi) - 10*np.pi*np.sin(10*t*np.pi)
#dy = lambda t: -8*np.pi*np.cos(-t*2*np.pi) + 10*np.pi*np.cos(10*t*np.pi)
#arc = lambda t: np.sqrt(dx(t)**2 + dy(t)**2)
#s4=[tstar3(arc,t,0.0001,0.0001) for t in np.linspace(0,1,200)]
#a4=animateCurve(x, y, s4)
