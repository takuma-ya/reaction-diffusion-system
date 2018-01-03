import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im_org  = np.array(Image.open('binary5/8b.png'))#.convert('L'))
#im = im_org[:,142:]
im = im_org[100:200,300:400]
im = im.astype(np.float64)
im = im/255
print(im.shape, im.dtype)
scale = 1.

p_a = 0.9
p_e = 6.5

p_b = 0.8
d = 30.0
p_h = 1.0
Tr = 1.0
Trc = 0.4
T = 1 
A_max = 30.0
R = 0.004 / scale
gamma = 625. * scale 
dx = 0.02 / scale
dt = 0.000001 * scale

size = 100  # size of the 2D grid
n = int(T/dt)
radius = gamma * R * R / (dx * dx) /scale
print(radius)

mask = im == 1
U = np.ones((size,size))/2. + np.random.rand(size,size)/100.
U[np.logical_not(mask)] = 0
V = np.random.rand(size,size)/100
V[mask] = V[mask] + 0.1
V[np.logical_not(mask)] = 0
C = im

def laplacian(Z):
    Ztop = Z[0:-2,1:-1]
    Zleft = Z[1:-1,0:-2]
    Zbottom = Z[2:,1:-1]
    Zright = Z[1:-1,2:]
    Zcenter = Z[1:-1,1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

def cellspace(U,C):
    cell = np.ones((100, 100), dtype=bool)
    n = 100
    for a in range(size):
        for b in range(size):
            y,x = np.ogrid[-a:n-a, -b:n-b]
            mask = x*x + y*y <= radius 
            for m in C[mask]:
                if m > Trc:
                    cell[a,b] = False 
                    break
    U[cell] = 0.
    return U

# We simulate the PDE with the finite difference method.
for i in range(n):
    # set activator 0 outside the cell
    U = cellspace(U,C)
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1,1:-1]
    Vc = V[1:-1,1:-1]
    Cc = C[1:-1,1:-1]
    f = p_a * Uc + Uc * Uc - p_b * Uc * Vc
    g = p_e * Uc * Uc * Uc - Vc
    # We update the variables.
    U[1:-1,1:-1], V[1:-1,1:-1] = \
        Uc + dt * (deltaU + gamma * f), \
        Vc + dt * (d * deltaV + gamma * g) 
    for a in range(size-2):
        for b in range(size-2):
            if U[a+1,b+1] > Tr:
                C[a+1,b+1] = C[a+1,b+1] + dt * (gamma * p_h * C[a+1,b+1] * ((Trc - 2.5 * (U[a+1,b+1] - Tr)) - C[a+1,b+1]) * (C[a+1,b+1] -1.)) 
            else:
                C[a+1,b+1] = C[a+1,b+1] + dt * (gamma * p_h * C[a+1,b+1] * (Trc - C[a+1,b+1]) * (C[a+1,b+1] -1.)) 
            if C[a+1,b+1] < 0.01:
                C[1:-1,1:-1] = Cc + np.random.rand(size-2,size-2)/100 #add small noise
    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0,:] = Z[1,:]
        Z[-1,:] = Z[-2,:]
        Z[:,0] = Z[:,1]
        Z[:,-1] = Z[:,-2]

    if i % 10 == 0:
        print(U[50,50])
        plt.imshow(C, cmap="Greys", vmin=0., vmax=1.);
        plt.xticks([]); plt.yticks([]);
        #plt.savefig("cell_c/cell"+str(i/10)+".png")
        #plt.imshow(U, cmap="Greys", vmin=0., vmax=1.);
        #plt.xticks([]); plt.yticks([]);
        #plt.savefig("cell_a/cell"+str(i/10)+".png")
        plt.show()
