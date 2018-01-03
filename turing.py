import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im_org  = np.array(Image.open('binary5/30b.png').convert('L'))
im = im_org[100:200,300:400]
im = im.astype(np.float64)
im = im/255
print(im.shape, im.dtype)

a = 2.8e-4
b = 5e-3
tau = .1
k = -.005

size = 100  # size of the 2D grid
dx = 2./size  # space step
T = 20.0  # total time
dt =.9 * dx**2/2  # time step
n = int(T/dt)
print(n)

U = im/2. + np.random.rand(size, size)/100
V = 0.1 + np.random.rand(size, size)/100

def laplacian(Z):
    Ztop = Z[0:-2,1:-1]
    Zleft = Z[1:-1,0:-2]
    Zbottom = Z[2:,1:-1]
    Zright = Z[1:-1,2:]
    Zcenter = Z[1:-1,1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

# We simulate the PDE with the finite difference method.
for i in range(n):
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1,1:-1]
    Vc = V[1:-1,1:-1]
    # We update the variables.
    U[1:-1,1:-1], V[1:-1,1:-1] = \
        Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k), \
        Vc + dt * (b * deltaV + Uc - Vc) / tau
    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0,:] = Z[1,:]
        Z[-1,:] = Z[-2,:]
        Z[:,0] = Z[:,1]
        Z[:,-1] = Z[:,-2]

    if i % 10000 == 0:
        plt.imshow(U, cmap="Greys", vmin=0., vmax=1.)#, extent=[-1,1,-1,1]);
        plt.xticks([]); plt.yticks([])
        #plt.savefig("turing/turing"+str(i/10000)+".png")
        plt.show()
