import numpy as np
import matplotlib.pyplot as plt

def fill_matrix(A,b,nx,ny):
    N = nx*ny
    dx = 1./ (nx-1)
    dy = 1./ (ny-1)
    Sx = 1 / dx**2
    Sy = 1 / dy**2
    
    # Boundary conditions bottom
    for i in range(0,nx):
        A[i,i] = 1
        b[i] = 0
    
    # Boundary condition top
    for i in range(0,nx):
        k = (ny-1)*nx+i
        A[k,k] = 1
        b[k] = 0

    # Boundary condition left
    for j in range(0,ny):
        k = nx*(j)
        A[k,k] = 1
        b[k] = 0

    # Boundary condition right
    for j in range(0,ny):
        k = (nx-1)+nx*(j-1)
        A[k,k] = 1
        b[k] = 0      

    # Intérieur
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            k = i+nx*j
            A[k,k-nx] = -Sy
            A[k,k-1] = -Sx
            A[k,k] = (2*Sx+2*Sy)
            A[k,k+1] = -Sx
            A[k,k+nx] = -Sy
            b[k] = 10

def calc_residual(A,b,x):
    res = np.dot(A,x) - b
    return np.linalg.norm(res)


def jacobi(A,b,tol):
    N = b.size
    x = np.zeros(b.size)
    
    it=0
    err=[]
    err.append(calc_residual(A,b,x))


    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for N times                               s                                                                                                                                           
    while (err[-1] > tol ):
        x = (b - np.dot(R,x)) / D
        it = it+1
        err.append(calc_residual(A,b,x))

    return x, err

def gauss_seidel(A,b,tol):
    N = b.size
    x = np.zeros(b.size)
    
    it=0
    err=[]
    err.append(calc_residual(A,b,x))


    # Iterate for N times                               s                                                                                                                                           
    while (err[-1] > tol ):
        for i in range(0,N):
            x[i] = (b[i] - np.dot(A[i,0:i],x[0:i]) - np.dot(A[i,i+1:],x[i+1:])) / A[i,i]
        it = it+1
        err.append(calc_residual(A,b,x))

    return x, err

def conjugate_gradient(A,b,tol):
    N = b.size
    it=0
    err=[]

    x = np.zeros(b.size)
    err.append(calc_residual(A,b,x))

    r = b - A.dot(x)
    p = r.copy()

    for i in range(N):
        Ap = A.dot(p)
        alpha = np.dot(p,r)/np.dot(p,Ap)
        x = x + alpha*p
        r = b - A.dot(x)
        if np.sqrt(np.sum((r**2))) < tol:
            print('Itr:', i)
            break
        else:
            beta = -np.dot(r,Ap)/np.dot(p,Ap)
            p = r + beta*p
        it = it+1
        err.append(calc_residual(A,b,x))

    return x , err

nx = 25
ny = 25
n = nx*ny
A = np.zeros((n,n)) 
b = np.zeros([n])
fill_matrix(A,b,nx,ny)
T = np.linalg.solve(A,b)

T_jac, err_jac=jacobi(A,b,1e-3)
T_gs, err_gs  =gauss_seidel(A,b,1e-3)
T_cg, err_cg  =conjugate_gradient(A,b,1e-3)

plt.semilogy(err_jac,label="Jacobi")
plt.semilogy(err_gs,label="Gauss-Seidel")
plt.semilogy(err_cg,label="Conjugate Gradient")

plt.legend()
plt.show()

#T_reshaped = T_jac.reshape(nx,ny).transpose()
#
#plt.contourf(T_reshaped)
#plt.colorbar()
#plt.show()