import numpy as np
import matplotlib.pyplot as plt

def fill_matrix_poisson(A,b,nx,ny):
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

    # Inside
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            k = i+nx*j
            A[k,k-nx] = -Sy
            A[k,k-1] = -Sx
            A[k,k] = (2*Sx+2*Sy)
            A[k,k+1] = -Sx
            A[k,k+nx] = -Sy
            b[k] = 10


def fill_adv_diff(Pe,A,b,nx,ny):
    dx = 1./ (nx-1)
    dy = 1./ (ny-1)
    Sx = 1. / dx**2
    Sy = 1. / dy**2
    UX = Pe / 2 / dx
    UY = Pe / 2 / dy
    
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

    # Inside
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            k = i+nx*j
            A[k,k-nx] = -Sy-UY
            A[k,k-1] = -Sx-UX
            A[k,k] = (2*Sx+2*Sy)
            A[k,k+1] = -Sx+UX
            A[k,k+nx] = -Sy+UY
            b[k] = 10

def calc_residual_norm(A,b,x):
    res = np.dot(A,x) - b
    return np.linalg.norm(res)


def jacobi(A,b,tol):
    N = b.size
    x = np.zeros(b.size)
    
    it=0
    err=[]
    err.append(calc_residual_norm(A,b,x))


    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for N times                               s                                                                                                                                           
    while (err[-1] > tol ):
        x = (b - np.dot(R,x)) / D
        it = it+1
        err.append(calc_residual_norm(A,b,x))

    return x, err

def gauss_seidel(A,b,tol):
    N = b.size
    x = np.zeros(b.size)
    
    it=0
    err=[]
    err.append(calc_residual_norm(A,b,x))


    # Iterate for N times                               s                                                                                                                                           
    while (err[-1] > tol ):
        for i in range(0,N):
            x[i] = (b[i] - np.dot(A[i,0:i],x[0:i]) - np.dot(A[i,i+1:],x[i+1:])) / A[i,i]
        it = it+1
        err.append(calc_residual_norm(A,b,x))

    return x, err

def conjugate_gradient(A,b,tol):
    N = b.size
    it=0
    err=[]

    x = np.zeros(b.size)
    err.append(calc_residual_norm(A,b,x))

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
        err.append(calc_residual_norm(A,b,x))

    return x , err

def arnoldi_single_iter(A, Q, k) :
    """Compute a single iteration of Arnoldi
    """
    q = A.dot(Q[:,k])
    h = np.zeros(k+2)
    for i in range(k+1):
        h[i] = q.T.dot(Q[:,i])
        q -= h[i]*Q[:,i]
    h[k+1] = np.linalg.norm(q)
    q /= h[k+1]
    return h,q

def givens_coeffs(a,b):
    """ """
    c = a / np.sqrt(a**2 + b**2)
    s = b / np.sqrt(a**2 + b**2)
    return c, s

def gmres(A, b, tol) :
    """Solve linear system via the Generalized Minimal Residual Algorithm (GMRES).

    Args:
        A: Square matrix of shape (n,n) (must be nonsingular).
        b: Right-hand side
        tol: Tolerance of the system to be reached

    Returns:
        x_k: Vector of shape (n,1) representing converged solution for x.
        err: list of size (it) corresponding of the evolution of the error through the iterations

    """
    x = np.zeros(b.size)
    max_iters = 50
    n = b.size


    err=[]
    err.append(calc_residual_norm(A,b,x))

    r = b - A.dot(x)
    q = r / np.linalg.norm(r)
    Q = np.zeros((n,max_iters))
    Q[:,0] = q.squeeze()
    beta = np.linalg.norm(r)
    xi = np.zeros((n,1))
    xi[0] = 1 # e_1 standard basis vector, xi will be updated
    H = np.zeros((n+1,n))

    F = np.zeros((max_iters,n,n))
    for i in range(max_iters):
        F[i] = np.eye(n)

    for k in range(max_iters-1):
        H[:k+2,k], Q[:,k+1] = arnoldi_single_iter(A,Q,k)

        # Don't need to do this for 0,...,k since completed
        c,s = givens_coeffs(H[k,k], H[k+1,k])
        # kth rotation matrix
        F[k, k,k] = c
        F[k, k,k+1] = s
        F[k, k+1,k] = -s
        F[k, k+1,k+1] = c

        # apply the rotation to both of these
        H[:k+2,k] = F[k,:k+2,:k+2].dot(H[:k+2,k])
        xi = F[k].dot(xi)
        err.append(beta * np.linalg.norm(xi[k+1]))

        if beta * np.linalg.norm(xi[k+1]) < tol:
            break

    # When terminated, solve the least squares problem.
    # `y` must be (k,1).
    y, _, _, _ = np.linalg.lstsq(H[:k+1,:k+1],xi[:k+1])
    # `Q_k` will have dimensions (n,k).
    x_k = x + Q[:,:k+1].dot(y)
    print(x_k)
    return x_k, err


def run_poisson(nx,ny):
  n = nx*ny
  A = np.zeros((n,n)) 
  b = np.zeros([n])
  fill_matrix_poisson(A,b,nx,ny)
  T = np.linalg.solve(A,b)
  
  T_jac, err_jac=jacobi(A,b,1e-3)
  T_gs, err_gs  =gauss_seidel(A,b,1e-3)
  T_cg, err_cg  =conjugate_gradient(A,b,1e-3)
  T_gmres, err_gmres  =gmres(A,b,1e-3)
  
  
  plt.semilogy(err_jac,label="Jacobi")
  plt.semilogy(err_gs,label="Gauss-Seidel")
  plt.semilogy(err_cg,label="Conjugate Gradient")
  plt.semilogy(err_gmres,label="GMRES")
  
  plt.legend()
  plt.show()
  
  T_reshaped = T_gmres.reshape(nx,ny).transpose()
  #
  plt.contourf(T_reshaped)
  plt.colorbar()
  plt.show()
  
def run_adv_diff(Pe,nx,ny):
  n = nx*ny
  A = np.zeros((n,n)) 
  b = np.zeros([n])
  fill_adv_diff(Pe,A,b,nx,ny)
  T = np.linalg.solve(A,b)
  
  T_jac, err_jac=jacobi(A,b,1e-3)
  T_gs, err_gs  =gauss_seidel(A,b,1e-3)
  T_cg, err_cg  =conjugate_gradient(A,b,1e-3)
  T_gmres, err_gmres  =gmres(A,b,1e-3)
  
  
  plt.semilogy(err_jac,label="Jacobi")
  plt.semilogy(err_gs,label="Gauss-Seidel")
  plt.semilogy(err_cg,label="Conjugate Gradient")
  plt.semilogy(err_gmres,label="GMRES")
  
  plt.legend()
  plt.show()
  
  T_reshaped = T.reshape(nx,ny).transpose()
  #
  plt.contourf(T_reshaped)
  plt.colorbar()
  plt.show()

run_adv_diff(10,25,25)


