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


def plot(residual,label):
    print("Plotting")
    

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

    G = np.identity(N) - np.linalg.inv(np.diagflat(D)).dot(A)
    eigs = np.linalg.eigvals(G)
    print("Jacobi - spectral radius is : ", np.max(np.abs(eigs)))

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

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diagflat(np.diag(A))
    E = -np.triu(A, k=1)
    G = np.identity(N) - np.linalg.inv(D-E).dot(A)
    eigs = np.linalg.eigvals(G)
    print("GS - spectral radius is : ", np.max(np.abs(eigs)))


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
            break
        else:
            beta = -np.dot(r,Ap)/np.dot(p,Ap)
            p = r + beta*p
        it = it+1
        err.append(calc_residual_norm(A,b,x))

    return x , err

def gmres(A, b, tol) :
    """ """
    x = np.zeros(b.size)
    max_iters = 100
    e1 = np.zeros([max_iters+1])
    e1[0] = 1
    n = b.size
    err=[]
    err.append(calc_residual_norm(A,b,x))
    
    
    Q = np.zeros((n,max_iters))
    H = np.zeros((max_iters+1,max_iters))
    
    r = b - A.dot(x)
    beta = np.linalg.norm(r)
    Q[:,0] = r / beta
    
    for j in range(max_iters-1):
        Q[:,j+1] = A.dot(Q[:,j])
        for i in range(j+1):
            H[i,j] = Q[:,i].dot(Q[:,j+1])
            Q[:,j+1] = Q[:,j+1]- H[i,j]*Q[:,i]
        H[j+1,j] = np.linalg.norm(Q[:,j+1])
        #if (H[j+1,j]>1e-14):
        Q[:,j+1] = Q[:,j+1]/H[j+1,j]
    
        HT = H[:j+2,:j+1].transpose()
        HH= HT.dot(H[:j+2,:j+1])
        Hb = beta * HT.dot(e1[:j+2])
        y = np.linalg.solve(HH,Hb)
    
        #res = np.linalg.norm(H[:j+2,:j+1].dot(y) - beta* e1[:j+2])
        x = Q[:,:j+1].dot(y)
        res = np.linalg.norm(b - A.dot(x))
        err.append(res)
        if (res<tol):
            break
    return x,err


def run_poisson(nx,ny,method):
  n = nx*ny
  A = np.zeros((n,n)) 
  b = np.zeros([n])
  fill_matrix_poisson(A,b,nx,ny)
  T, err=method(A,b,1e-3)
  return T,err
  
def run_adv_diff(Pe,nx,ny,method):
  n = nx*ny
  A = np.zeros((n,n)) 
  b = np.zeros([n])
  fill_adv_diff(Pe,A,b,nx,ny)
  
  T, err=method(A,b,1e-3)
  return T,err
  
  
  


#run_poisson(25,25)
#run_adv_diff(10,25,25)

#Run Jacobi and see mesh influence
# Poisson problem
j_meshes=[5,10,20,30]
j_its=[]
for i in j_meshes:
    T,err = run_poisson(i,i,jacobi)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-o',label="Problem A")

# Peclet=1
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(1,i,i,jacobi)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-^',label="Problem B - Pe=1")

# Peclet=10
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(10,i,i,jacobi)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-s',label="Problem B - Pe=10")
plt.legend()
plt.show()


#Run Gauss-Seidel and see mesh influence
# Poisson problem
j_meshes=[5,10,20,30]
j_its=[]
for i in j_meshes:
    T,err = run_poisson(i,i,gauss_seidel)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-o',label="Problem A")

# Peclet=1
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(1,i,i,gauss_seidel)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-^',label="Problem B - Pe=1")

# Peclet=10
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(10,i,i,gauss_seidel)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-s',label="Problem B - Pe=10")
plt.legend()
plt.show()


#Run CG and see mesh influence
# Poisson problem
j_meshes=[5,10,20,30]
j_its=[]
for i in j_meshes:
    T,err = run_poisson(i,i,conjugate_gradient)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-o',label="Problem A")

# Peclet=1
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(1,i,i,conjugate_gradient)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-^',label="Problem B - Pe=1")

# Peclet=10
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(10,i,i,conjugate_gradient)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-s',label="Problem B - Pe=10")
plt.legend()
plt.show()

#Run GMRES and see mesh influence
# Poisson problem
j_meshes=[5,10,20,30]
j_its=[]
for i in j_meshes:
    T,err = run_poisson(i,i,gmres)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-o',label="Problem A")

# Peclet=1
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(1,i,i,gmres)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-^',label="Problem B - Pe=1")

# Peclet=10
j_its=[]
for i in j_meshes:
    T,err = run_adv_diff(10,i,i,gmres)
    j_its.append(len(err))

plt.plot(j_meshes,j_its,'-s',label="Problem B - Pe=10")
plt.legend()
plt.show()







  

