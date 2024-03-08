import numpy as np
import matplotlib.pyplot as plt


font = {'weight' : 'normal',
        'size'   : 13}

plt.rc('font', **font)
colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
cycle = plt.cycler("color", colors) 
myparams = {'axes.prop_cycle': cycle}
plt.rcParams.update(myparams)

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
    while (err[-1]/err[0] > tol ):
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
    while (err[-1]/err[0] > tol ):
        for i in range(0,N):
            x[i] = (b[i] - np.dot(A[i,0:i],x[0:i]) - np.dot(A[i,i+1:],x[i+1:])) / A[i,i]
        it = it+1
        err.append(calc_residual_norm(A,b,x))
5

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
        if np.sqrt(np.sum((r**2))) /err[0] < tol:
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
        if (res/err[0]<tol):
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
  
#Run Jacobi and see mesh influence
# Poisson problem
j_meshes=[5,10,20,30]
j_a_its=[]
for i in j_meshes:
    T,err = run_poisson(i,i,jacobi)
    j_a_its.append(len(err))


# Peclet=1
j_b_its=[]
for i in j_meshes:
    T,err = run_adv_diff(1,i,i,jacobi)
    j_b_its.append(len(err))


# Peclet=10
j_c_its=[]
for i in j_meshes:
    T,err = run_adv_diff(10,i,i,jacobi)
    j_c_its.append(len(err))

plt.plot(j_meshes,j_a_its,'-o',ms=9,label="Problem A")
plt.plot(j_meshes,j_b_its,'-^',ms=9,label="Problem B - Pe=1")
plt.plot(j_meshes,j_c_its,'-s',ms=9,label="Problem B - Pe=10")
plt.xlabel("Number of nodes per axis ($\sqrt{n_{dofs}}$)")
plt.ylabel("Number of iterations (tol=$10^{-3}$)")
plt.legend()
plt.tight_layout()
plt.savefig("../slides/images/j_its.pdf")
plt.show()


#Run Gauss-Seidel and see mesh influence
# Poisson problem
j_meshes=[5,10,20,30]
gs_a_its=[]
for i in j_meshes:
    T,err = run_poisson(i,i,gauss_seidel)
    gs_a_its.append(len(err))


# Peclet=1
gs_b_its=[]
for i in j_meshes:
    T,err = run_adv_diff(1,i,i,gauss_seidel)
    gs_b_its.append(len(err))


# Peclet=10
gs_c_its=[]
for i in j_meshes:
    T,err = run_adv_diff(10,i,i,gauss_seidel)
    gs_c_its.append(len(err))

plt.plot(j_meshes,gs_a_its,'-o',ms=9,label="Problem A")
plt.plot(j_meshes,gs_b_its,'-^',ms=9,label="Problem B - Pe=1")
plt.plot(j_meshes,gs_c_its,'-s',ms=9,label="Problem B - Pe=10")

plt.xlabel("Number of nodes per axis ($\sqrt{n_{dofs}}$)")
plt.ylabel("Number of iterations (tol=$10^{-3}$)")
plt.legend()
plt.tight_layout()
plt.savefig("../slides/images/gs_its.pdf")
plt.show()

plt.plot(j_meshes,j_a_its,'--o',mfc='none', mec=colors[0],ms=9,label="Jacobi A")
plt.plot(j_meshes,j_c_its,'--s',mfc='none', mec=colors[1],ms=9,label="Jacobi B")
plt.plot(j_meshes,gs_a_its,'-o',color=colors[0],ms=9,label="GS A")
plt.plot(j_meshes,gs_c_its,'-s',color=colors[1],ms=9,label="GS B")
plt.savefig("../slides/images/gs_j_its.pdf")
plt.xlabel("Number of nodes per axis ($\sqrt{n_{dofs}}$)")
plt.ylabel("Number of iterations (tol=$10^{-3}$)")
plt.legend()
plt.tight_layout()
plt.show()


#Run CG and see mesh influence
# Poisson problem
cg_meshes=[5,10,20,30,40,50]
cg_a_its=[]
for i in cg_meshes:
    T,err = run_poisson(i,i,conjugate_gradient)
    cg_a_its.append(len(err))


# Peclet=1
cg_b_its=[]
for i in cg_meshes:
    T,err = run_adv_diff(1,i,i,conjugate_gradient)
    cg_b_its.append(len(err))


# Peclet=10
cg_c_its=[]
for i in cg_meshes:
    T,err = run_adv_diff(10,i,i,conjugate_gradient)
    cg_c_its.append(len(err))

plt.plot(cg_meshes,cg_a_its,'-o',ms=9,label="Problem A")
plt.plot(cg_meshes,cg_b_its,'-^',ms=9,label="Problem B - Pe=1")
plt.plot(cg_meshes,cg_c_its,'-s',ms=9,label="Problem B - Pe=10")
plt.xlabel("Number of nodes per axis ($\sqrt{n_{dofs}}$)")
plt.ylabel("Number of iterations (tol=$10^{-3}$)")
plt.legend()
plt.tight_layout()
plt.savefig("../slides/images/cg_its.pdf")
plt.show()

#Run GMRES and see mesh influence
# Poisson problem
gmres_a_its=[]
for i in cg_meshes:
    T,err = run_poisson(i,i,gmres)
    gmres_a_its.append(len(err))


# Peclet=1
gmres_b_its=[]
for i in cg_meshes:
    T,err = run_adv_diff(1,i,i,gmres)
    gmres_b_its.append(len(err))


# Peclet=10
gmres_c_its=[]
for i in cg_meshes:
    T,err = run_adv_diff(10,i,i,gmres)
    gmres_c_its.append(len(err))

plt.plot(cg_meshes,gmres_a_its,'-o',ms=9,label="Problem A")
plt.plot(cg_meshes,gmres_b_its,'-^',ms=9,label="Problem B - Pe=1")
plt.plot(cg_meshes,gmres_c_its,'-s',ms=9,label="Problem B - Pe=10")
plt.xlabel("Number of nodes per axis ($\sqrt{n_{dofs}}$)")
plt.ylabel("Number of iterations (tol=$10^{-3}$)")
plt.legend()
plt.tight_layout()
plt.savefig("../slides/images/gmres_its.pdf")
plt.show()









  

