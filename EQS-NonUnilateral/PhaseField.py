# Phase field fracture implementation in FEniCS
# The code is distributed under a BSD license

# If using this code for research or industrial purposes, please cite:
# Hirshikesh, S. Natarajan, R. K. Annabattula, E. Martinez-Paneda.
# Phase field modelling of crack propagation in functionally graded materials.
# Composites Part B: Engineering 169, pp. 239-248 (2019)
# doi: 10.1016/j.compositesb.2019.04.003

# Emilio Martinez-Paneda (mail@empaneda.com)
# University of Cambridge

# Preliminaries and mesh
from dolfin import *
mesh = Mesh('mesh.xml')
ndim = mesh.geometry().dim() # get number of space dimensions

# Define Space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)
p, q = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)

# Introduce manually the material parameters
Gc =  2.7
l = 0.0375
lmbda = 121.1538e3
mu = 80.7692e3
body_force = Constant((0.,0.))
# Constituive functions
def epsilon(u):
    return sym(grad(u))
def dev_eps(u):
	return epsilon(u) - 1/3*tr(epsilon(u))*Identity(ndim)
def sigma(u):
    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))
#def psi(u):
    #return 0.5*(lmbda+2*mu/3)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+\
           #mu*inner(dev_eps(u),dev_eps(u))
def psi(u):
    return  0.5 * lmbda * tr(epsilon(u))**2 + mu * epsilon(u)**2
def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
def elastic_energy(u,p):
    return  0.5*(1-p)**2*inner(sigma(u), epsilon(u))*dx
def external_work(u):
    return  dot(body_force, u)*dx
def dissipated_energy(p):
    return  Gc/2*(p/l+ 0.5*l*dot(grad(p), grad(p)))*dx
def total_energy(u,p):
    return  elastic_energy(u,p) + dissipated_energy(p) - external_work(u)

# Boundary conditions
top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")
def Crack(x):
    return abs(x[1]) < 1e-03 and x[0] <= 0.0
load = Expression("t", t = 0.0, degree=1)
bcbot= DirichletBC(W, Constant((0.0,0.0)), bot)
bctop = DirichletBC(W.sub(1), load, top)
bc_u = [bcbot, bctop]
bc_phi = [DirichletBC(V, Constant(1.0), Crack)]
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Variational form
unew, uold = Function(W), Function(W)
pnew, pold, Hold = Function(V), Function(V), Function(V)
E_du = ((1.0-pold)**2)*inner(grad(v),sigma(u))*dx
E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))\
            *inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx
p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)
p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)
solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)

# Initialization of the iterative procedure and output requests
t = 0
u_r = 0.007
deltaT  = 0.1
tol = 1e-3
conc_phi = File ("./EQS-NonUnilateral/phi.pvd")
conc_u = File ("./EQS-NonUnilateral/u.pvd")
fname = open('output.txt', 'w')

# Staggered scheme
while t<=1.0:
    t += deltaT
    if t >=0.8:
        deltaT = 0.05
    load.t=t*u_r
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        solver_disp.solve()
        solver_phi.solve()
        err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)

        uold.assign(unew)
        pold.assign(pnew)
        Hold.assign(project(psi(unew), WW))

        if err < tol:

            print ('Iterations:', iter, ', Total time', t)

            if round(t*1e4) % 10 == 0:
                conc_phi << pnew
                conc_u << unew

                Traction = dot(sigma(unew),n)
                fy = Traction[1]*ds(1)
                #total_energy=elastic_energy + dissipated_energy - external_work
                fname.write(str(t*u_r) + "\t")
                fname.write(str(assemble(fy)) + "\t")
                fname.write(str(assemble(elastic_energy(unew,pnew))) + "\t")
                fname.write(str(assemble(external_work(unew))) + "\t")
                fname.write(str(assemble(dissipated_energy(pnew))) + "\t")
                fname.write(str(assemble(total_energy(unew,pnew))) + "\n")

fname.close()
print ('Simulation completed')
