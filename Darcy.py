from dolfin import *
import matplotlib.pyplot as plt
# Create mesh
mesh = Mesh('mesh_002.xml')
# Define finite elements spaces and build mixed space
BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)
# Introduce manually the parameters
deltaT  = 0.1
tol = 1e-3
t=0.0
load = Expression("t", t = 0.0, degree=1)
K = 2.11e-13;
mu=0.001
rho_fluid = 1.e3;
KK = pow(rho_fluid, 2) * K
TenK = pow(rho_fluid, 2) * K * Identity(2)
# Define trial and test functions
(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)
f = Expression("0", degree=1)

# Define variational form
#a = (dot(sigma, tau) - inner(TenK, u * grad(tau)) + div(sigma) * v)*dx
a = (dot(sigma, tau) - K/mu*u*div(tau) + div(sigma) * v)*dx
L = - f*v*dx


# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = -2
        values[0] = g*n[0]*t
        values[1] = g*n[1]*t
    def value_shape(self):
        return (2,)

Gcrack = BoundarySource(mesh, degree=2)
"""
class BoundarySource1(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = 0
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)

Gboundary = BoundarySource1(mesh, degree=2)
"""
# Define essential boundary
def crackleftboundary(x):
    return near(x[0],-0.5) and abs(x[1])<=0.02
def leftboundary(x):
    return near(x[0],-0.5) and abs(x[1])>=0.02
def botboundary(x):
    return near(x[1],-0.5) 
def rightboundary(x):
    return near(x[0],0.5) 
def topboundary(x):
    return near(x[1],0.5)
# Now, all the pieces are in place for the construction of the essential
# boundary condition: ::

bc = DirichletBC(W.sub(0), Gcrack, crackleftboundary)
#bcleft = DirichletBC(W.sub(0), Gboundary, leftboundary)
#bcbot = DirichletBC(W.sub(0), Gboundary, botboundary)
#bcright = DirichletBC(W.sub(0), Gboundary, rightboundary)
#bctop = DirichletBC(W.sub(0), Gboundary, topboundary)
#bc= [bccrack,bcleft,bcbot,bcright,bctop]
#bc= [bccrack,bcleft,bcbot,bctop]
"""
prseeurebccrack = DirichletBC(W.sub(1), Constant(0.0), crackleftboundary)
prseeurebcleft = DirichletBC(W.sub(1), Constant(0.0), leftboundary)
prseeurebcbot = DirichletBC(W.sub(1), Constant(0.0), botboundary)
prseeurebcright = DirichletBC(W.sub(1), Constant(0.0), rightboundary)
prseeurebctop = DirichletBC(W.sub(1), Constant(0.0), topboundary)

pressurebc=[prseeurebccrack,prseeurebcleft,prseeurebcbot,prseeurebcright,prseeurebctop]

bc=[fluxbccrack,pressurebc]
"""

conc_flux = File ("./Darcy_flow/flux.pvd")
conc_pressure = File ("./Darcy_flow/pressure.pvd")
# Compute solution
w = Function(W)
while t<=2.0 :
    t += deltaT
    load.t=t
    iter = 0
    iter += 1

    solve(a == L, w, bc)
    (sigma, u) = w.split()
 
    conc_flux << sigma
    conc_pressure <<u
