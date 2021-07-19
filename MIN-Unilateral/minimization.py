# FEnics code  Variational Fracture Mechanics
#
# A static solution of the variational fracture mechanics problems using the minimization principle
# authors:
# zhangxiaodongxi@gmail.com
# Zhang Xiaodong

from fenics import *
from dolfin import *
import sympy, sys, math, os, subprocess, shutil

mesh=Mesh('mesh.xml')

V_u = VectorFunctionSpace(mesh, "CG", 1)
V_alpha = FunctionSpace(mesh, "CG", 1)

u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
alpha_, alpha, alpha_t = Function(V_alpha), TrialFunction(V_alpha), TestFunction(V_alpha)

lmbda = 121.1538e3 # E*nu/((1.0+nu)*(1.0-2.0*nu))
mu =80.7692e3 # E / (2.0 * (1.0 + nu)) shear modulus
E = 210.e3  # Young modulus

nu = 0.3  # Poisson ratio
ell = 0.0375# internal length scale
Gc = 2.7 # fracture toughness MPa.mm

#=======================================================================================
# Loading
u_r = 7.e-3 # reference value for the loading (imposed displacement)
body_force = Constant((0.,0.))  # bulk load

top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")
def Crack(x):
    return abs(x[1]) < 1e-03 and x[0] <= 0.0
load = Expression("t", t = 0.0, degree=1)
#=======================================================================================
def w(alpha_):
	return alpha_**2

def angle_bracket_plus(a):
	return (a+abs(a))/2

def angle_bracket_minus(a):
	return (a-abs(a))/2
#=======================================================================================
# strain, stress and strain energy for Isotropic and Amor's model
#=======================================================================================
def g(alpha_):#degradation function
    return (1-alpha_)**2
	#----------------------------------------------------------------------------------------
def eps(u_):#Geometrical strain
	return sym(grad(u_))

def dev_eps(u_):
	return eps(u_) - 1/3*tr(eps(u_))*Identity(ndim)
	#----------------------------------------------------------------------------------------
def sigma0(u_):#Application of the sound elasticy tensor on the strain tensor
	Id = Identity(len(u_))
	return 2.0*mu*eps(u_) + lmbda*tr(eps(u_))*Id

#def sigma(u_, alpha_):
	#return g(alpha_) * sigma0(u_)

def sigma(u_,alpha_):#stress model B
    return  g(alpha_) * ( (lmbda+2/3*mu) * ( angle_bracket_plus( tr(eps(u_))) * Identity(ndim) )+ 2*mu*dev_eps(u_) ) + (lmbda+2/3*mu)*( angle_bracket_minus(tr(eps(u_))) * Identity(ndim))
#----------------------------------------------------------------------------------------
#def psi_0(u_):#The strain energy density for a linear isotropic material
	#return  0.5 * lmbda* tr(eps(u_))**2#+mu*inner(dev(eps(u_)),dev(eps(u_)))

def psi_0(u_):#The strain energy density for a linear isotropic material
	return  0.5 * lmbda * tr(eps(u_))**2 + mu * eps(u_)**2

#def psi(u_, alpha_):#The strain energy density
	#return g(alpha_) * psi_0(u_)

def psi(u_,alpha_):#The strain energy for model B
    return  g(alpha_) * (0.5*(lmbda+2/3*mu) * ( angle_bracket_plus(tr(eps(u_))))**2 + mu*dev_eps(u_)**2) + 0.5*(lmbda+2/3*mu) * ( angle_bracket_minus(tr(eps(u_))))**2




# Normalization constant for the dissipated energy
# to get Griffith surface energy for ell going to zero
z = sympy.Symbol("z", positive=True)
c_w = float(4*sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))

bcbot= DirichletBC(V_u, Constant((0.0,0.0)), bot)
bctop = DirichletBC(V_u.sub(1), load, top)
bc_u = [bcbot, bctop]

bc_alpha = [DirichletBC(V_alpha, Constant(1.0), Crack)]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)
ndim = mesh.geometry().dim() # get number of space dimensions


boundaries.set_all(0)
top.mark(boundaries,1)


alpha_0 = interpolate(Constant(0.0), V_alpha) # initial (known) alpha, undamaged everywhere.

#====================================================================================
# Define  problem and solvers
#====================================================================================
elastic_energy = psi(u_, alpha_)*dx
external_work = dot(body_force, u_)*dx #+ dot(sigma_T, u_)*ds(3)+ dot(sigma_B, u_)*ds(4)
dissipated_energy = Gc/float(c_w)*(w(alpha_)/ell + ell*inner(grad(alpha_), grad(alpha_)))*dx

print (c_w)
total_energy = elastic_energy + dissipated_energy - external_work

# First derivatives of energies (Residual)
Du_total_energy = derivative(total_energy, u_, u_t)
Dalpha_total_energy = derivative(total_energy, alpha_, alpha_t)

import ufl
Du_total_energy=ufl.replace(Du_total_energy,{u_:u})
# Second derivatives of energies (Jacobian)
J_alpha = derivative(Dalpha_total_energy, alpha_, alpha)

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(Du_total_energy),rhs(Du_total_energy), u_, bc_u)

# Variational problem for the damage (non-linear to use variational inequality solvers of petsc)
# Define the minimisation problem by using OptimisationProblem class
class DamageProblem(OptimisationProblem):

	  def __init__(self):
		          OptimisationProblem.__init__(self)

	    # Objective function
	  def f(self, x):
		  alpha_.vector()[:] = x
		  return assemble(total_energy)

	    # Gradient of the objective function
	  def F(self, b, x):
		  alpha_.vector()[:] = x
		  assemble(Dalpha_total_energy, tensor=b)

	    # Hessian of the objective function
	  def J(self, A, x):
		  alpha_.vector()[:] = x
		  assemble(J_alpha, tensor=A)

# Create the PETScTAOSolver
problem_alpha = DamageProblem()

solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters["linear_solver"] = 'gmres'
solver_u.parameters["preconditioner"] = 'ilu'

#prm = solver_u.parameters["krylov_solver"]  # short form
#prm["absolute_tolerance"] = 1E-7
#prm["relative_tolerance"] = 1E-4
#prm["maximum_iterations"] = 10000


#info(solver_u.parameters,True)

solver_alpha = PETScTAOSolver()
'''
solver_alpha.parameters["method"] = "tron"
solver_alpha.parameters["monitor_convergence"] = True
solver_alpha.parameters["report"] = True
solver_alpha.parameters["maximum_iterations"] = 1000
'''
#info(solver_alpha.parameters,True) # uncomment this line to see available parameters
alpha_lb = interpolate(Expression("0.", degree =1), V_alpha) # lower bound, set to 0
alpha_ub = interpolate(Expression("1.", degree =1), V_alpha) # upper bound, set to 1

for bc in bc_alpha:
	bc.apply(alpha_lb.vector())

for bc in bc_alpha:
	bc.apply(alpha_ub.vector())

results = []
file_alpha = File("./1/alpha.pvd")
file_u = File("./1/u.pvd")
fname = open('ForeceDisp.txt','w')
deltaT=0.1
t=0
tol=1e-3
while t<=1.0:
    t += deltaT
    if t >=0.8:
        deltaT = 0.05
    load.t=t*u_r
    iter = 0
    err_alpha = 1

    while err_alpha > tol:
        iter += 1
        solver_u.solve()
        solver_alpha.solve(problem_alpha, alpha_.vector(), alpha_lb.vector(), alpha_ub.vector())
        err_alpha = (alpha_.vector() - alpha_0.vector()).norm('linf')

        alpha_0.vector()[:] = alpha_.vector()
        alpha_lb.vector()[:] = alpha_.vector()

        if err_alpha < tol:

            print ('Iterations:', iter, ', Total time', t)
            elastic_energy_value = assemble(elastic_energy)
            surface_energy_value = assemble(dissipated_energy)
            print ('elastic_energy_value', elastic_energy_value, ', surface_energy_value', surface_energy_value)

            if round(t*1e4) % 10 == 0:
                file_alpha << (alpha_,t)
                file_u << (u_,t)
                Traction = dot(sigma0(u_),n)
                fy = Traction[1]*ds(1)
                fname.write(str(t*u_r)+"\t")
                fname.write(str(assemble(fy))+"\t")
                fname.write(str(elastic_energy_value) + "\t")
                fname.write(str(assemble(external_work)) + "\t")
                fname.write(str(surface_energy_value) + "\t")
                fname.write(str(assemble(total_energy)) + "\n")
fname.close()

print ('Simulation completed')
