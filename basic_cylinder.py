#%%
import dolfin as fe
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%%
''' 
CONSTANTS ----------------------------------------------------------------------
''' 

# MESH PARAMS
LENGTH = 8
RADIUS = 2
X_MIN, X_MAX, Y_MIN, Y_MAX = 0, LENGTH, 0, RADIUS
N_X, N_Y = 200, 50                      

# MATRIX PARAMS
K_0 = fe.Constant(800)                # Bulk modulus
MU_0 = fe.Constant(600)               # Shear moduls

# FIBRE PARAMS
K_1 = fe.Constant(600)                # ~ Tensile modulus of fibres
K_2 = fe.Constant(1.5)                # ~ Ropiness (K_1 defined in main script)
a0 = fe.Constant((1.0, 0))            # Initial longitudinal fibre direction
FIBRE_BUDGET = 100
FIBRE_CONSTANT = FIBRE_BUDGET/LENGTH
initial_fibre_distribution = FIBRE_CONSTANT*np.ones(N_X + 1)

# DISPLACEMENT PARAMS
DISPLACEMENT = 0.5

#%%
'''
FUNCTIONS ----------------------------------------------------------------------
'''

# Cauchy stress
def sigma():
    cauchy_stress = ((K_0 * (J - 1) * I) 
    + (MU_0 * J**(-5/3) * (B - (1/3) * I_1 * I)) 
    + (2 * K_1 * (I_4 - 1) * fe.exp(K_2 * (I_4 - 1)**2) * fe.outer(a4, a4)))
    return cauchy_stress

# Strain energy
def psi():    
    strain_energy = ((K_0/2) * (J - 1)**2               # Volumetric
    + (MU_0/2) * (I_1_ico - 3)                          # Isochoric
    + (K_1 / 2*K_2) * (fe.exp(K_2 * (I_4 - 1)**2) - 1)) # Anisotropic
    return strain_energy

#%%
'''
OPTIMIZATION OPTIONS FOR THE FORM COMPILER -------------------------------------
'''

fe.set_log_level(50)
fe.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, "eliminate_zeros": True, 
    "precompute_basis_const": True, "precompute_ip_const": True}

#%%
'''
MESH AND BOUNDARY CONDITIONS ---------------------------------------------------
'''

# Create mesh and define function space
mesh = fe.RectangleMesh(fe.Point(X_MIN, Y_MIN), fe.Point(X_MAX, Y_MAX), 
    N_X, N_Y)
V = fe.VectorFunctionSpace(mesh, "Lagrange", 1)

# Get spatial coordinates and x coordinates of mesh
x = fe.SpatialCoordinate(mesh)
coordinates = mesh.coordinates()
x_coordinates = np.unique(coordinates[:, 0])

# Mark boundary subdomians
left_boundary =  fe.CompiledSubDomain("near(x[0], side) && on_boundary",
    side = X_MIN)
right_boundary = fe.CompiledSubDomain("near(x[0], side) && on_boundary",
    side = X_MAX)

# Define Dirichlet boundary (x = X_MIN or x = X_MAX)
left_boundary_displacement = fe.Expression(("l_displacement", "0.0"), 
    l_displacement = -DISPLACEMENT/2, degree = 1)
right_boundary_displacement = fe.Expression(("r_displacement", "0.0"), 
    r_displacement = DISPLACEMENT/2, degree = 1)
left_bc = fe.DirichletBC(V, left_boundary_displacement, left_boundary)
right_bc = fe.DirichletBC(V, right_boundary_displacement, right_boundary)
boundary_conditions = [left_bc, right_bc]

#%%
'''
SOLVER -------------------------------------------------------------------------
'''

# Define functions
du = fe.TrialFunction(V)                # Incremental displacement
v  = fe.TestFunction(V)                 # Test function
u  = fe.Function(V)                     # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = fe.Identity(d)                      # Identity
F = I + fe.grad(u)                      # Deformation gradient
a4 = F*a0                               # Updated fibre direction
J = ((1 + u[1]/x[1]) * fe.det(F))       # Volume ratio
C = F.T*F                               # Right Cauchy-Green tensor
B = F*F.T                               # Left Cauchy-Green tensor
I_1 = (fe.tr(C) + (1 + u[1]/x[1])**2)   # First invariant
I_1_ico = J**(-2/3) * I_1               # Isochoric first invariant
I_4 = fe.dot(a0, C*a0)                  # Anisotropic pseudo invariant

# Total potential energy (ignoring body forces and traction)
Pi = psi()*fe.dx

# Compute first variation of Pi (directional derivative about u in the 
# direction of v)
first_variation = fe.derivative(Pi, u, v)

# Compute Jacobian of first variation
Jacobian = fe.derivative(first_variation, u, du)

# Solve variational problem
fe.solve(first_variation == 0, u, boundary_conditions, J = Jacobian, 
    form_compiler_parameters = ffc_options)

#%%
'''
STRESSES -----------------------------------------------------------------------
'''

# Compute and correct von Mises stress field
deviatoric_cauchy_stress = sigma() - 1/3 * fe.tr(sigma()) * I
von_Mises = fe.sqrt(3/2 * fe.inner(deviatoric_cauchy_stress, 
            deviatoric_cauchy_stress))
# ------------------------------------------------------------------------------
vm_correction = -1.7320508075688772*((MU_0*((F[1, 1])*(F[0, 1]) + (F[1, 0])*(F[0, 0]))/J**1.6666666666666667 + 2*K_1*(I_4 - 1)*fe.exp(K_2*(I_4 - 1)**2)*(F[1, 0])*(F[0, 0]))**2 + 0.5*(MU_0*(-0.33333333333333331*I_1 + (F[1, 1])**2 + (F[1, 0])**2)/J**1.6666666666666667 + K_0*(J - 1) + 2*K_1*(I_4 - 1)*fe.exp(K_2*(I_4 - 1)**2)*(F[1, 0])**2)**2 + 0.5*(MU_0*(-0.33333333333333331*I_1 + (F[0, 1])**2 + (F[0, 0])**2)/J**1.6666666666666667 + K_0*(J - 1) + 2*K_1*(I_4 - 1)*fe.exp(K_2*(I_4 - 1)**2)*(F[0, 0])**2)**2)**0.5 + 1.7320508075688772*(0.5*(MU_0*(-0.33333333333333331*I_1 + (x[1] + u[1])**2/(x[1])**2)/J**1.6666666666666667 + K_0*(J - 1))**2 + (MU_0*((F[1, 1])*(F[0, 1]) + (F[1, 0])*(F[0, 0]))/J**1.6666666666666667 + 2*K_1*(I_4 - 1)*fe.exp(K_2*(I_4 - 1)**2)*(F[1, 0])*(F[0, 0]))**2 + 0.5*(MU_0*(-0.33333333333333331*I_1 + (F[1, 1])**2 + (F[1, 0])**2)/J**1.6666666666666667 + K_0*(J - 1) + 2*K_1*(I_4 - 1)*fe.exp(K_2*(I_4 - 1)**2)*(F[1, 0])**2)**2 + 0.5*(MU_0*(-0.33333333333333331*I_1 + (F[0, 1])**2 + (F[0, 0])**2)/J**1.6666666666666667 + K_0*(J - 1) + 2*K_1*(I_4 - 1)*fe.exp(K_2*(I_4 - 1)**2)*(F[0, 0])**2)**2)**0.5
# ----------------------------------------------
von_Mises_corrected = von_Mises - vm_correction

# Project von Mises stresses to mesh
P = fe.FunctionSpace(mesh, 'Lagrange', 1)
von_Mises = fe.project(von_Mises, P)
von_Mises_corrected = fe.project(von_Mises_corrected, P)
vm_correction = fe.project(vm_correction, P)

# Get current date in YYYY-MM-DD format
date = datetime.date.today().strftime('%m-%d')

# Save solution in VTK format with date
ufile_pvd = fe.File("../../../mnt/c/Users/ks16239/Documents/phd/code/results/displacement_{}.pvd".format(date))
ufile_pvd << u
vmfile_pvd = fe.File("../../../mnt/c/Users/ks16239/Documents/phd/code/results/vm_{}.pvd".format(date))
vmfile_pvd << von_Mises_corrected


# POLYNOMIAL_DEGREE = 35
# fibre_distribution = initial_fibre_distribution
# alpha = np.inf
# TOLERANCE = 0.01
# counter = 0
# output = []

# while alpha > TOLERANCE:
#     # Define functions
#     du = fe.TrialFunction(V)            # Incremental displacement
#     v  = fe.TestFunction(V)             # Test function
#     u  = fe.Function(V)                 # Displacement from previous iteration

#     # Kinematics
#     d = u.geometric_dimension()
#     I = fe.Identity(d)                  # Identity tensor
#     F = I + fe.grad(u)                  # Deformation gradient
#     A_4 = F*A_0 
#     J = ((1 + u[1]/x[1]) * fe.det(F))   # Volume ratio
#     C = F.T*F                           # Right Cauchy-Green tensor
#     B = F*F.T                           # Left Cauchy-Green tensor
#     I_1 = (fe.tr(C) + (1 + u[1]/x[1])**2) # First invariant
#     I_1_ico = J**(-2/3) * I_1           # Isochoric first invariant
#     I_4 = fe.dot(A_4, C*A_4)            # Anisotropic pseudo invariant
#     I_4_ico = J**(-2/3) * I_4           # Isochoric anisotropic pseudo invariant
#                                         #   (we don't use this though)

#     # Fit polynomial to fibre distribution
#     f = np.polyfit(x_coordinates, fibre_distribution, POLYNOMIAL_DEGREE)
#     plt.plot(x_coordinates, np.polyval(f, x_coordinates), label = 'Polyfit')
#     print(np.polyval(f, x_coordinates))

#     # Generate the cpp expression for defining the polynomial with FEniCS
#     expression_string = " + ".join([f"f{i}*pow(x[0], {POLYNOMIAL_DEGREE-i})" 
#         for i in range(POLYNOMIAL_DEGREE + 1)])
#     K_1 = fe.Expression(expression_string, degree=1, 
#         **{f"f{i}": f[i] for i in range(POLYNOMIAL_DEGREE + 1)})

#     # Total potential energy (ignoring body forces and traction)
#     Pi = psi()*fe.dx

#     # Compute first variation of Pi (directional derivative about u in the 
#     # direction of v)
#     first_variation = fe.derivative(Pi, u, v)

#     # Compute Jacobian of first variation
#     Jacobian = fe.derivative(first_variation, u, du)

#     # Solve variational problem
#     fe.solve(first_variation == 0, u, boundary_conditions, J = Jacobian, 
#         form_compiler_parameters = ffc_options)

#     # Compute and project von Mises stresses to mesh
#     deviatoric_cauchy_stress = sigma() - 1/3 * (fe.tr(sigma()) + K_0*(J - 1)
#         - (MU_0*J**(-5/3)) * 1/3 * fe.inner(F, F)) * I
#     # deviatoric_cauchy_stress = sigma() - 1/3 * (fe.tr(sigma()) 
#     #     + (K_0 * (J - 1)) + (MU_0 * J(-5/3) * ((1 + u[1]/x[1])**2 - I_1/3))) * I
#     von_Mises = fe.sqrt(3/2 * fe.inner(deviatoric_cauchy_stress, 
#                 deviatoric_cauchy_stress))
#     # Look up von Mises stress in cylindrical coordinates
#     P = fe.FunctionSpace(mesh, 'Lagrange', 1)
#     von_Mises = fe.project(von_Mises, P)

#     # Calculate maximum stress profile
#     vertex_values = von_Mises.compute_vertex_values(mesh)
#     max_stress_profile = np.max(vertex_values.reshape(N_Y + 1, N_X + 1), 0)
#     plt.plot(x_coordinates, max_stress_profile, label = 'MSP')
#     plt.legend()
#     plt.grid()

#     # Calculate bounds on alpha, starting with subtracting the average value of 
#     # the maximum stress profile to shift integral to zero.
#     average_value = np.mean(max_stress_profile)
#     zeroed_max_stress_profile = [max_stress - average_value for max_stress 
#         in max_stress_profile]
#     potential_alphas = [i/max(-j, 0) for i, j 
#         in zip(fibre_distribution, zeroed_max_stress_profile)]
#     alpha_upper_bound = min(potential_alphas)

#     # Generate scaled fibre distribution
#     alpha = alpha_upper_bound/2
#     scaled_zeroed_max_stress_profile = [i*alpha for i 
#         in zeroed_max_stress_profile]

#     # Update fibre distribution
#     updated_fibre_distribution = [a + b for a, b 
#         in zip(fibre_distribution, scaled_zeroed_max_stress_profile)]
#     fibre_distribution = updated_fibre_distribution

#     print(f'Max von Mises stress: {max(vertex_values)}')
#     print(f'Mean von Mises stress: {np.mean(vertex_values)}')
#     print(f'alpha = {alpha}')
#     output.append((counter, max(vertex_values), np.mean(vertex_values), alpha))
#     counter += 1

# print(f'Finished after {counter} iterations with a final maximum von Mises \
# stress of {max(vertex_values)}.')

# #%%
# # Plot max and mean von Mises against iterations
# plt.plot([row[0] for row in output], [row[1] for row in output], 
#     label = 'Maximum von Mises stress')
# plt.plot([row[0] for row in output], [row[2] for row in output], 
#     label = 'Mean von Mises stress')
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.xlim(0, counter - 1)
# plt.legend()
# plt.show()

# # Plot alpha against iterations
# plt.Figure()
# plt.plot([row[0] for row in output], [row[3] for row in output], 
#     label = 'alpha')
# plt.xlabel('Iteration')
# plt.xlim(0, counter - 1)
# plt.legend()
# plt.show()

# # Plot fibre distributions
# plt.Figure()
# plt.plot(x_coordinates, initial_fibre_distribution,
#     label = 'Initial fibre distribution')
# plt.plot(x_coordinates, fibre_distribution, 
#     label = 'Final fibre distribution')
# plt.ylim(0, 500)
# plt.legend()



# # %%
# %%
