from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, pi
from scipy.linalg import eigh
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation

def second_order_second_derivative_matrix(N, dx):
    """Creates the second-order finite difference matrix for the second derivative."""
    diagonals = [-2 * np.ones(N), np.ones(N - 1), np.ones(N - 1)]
    offsets = [0, -1, 1]
    return diags(diagonals, offsets) / dx**2

def infinite_potential_well(n, L, dx, x_min=0, x_max=None, V=None, method='second-order', mass=1):
    """
    Computes the wavefunction of a particle in an infinite potential well using the finite difference method.
    Assumes hbar = 1 and mass = 1.
    """
    if x_max is None:
        x_max = L
    x = np.arange(x_min, x_max, dx)
    N = len(x)
    
    if V is None:
        V = np.zeros(N)  # Default to zero potential if not provided
    
    if method == 'second-order':
        # Simplified Hamiltonian with hbar = 1 and mass = 1
        H = -0.5 * second_order_second_derivative_matrix(N, dx).toarray() + np.diag(V)
    else:
        raise ValueError('Invalid method')
    
    print("Solving eigenvalue problem...")
    w, v = eigh(H)
    print("Eigenvalue problem solved.")
    psi = v[:, n-1] / np.sqrt(dx)
    
    # Ensure consistent sign for the wavefunction
    if psi[0] < 0:
        psi = -psi
    
    return x, psi, w

def infinite_potential_well_analytical(n, L, x_min=0, x_max=None):
    """Computes the analytical wavefunction of a particle in an infinite potential well."""
    if x_max is None:
        x_max = L
    x = np.linspace(x_min, x_max, 1000)
    L_eff = x_max - x_min
    psi = np.sqrt(2 / L_eff) * np.sin(n * pi * (x - x_min) / L_eff)
    return x, psi

def infinite_potential_well_energy_analytical(n, L, mass=1):
    """Computes the analytical energy levels of a particle in an infinite potential well."""
    return (n**2 * pi**2) / (2 * mass * L**2)  # hbar = 1, mass = 1

def plot_infinite_potential_well(L, dx, n_levels, mass=m_e, V=None, x_min=0, x_max=None):
    """Plots the wavefunctions and energy levels for an infinite potential well."""
    if x_max is None:
        x_max = L

    plt.figure()
    plt.title('Infinite Potential Well')
    plt.xlabel('Position $x$')
    plt.ylabel('Wavefunction $\psi(x)$')
    plt.grid(True)

    # Calculate the range of the first energy level
    x, psi, eigenvalues = infinite_potential_well(1, L, dx, x_min, x_max, V=V, method='second-order', mass=mass)
    offset = 2.1*(np.max(psi) - np.min(psi))

    energy_levels = []

    for n in range(1, n_levels + 1):
        # Numerical solution
        x, psi, eigenvalues = infinite_potential_well(n, L, dx, x_min, x_max, V=V, method='second-order', mass=mass)
        plt.scatter(x, psi + eigenvalues[n-1] +(n-1)* offset, label=f'Numerical $n={n}$', s=1)
        
        if V is None:
            # Analytical solution
            x_analytical, psi_analytical = infinite_potential_well_analytical(n, L, x_min, x_max)
            plt.plot(x_analytical, psi_analytical + eigenvalues[n-1] + (n-1) * offset, label=f'Analytical $n={n}$')
        
        # Plot energy level
        plt.axhline(y=eigenvalues[n-1] + (n-1) * offset, color='k', linestyle='--', linewidth=0.5)
        energy_levels.append(eigenvalues[n-1] + (n-1) * offset)
    
    # Plot vertical lines for the potential
    plt.axvline(x=x_min, color='k', linestyle='-', linewidth=1)
    plt.axvline(x=x_max, color='k', linestyle='-', linewidth=1)
    plt.text(x_min, max(eigenvalues) * 1.1 + (n_levels) * offset, r'$\infty$', fontsize=12, ha='center')
    plt.text(x_max, max(eigenvalues) * 1.1 + (n_levels) * offset, r'$\infty$', fontsize=12, ha='center')

    # Plot the potential
    if V is not None:
        plt.plot(x, (V*3)-0.7*offset, label='Potential $V(x)$', color='r')

    # Customize y-axis to show energy levels
    def energy_level_formatter(y, pos):
        for i, level in enumerate(energy_levels):
            if np.isclose(y, level, atol=0.1):
                return f'$E_{i+1}$'
        return ''

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(energy_level_formatter))
    plt.gca().yaxis.set_major_locator(ticker.FixedLocator(energy_levels))

    plt.show()

    # Compare energy levels
    for n in range(1, n_levels + 1):
        E_numerical = eigenvalues[n-1]
        if V is None:
            E_analytical = infinite_potential_well_energy_analytical(n, L, mass=mass)
            energy_difference = E_numerical - E_analytical
            percentage_difference_to_analytical = np.abs(energy_difference / E_analytical) * 100

            print(f"Numerical energy (n={n}): {E_numerical:.5e} J")
            print(f"Percentage difference: {percentage_difference_to_analytical:.5f} %")
    # Compare eigenfunctions
    if V is None:
        compare_eigenfunctions(n_levels, L, dx, mass, x_min, x_max)

def compare_eigenfunctions(n_levels, L, dx, mass=m_e, x_min=0, x_max=None):
    """Compares the eigenfunctions by calculating the dot product of the analytical and numerical eigenstates."""
    if x_max is None:
        x_max = L

    for n in range(1, n_levels + 1):
        # Numerical solution
        x_num, psi_num, _ = infinite_potential_well(n, L, dx, x_min, x_max, method='second-order', mass=mass)
        
        # Analytical solution
        x_analytical, psi_analytical = infinite_potential_well_analytical(n, L, x_min, x_max)
        
        # Interpolate numerical solution to match the analytical grid
        psi_num_interp = np.interp(x_analytical, x_num, psi_num)
        
        # Calculate dot product
        dot_product = np.dot(psi_analytical, psi_num_interp) * (x_analytical[1] - x_analytical[0])
        
        print(f"Dot product of analytical and numerical eigenstates for n={n}: {dot_product:.5f}")

def define_potential(x, potential_type='constant', mass=1, omega=1):
    """Defines different potential functions."""
    if potential_type == 'constant':
        return np.zeros_like(x)
    elif potential_type == 'cosine':
        return np.cos(x) * 1e-2
    elif potential_type == 'gaussian':
        return np.exp(-(x-25)**2 / (2 * (1)**2)) * 1e-2
    elif potential_type == 'qsho':
        return 0.5 * mass * omega**2 * x**2 * 1e-5
    else:
        raise ValueError('Invalid potential type')

def qsho_energy_analytical(n,L_well=1, mass=1):
    """
    Computes the analytical energy levels for a quantum simple harmonic oscillator using perturbation theory.
    """
    # Base energy levels for the infinite potential well
    E_0 = (n**2 * pi**2) / (2 * mass * L_well**2)  # hbar = 1, mass = 1
    
    # First-order correction due to the QSHO potential
    x2_expectation = (L_well**2 / 4) * (1 - (6 / (n**2 * pi**2)))
    correction = 0.5 * mass * x2_expectation *10e-5 # omega = 1
    
    return E_0 + correction

def cosine_energy_analytical(n, L_well, V0, mass=1):
    """
    Computes the analytical energy levels for a cosine potential using perturbation theory.
    """
    # Base energy levels for the infinite potential well
    E_0 = (n**2 * pi**2) / (2 * mass * L_well**2)  # hbar = 1, mass = 1
    
    # First-order correction due to the cosine potential
    def integrand(x):
        psi_n_squared = (2 / L_well) * np.sin(n * pi * x / L_well)**2
        return psi_n_squared * V0 * np.cos(x)
    
    # Perform the integral over the well
    x_vals = np.linspace(0, L_well, 1000)
    correction = np.trapz(integrand(x_vals), x_vals)
    
    return E_0 + correction
    
def display_eigenvalues_table(L, dx, n_levels, mass=m_e, V=None, x_min=0, x_max=None, potential_type=None):
    """
    Displays a table of eigenvalues for the given potential, including analytical solutions if available.
    For zero potential, also includes the dot product of numerical and analytical eigenfunctions.
    """
    if x_max is None:
        x_max = L

    eigenvalues_table = []

    for n in range(1, n_levels + 1):
        # Numerical solution
        x_num, psi_num, eigenvalues = infinite_potential_well(n, L, dx, x_min, x_max, V=V, method='second-order', mass=mass)
        E_numerical = eigenvalues[n-1]

        if potential_type == 'zero':
            # Analytical solution for zero potential (infinite potential well)
            E_analytical = infinite_potential_well_energy_analytical(n, L, mass=mass)

            # Compute the dot product of numerical and analytical eigenfunctions
            x_analytical, psi_analytical = infinite_potential_well_analytical(n, L, x_min, x_max)
            psi_num_interp = np.interp(x_analytical, x_num, psi_num)  # Interpolate numerical solution to match analytical grid
            dot_product = np.dot(psi_analytical, psi_num_interp) * (x_analytical[1] - x_analytical[0])
            dot_product = round(dot_product, 2)  # Round to 2 decimal places
        elif potential_type == 'qsho':
            # Analytical solution for QSHO
            E_analytical = qsho_energy_analytical(n,L, mass=mass)
            dot_product = None
        elif potential_type == 'cosine':
            # Analytical solution for cosine potential
            E_analytical = cosine_energy_analytical(n, L, V0=1e-40, mass=mass)
            dot_product = None
        else:
            E_analytical = None
            dot_product = None

        if E_analytical is not None:
            energy_difference = E_numerical - E_analytical
            percentage_difference_to_analytical = np.abs(energy_difference / E_analytical) * 100
            if potential_type == 'zero':
                eigenvalues_table.append([n, E_numerical, E_analytical, percentage_difference_to_analytical, dot_product])
            else:
                eigenvalues_table.append([n, E_numerical, E_analytical, percentage_difference_to_analytical])
        else:
            eigenvalues_table.append([n, E_numerical])

    # Define headers
    if potential_type == 'zero':
        headers = ["n", "Numerical EValue (J)", "Analytical EValue (J)", "Difference (%)", "Dot Product"]
    elif potential_type in ['cosine', 'qsho']:
        headers = ["n", "Numerical EValue (J)", "Analytical EValue (J)", "Difference (%)"]
    else:
        headers = ["n", "Numerical Eigenvalue (J)"]

    # Print the table
    print("\n".join(["\t".join(map(lambda x: f"{x:.2e}" if isinstance(x, float) else str(x), row)) for row in [headers] + eigenvalues_table]))
    
    # Display the table using matplotlib
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[[f"{item:.2e}" if isinstance(item, float) else str(item) for item in row] for row in eigenvalues_table], 
                     colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.auto_set_column_width(True)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    plt.show()
    
# Example usage
L_well = 50
dx_well = 0.1
n_levels = 10
x_min = 0
x_max = 50

# Define potentials
x = np.arange(0, L_well, dx_well)
V_cosine = define_potential(x, 'cosine')
V_gaussian = define_potential(x, 'gaussian')
V_qsho = define_potential(x, 'qsho', mass=1, omega=1.0)

print("QSHO Energy Levels:")
for n in range(1, 5):
    E_qsho = qsho_energy_analytical(n, L_well, mass=1)
    print(f"n={n}: {E_qsho:.5e} J")
    
# Plot and compare for zero potential
print("Zero Potential:")
plot_infinite_potential_well(L_well, dx_well, n_levels, mass=1, x_min=x_min, x_max=x_max)
display_eigenvalues_table(L_well, dx_well, n_levels, mass=1, x_min=x_min, x_max=x_max, potential_type='zero')

# Plot and compare for cosine potential
print("Cosine Potential:")
plot_infinite_potential_well(L_well, dx_well, n_levels, mass=1, V=V_cosine, x_min=x_min, x_max=x_max)
display_eigenvalues_table(L_well, dx_well, n_levels, mass=1, V=V_cosine, x_min=x_min, x_max=x_max, potential_type='cosine')

#Plot and compare for gaussian potential
print("Gaussian Potential:")
plot_infinite_potential_well(L_well, dx_well, n_levels, mass=1, V=V_gaussian, x_min=x_min, x_max=x_max)
display_eigenvalues_table(L_well, dx_well, n_levels, mass=1, V=V_gaussian, x_min=x_min, x_max=x_max)

# Plot and compare for QSHO potential
print("QSHO Potential:")
plot_infinite_potential_well(L_well, dx_well, n_levels, mass=1, V=V_qsho, x_min=x_min, x_max=x_max)
display_eigenvalues_table(L_well, dx_well, n_levels, mass=1, V=V_qsho, x_min=x_min, x_max=x_max, potential_type='qsho')

def superposition_time_evolution(c, n_levels, L, dx, mass=m_e, V=None, x_min=0, x_max=None, t=0):
    """
    Computes the superposition of eigenfunctions with given amplitudes and phases, including time evolution.
    
    Parameters:
    c (list of tuples): List of tuples (c_n, phi_n) where c_n is the amplitude and phi_n is the phase for the nth eigenfunction.
    n_levels (int): Number of energy levels to include in the superposition.
    L (float): Length of the potential well.
    dx (float): Grid spacing.
    mass (float): Mass of the particle.
    V (array): Potential array.
    x_min (float): Minimum x value.
    x_max (float): Maximum x value.
    t (float): Time value for the time evolution.
    
    Returns:
    x (array): Position array.
    psi_t (array): Superposition wavefunction at time t.
    """
    if x_max is None:
        x_max = L

    x = np.arange(x_min, x_max, dx)
    psi_t = np.zeros_like(x, dtype=complex)

    for n in range(1, n_levels + 1):
        c_n, phi_n = c[n-1]
        _, psi_n, eigenvalues = infinite_potential_well(n, L, dx, x_min, x_max, V=V, method='second-order', mass=mass)
        E_n = eigenvalues[n-1]
        psi_t += c_n * psi_n * np.exp(-1j * (E_n * t / hbar + phi_n))

    return x, psi_t

# Define the initial wave packet
def initial_wave_packet(x, x0, k0, sigma):
    """Creates an initial Gaussian wave packet."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Parameters
L_well = 50  # Length of the well
dx_well = 1 # Grid spacing for the well (reduced for better resolution)
n_levels = 10  # Number of energy levels to plot
x_min = 0  # Minimum x value
x_max = 50  # Maximum x value

# Initial wave packet parameters
x = np.arange(x_min, x_max, dx_well)
x0 = 25  # Initial position of the wave packet (set away from the boundary)
k0 = 10e-10  # Wave number for momentum
sigma = 1.0  # Width of the wave packet (increased for better representation)
psi_0 = initial_wave_packet(x, x0, k0, sigma)

# Normalize the initial wave packet
psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2) * dx_well)

# Decompose the initial wave packet into eigenfunctions
c = []
for n in range(1, n_levels + 1):
    _, psi_n, _ = infinite_potential_well(n, L_well, dx_well, x_min, x_max, V=None, method='second-order', mass=1)
    c_n = np.dot(psi_0, psi_n) * dx_well
    phi_n = 0  # Initial phase
    c.append((c_n, phi_n))

# Animation setup
fig, ax = plt.subplots()
line_prob, = ax.plot([], [], lw=2, label='Probability density')
particle_most_probable, = ax.plot([], [], 'bo', label='Most Probable', markersize=5)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.7)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Wavefunction $\psi(x, t)$')
ax.set_title('Superposition Time Evolution')
ax.grid(True)
ax.legend(loc='upper right')

# Plot vertical lines for the potential
plt.axvline(x=x_min, color='k', linestyle='-', linewidth=1)
plt.axvline(x=x_max, color='k', linestyle='-', linewidth=1)
plt.text(x_min, 2 * 1.1, r'$\infty$', fontsize=12, ha='center')
plt.text(x_max, 2 * 1.1, r'$\infty$', fontsize=12, ha='center')

def init():
    line_prob.set_data([], [])
    particle_most_probable.set_data([], [])
    return line_prob, particle_most_probable

def animate(t):
    x, psi_t = superposition_time_evolution(c, n_levels, L_well, dx_well, mass=1, V=None, x_min=x_min, x_max=x_max, t=3*t*10e-33)
    prob_density = np.abs(psi_t)**2
    line_prob.set_data(x, prob_density)
    
    # Update y-axis limits based on max probability density
    max_prob_density = np.max(prob_density)
    ax.set_ylim(-1.5 * max_prob_density, 1.5 * max_prob_density)
    
    # Find the most probable position for the particle
    particle_most_probable_position = x[np.argmax(prob_density)]
    particle_most_probable.set_data(particle_most_probable_position, 0.5 * max_prob_density)
    
    return line_prob, particle_most_probable

ani = FuncAnimation(fig, animate, init_func=init, frames=np.linspace(0, 2 * np.pi, 600), interval=100, blit=True)

plt.show()