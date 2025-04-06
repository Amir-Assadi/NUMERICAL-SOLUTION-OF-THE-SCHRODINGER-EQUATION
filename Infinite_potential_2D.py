import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import eigh
from scipy.constants import hbar as default_hbar, m_e as default_mass, pi
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm
from scipy.interpolate import RegularGridInterpolator


def second_order_second_derivative_matrix_2d(Nx, Ny, dx, dy):
    """Creates the second-order finite difference matrix for the 2D second derivative."""
    N = Nx * Ny
    diagonals = [-4 * np.ones(N), np.ones(N - 1), np.ones(N - 1), np.ones(N - Nx), np.ones(N - Nx)]
    offsets = [0, -1, 1, -Nx, Nx]

    # Handle boundary conditions
    for i in range(1, Ny):
        diagonals[1][i * Nx - 1] = 0  # Remove coupling between rows
        diagonals[2][i * Nx] = 0

    return diags(diagonals, offsets).toarray() / dx**2


def infinite_potential_well_2d(n, Lx, Ly, dx, dy, V=None, mass=default_mass, hbar=default_hbar):
    """Computes the wavefunction of a particle in a 2D infinite potential well."""
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    N = Nx * Ny

    if V is None:
        V = np.zeros((Ny, Nx))  # Default to zero potential if not provided

    V_flat = V.flatten()
    H = -hbar**2 / (2 * mass) * second_order_second_derivative_matrix_2d(Nx, Ny, dx, dy) + np.diag(V_flat)

    print("Solving eigenvalue problem for 2D...")
    w, v = eigh(H)
    print("Eigenvalue problem solved.")

    psi = v[:, n - 1].reshape((Ny, Nx))
    psi /= np.sqrt(np.sum(psi**2) * dx * dy)  # Normalize the wavefunction

    # Ensure the wavefunction has a positive maximum amplitude
    if np.max(psi) < 0:
        psi *= -1

    return psi, w[n - 1]


def infinite_potential_well_2d_analytical(n_x, n_y, Lx, Ly, mass=default_mass, hbar=default_hbar):
    """Computes the analytical wavefunction and energy of a particle in a 2D infinite potential well."""
    x = np.linspace(0, Lx, 100)
    y = np.linspace(0, Ly, 100)
    X, Y = np.meshgrid(x, y)

    psi_x = np.sqrt(2 / Lx) * np.sin(n_x * pi * X / Lx)
    psi_y = np.sqrt(2 / Ly) * np.sin(n_y * pi * Y / Ly)
    psi = psi_x * psi_y

    energy = (hbar**2 * pi**2 / (2 * mass)) * (n_x**2 / Lx**2 + n_y**2 / Ly**2)
    return X, Y, psi, energy


def define_potential_2d(x, y, potential_type='constant'):
    """Defines different 2D potential functions."""
    X, Y = np.meshgrid(x, y)
    if potential_type == 'constant':
        return np.zeros_like(X)
    elif potential_type == 'cosine':
        return np.cos(X) * np.cos(Y)  
    elif potential_type == 'gaussian':
        return np.exp(-((X - 25)**2 + (Y - 25)**2) / (2 * (3)**2)) * 50
    elif potential_type == 'qsho':
        return 0.5 * default_mass * (X**2 + Y**2) * 10e25
    else:
        raise ValueError('Invalid potential type')


def plot_infinite_potential_well_2d(Lx, Ly, dx, dy, first_n_states, V=None, mass=default_mass, hbar=default_hbar, potential_name=""):
    """
    Plots the wavefunctions and energy levels for a 2D infinite potential well.
    Combines all contour plots and 3D plots for the specified energy levels into one figure.
    Displays a table comparing analytical and numerical energies only for zero potential.
    """
    # Ensure dx and dy divide Lx and Ly evenly
    Lx = round(Lx / dx) * dx
    Ly = round(Ly / dy) * dy

    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)
    X, Y = np.meshgrid(x, y)

    # Compute analytical energies and sort them
    analytical_energies = []
    quantum_numbers = []
    for n_x in range(1, 4):  # Assuming 3 states in x-direction
        for n_y in range(1, 4):  # Assuming 3 states in y-direction
            energy = (hbar**2 * pi**2 / (2 * mass)) * (n_x**2 / Lx**2 + n_y**2 / Ly**2)
            analytical_energies.append(energy)
            quantum_numbers.append((n_x, n_y))
    analytical_energies = np.array(analytical_energies)
    quantum_numbers = np.array(quantum_numbers)
    sorted_indices = np.argsort(analytical_energies)
    analytical_energies = analytical_energies[sorted_indices]
    quantum_numbers = quantum_numbers[sorted_indices]

    # Solve the numerical eigenvalue problem
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    N = Nx * Ny
    if V is None:
        V = np.zeros((Ny, Nx))  # Default to zero potential if not provided
    V_flat = V.flatten()
    H = -hbar**2 / (2 * mass) * second_order_second_derivative_matrix_2d(Nx, Ny, dx, dy) + np.diag(V_flat)
    w, v = eigh(H)

    # Create a 3x3 grid for contour plots
    fig_contour, axes_contour = plt.subplots(3, 3, figsize=(6, 6))
    fig_contour.suptitle(f"Contour Plots for {potential_name} Potential", fontsize=16)
    axes_contour = axes_contour.flatten()

    # Create a 3x3 grid for 3D plots
    fig_3d, axes_3d = plt.subplots(3, 3, figsize=(5, 5), subplot_kw={'projection': '3d'})
    fig_3d.suptitle(f"3D Plots for {potential_name} Potential", fontsize=16)
    axes_3d = axes_3d.flatten()

    # Prepare data for the table (only for zero potential)
    table_data = []
    table_columns = ["State", "Analytical Energy (J)", "Numerical Energy (J)", "Difference (%)", "Dot Product"]

    # Plot the first_n_states
    for state_count in range(first_n_states):
        n_x, n_y = quantum_numbers[state_count]
        E_analytical = abs(analytical_energies[state_count])  # Ensure analytical energy is positive
        psi_numerical = v[:, state_count].reshape((Ny, Nx))
        E_numerical = abs(w[state_count])  # Ensure numerical energy is positive

        # Normalize the numerical wavefunction
        psi_numerical /= np.sqrt(np.sum(psi_numerical**2) * dx * dy)

        # Ensure the wavefunction has a positive maximum amplitude
        if np.max(psi_numerical) < 0:
            psi_numerical *= -1

        # Analytical solution
        x_analytical = np.linspace(0, Lx, 100)
        y_analytical = np.linspace(0, Ly, 100)
        X_analytical, Y_analytical, psi_analytical, _ = infinite_potential_well_2d_analytical(n_x, n_y, Lx, Ly, mass=mass, hbar=hbar)

        # Interpolate the analytical wavefunction onto the numerical grid
        interpolator = RegularGridInterpolator((x_analytical, y_analytical), psi_analytical)
        psi_analytical_interpolated = interpolator((X.flatten(), Y.flatten())).reshape(X.shape)

        # Normalize the interpolated analytical wavefunction
        psi_analytical_interpolated /= np.sqrt(np.sum(psi_analytical_interpolated**2) * dx * dy)

        # Calculate energy difference
        energy_difference = abs(E_analytical - E_numerical) / E_analytical * 100

        # Calculate dot product (overlap) between numerical and analytical wavefunctions
        dot_product = np.abs(np.sum(psi_numerical * psi_analytical_interpolated) * dx * dy)

        # Add data to the table (only for zero potential)
        if potential_name.lower() == "zero":
            table_data.append([f"ψ{n_x}{n_y}", f"{E_analytical:.2e}", f"{E_numerical:.2e}", f"{energy_difference:.2f}%", f"{dot_product:.4f}"])

        # Contour plot
        contour = axes_contour[state_count].contourf(X, Y, psi_numerical, levels=50, cmap='viridis')
        fig_contour.colorbar(contour, ax=axes_contour[state_count], label='Wavefunction Amplitude')
        axes_contour[state_count].set_title(f"ψ{n_x}{n_y}, E={E_numerical:.2e} J")
        axes_contour[state_count].set_xlabel('x')
        axes_contour[state_count].set_ylabel('y')

        # 3D plot
        axes_3d[state_count].plot_surface(X, Y, psi_numerical, cmap='viridis', edgecolor='none')
        axes_3d[state_count].set_title(f"ψ{n_x}{n_y}, E={E_numerical:.2e} J")
        axes_3d[state_count].set_xlabel('x')
        axes_3d[state_count].set_ylabel('y')
        axes_3d[state_count].set_zlabel('Amplitude')

    # Adjust layout for contour plots
    fig_contour.tight_layout()
    fig_contour.subplots_adjust(top=0.90, wspace=0.4, hspace=0.5)  # Increase spacing between subplots

    # Adjust layout for 3D plots
    fig_3d.tight_layout()
    fig_3d.subplots_adjust(top=0.90, wspace=0.4, hspace=0.2)  # Increase spacing between subplots

    # Create a new figure for the table (only for zero potential)
    if potential_name.lower() == "zero":
        fig_table, ax_table = plt.subplots(figsize=(8, 4))
        fig_table.suptitle(f"Energy Comparison Table for {potential_name} Potential", fontsize=16)
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data, colLabels=table_columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(table_columns))))

    # Show all plots
    plt.show()

# Example usage for 2D infinite potential well with different potentials
Lx, Ly = 50, 50  # Length of the well in x and y directions
dx, dy = 0.9, 0.9  # Grid spacing in x and y directions
first_n_states = 9 # Number of states to plot
mass_particle = 1  # Custom mass of the particle (e.g., electron mass)
custom_hbar = 1  # Custom Planck's constant

# Define the potential
x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)
V_cosine = define_potential_2d(x, y, 'cosine')
V_gaussian = define_potential_2d(x, y, 'gaussian')
V_qsho = define_potential_2d(x, y, 'qsho')

# Plot and compare for zero potential
print("Zero Potential:")
plot_infinite_potential_well_2d(Lx, Ly, dx, dy, first_n_states, mass=mass_particle, hbar=custom_hbar, potential_name="Zero")

# Plot and compare for cosine potential
print("Cosine Potential:")
plot_infinite_potential_well_2d(Lx, Ly, dx, dy, first_n_states, V=V_cosine, mass=mass_particle, hbar=custom_hbar, potential_name="Cosine")

# Plot and compare for Gaussian potential
print("Gaussian Potential:")
plot_infinite_potential_well_2d(Lx, Ly, dx, dy, first_n_states, V=V_gaussian, mass=mass_particle, hbar=custom_hbar, potential_name="Gaussian")

# Plot and compare for QSHO potential
print("QSHO Potential:")
plot_infinite_potential_well_2d(Lx, Ly, dx, dy, first_n_states, V=V_qsho, mass=mass_particle, hbar=custom_hbar, potential_name="QSHO")

def animate_gaussian_wave_packet_3d(Lx, Ly, dx, dy, dt, total_time, mass=default_mass, hbar=default_hbar, potential_type='constant'):
    """
    Animates the evolution of a Gaussian wave packet in a 2D infinite potential well with a 3D plot.
    Uses the exact same potentials defined in the `define_potential_2d` function.
    """
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    N = Nx * Ny

    # Define the spatial grid
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Define the initial Gaussian wave packet
    x0, y0 = Lx / 2, Ly / 2  # Initial position of the wave packet
    sigma_x, sigma_y = Lx / 10, Ly / 10  # Width of the wave packet
    kx, ky = 0, 0  # Initial momentum components
    psi0 = np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))) * np.exp(1j * (kx * X + ky * Y))
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx * dy)  # Normalize the wavefunction

    # Define the potential using the `define_potential_2d` function
    V = define_potential_2d(x, y, potential_type)
    V_flat = V.flatten()

    # Define the Hamiltonian
    H = -hbar**2 / (2 * mass) * second_order_second_derivative_matrix_2d(Nx, Ny, dx, dy) + np.diag(V_flat)

    # Time evolution operator
    U = expm(-1j * H * dt / hbar)

    # Flatten the initial wavefunction
    psi = psi0.flatten()

    # Set up the figure for animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Time Evolution of Gaussian Wave Packet (3D) with {potential_type.capitalize()} Potential")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Probability Density")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, 0.05)  # Adjust based on the expected maximum probability density

    # Plot the initial probability distribution
    prob = np.abs(psi0)**2
    surface = [ax.plot_surface(X, Y, prob, cmap='viridis', edgecolor='none')]

    def update(frame):
        nonlocal psi, surface
        psi = U @ psi  # Apply the time evolution operator
        prob = np.abs(psi.reshape((Ny, Nx)))**2

        # Remove the previous surface
        for surf in surface:
            surf.remove()

        # Plot the updated surface
        surface[0] = ax.plot_surface(X, Y, prob, cmap='viridis', edgecolor='none')
        return surface

    # Create the animation
    frames = int(total_time / dt)
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    plt.show()

# Example usage of the 3D animation function
Lx, Ly = 50, 50  # Length of the well in x and y directions
dx, dy = 1, 1  # Grid spacing in x and y directions
mass_particle = 1  # Custom mass of the particle (e.g., electron mass)
custom_hbar = 1  # Custom Planck's constant
dt = 1  # Time step
total_time = 100 # Total simulation time

# Animate with a constant potential
animate_gaussian_wave_packet_3d(Lx=50, Ly=50, dx=1, dy=1, dt=1, total_time=100, mass=1, hbar=1, potential_type='constant')

# Animate with a cosine potential
animate_gaussian_wave_packet_3d(Lx=50, Ly=50, dx=1, dy=1, dt=1, total_time=100, mass=1, hbar=1, potential_type='cosine')

# Animate with a Gaussian potential
animate_gaussian_wave_packet_3d(Lx=50, Ly=50, dx=1, dy=1, dt=1, total_time=100, mass=1, hbar=1, potential_type='gaussian')

# Animate with a QSHO potential
animate_gaussian_wave_packet_3d(Lx=50, Ly=50, dx=1, dy=1, dt=1, total_time=100, mass=1, hbar=1, potential_type='qsho')