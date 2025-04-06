import numpy as np
from scipy.sparse import diags, kron
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def create_potential_barriers_2D(grid_x, grid_y, barriers):
    """
    Create a 2D potential barrier grid.
    
    Parameters:
        grid_x, grid_y: 1D arrays representing the x and y coordinates of the grid.
        barriers: List of dictionaries, each specifying a barrier with keys:
                  - 'x_range': (start_x, end_x)
                  - 'y_range': (start_y, end_y)
                  - 'height': Potential height of the barrier.
    
    Returns:
        potential: 2D array of potential values.
    """
    potential = np.zeros((len(grid_x), len(grid_y)))
    
    for barrier in barriers:
        x_start, x_end = barrier['x_range']
        y_start, y_end = barrier['y_range']
        height = barrier['height']
        
        x_indices = (grid_x >= x_start) & (grid_x < x_end)
        y_indices = (grid_y >= y_start) & (grid_y < y_end)
        
        potential[np.ix_(x_indices, y_indices)] = height
    
    # Set very large finite potential at the boundaries
    potential[0, :] = 1e10
    potential[-1, :] = 1e10
    potential[:, 0] = 1e10
    potential[:, -1] = 1e10
    
    return potential

def gaussian_wavepacket_2D(grid_x, grid_y, x0, y0, kx, ky, sigma_x=0.1, sigma_y=0.1, amplitude=1.0):
    """
    Create a 2D Gaussian wave packet.
    
    Parameters:
        grid_x, grid_y: 1D arrays representing the x and y coordinates of the grid.
        x0, y0: Initial position of the wave packet.
        kx, ky: Wave vector components.
        sigma_x, sigma_y: Width of the wave packet in x and y directions.
        amplitude: Amplitude of the wave packet.
    
    Returns:
        psi0: 2D array representing the initial wave packet.
    """
    X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    gaussian = (np.sqrt(1 / (np.sqrt(np.pi) * sigma_x * sigma_y)) *
                np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))) * amplitude)
    return np.exp(1j * (kx * (X - x0) + ky * (Y - y0))) * gaussian

def CrankNicolson_2D(psi0, V, grid_x, grid_y, dt, N=100, print_norm=False):
    """
    Perform time evolution using the Crank-Nicolson method in 2D.
    
    Parameters:
        psi0: Initial wave function (2D array).
        V: Potential (2D array).
        grid_x, grid_y: 1D arrays representing the x and y coordinates of the grid.
        dt: Time step.
        N: Number of time steps.
        print_norm: Whether to print the norm at each time step.
    
    Returns:
        PSI_t: 3D array of wave function values at each time step.
    """
    Jx, Jy = len(grid_x) - 1, len(grid_y) - 1
    dx, dy = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]

    # Debugging: Print grid and step sizes
    print(f"Grid size: Jx={Jx+1}, Jy={Jy+1}")
    print(f"dx={dx}, dy={dy}")

    # Ensure psi0 and V have the correct shapes
    if psi0.shape != (Jx+1, Jy+1):
        raise ValueError(f"psi0 must have shape {(Jx+1, Jy+1)}, but got {psi0.shape}")
    if V.shape != (Jx+1, Jy+1):
        raise ValueError(f"V must have shape {(Jx+1, Jy+1)}, but got {V.shape}")

    # Create 1D kinetic energy operators
    T_x = (-1 / (2 * dx**2)) * diags([1, -2, 1], [-1, 0, 1], shape=(Jx+1, Jx+1))
    T_y = (-1 / (2 * dy**2)) * diags([1, -2, 1], [-1, 0, 1], shape=(Jy+1, Jy+1))

    # Debugging: Print shapes of T_x and T_y
    print(f"T_x shape: {T_x.shape}, T_y shape: {T_y.shape}")

    # Construct the full 2D kinetic energy operator using Kronecker products
    T = kron(np.eye(Jy+1), T_x) + kron(T_y, np.eye(Jx+1))

    # Debugging: Print shape of T
    print(f"T shape: {T.shape}")

    # Flatten the potential and construct the potential operator
    V_flat = V.flatten()
    V_diag = diags(V_flat, 0)

    # Debugging: Print shape of V_flat and V_diag
    print(f"V_flat shape: {V_flat.shape}, V_diag shape: {V_diag.shape}")

    # Construct the Crank-Nicolson matrices
    H = T + V_diag
    I = diags([1], [0], shape=H.shape)  # Create an identity matrix of the same shape as H
    U2 = I + (1j * 0.5 * dt) * H
    U1 = I - (1j * 0.5 * dt) * H
    U2 = U2.tocsc()
    LU = splu(U2)

    # Debugging: Print shapes of U1 and U2
    print(f"U1 shape: {U1.shape}, U2 shape: {U2.shape}")

    # Initialize the wave function array
    PSI_t = np.zeros((Jx+1, Jy+1, N), dtype=complex)
    PSI_t[:, :, 0] = psi0

    # Debugging: Print initial shape of PSI_t
    print(f"PSI_t shape: {PSI_t.shape}")

    # Time evolution loop
    for n in range(N-1):
        # Flatten the current wave function to match the shape of U1
        psi_flat = PSI_t[:, :, n].flatten()

        # Debugging: Print shape of psi_flat
        print(f"Time step {n}, psi_flat shape: {psi_flat.shape}")

        if psi_flat.shape[0] != U1.shape[1]:
            raise ValueError(f"Flattened wave function has shape {psi_flat.shape}, but U1 expects {U1.shape[1]}")
        b = U1.dot(psi_flat)

        # Debugging: Print shape of b
        print(f"Time step {n}, b shape: {b.shape}")

        # Solve the linear system and reshape the result back to 2D
        psi_next_flat = LU.solve(b)
        PSI_t[:, :, n+1] = psi_next_flat.reshape((Jx+1, Jy+1))

        # Debugging: Print shape of psi_next_flat
        print(f"Time step {n}, psi_next_flat shape: {psi_next_flat.shape}")

        if print_norm:
            # Compute and print the norm of the wave function
            norm = np.sum(np.abs(PSI_t[:, :, n+1])**2) * dx * dy
            print(f"Time step {n+1}, Norm: {norm}")

    return PSI_t

if __name__ == "__main__":
    # Constants
    x_min, x_max = 0, 40
    y_min, y_max = 0, 10
    grid_x = np.linspace(x_min, x_max, 200)
    grid_y = np.linspace(y_min, y_max, 200)
    dt = 0.1
    total_time = 15
    N = int(total_time / dt)

    # Create potential
    barriers = [
        {'x_range': (15, 15.5), 'y_range': (0, 4.4), 'height': 10000},
        {'x_range': (15, 15.5), 'y_range': (4.55, 5.45), 'height': 10000},
        {'x_range': (15, 15.5), 'y_range': (5.6, 10), 'height': 10000} 
    ]
    potential = create_potential_barriers_2D(grid_x, grid_y, barriers)

    # Create initial wave packet
    x0, y0 = 3, 5
    kx, ky = 0.01, 0
    sigma_x, sigma_y = 0.5, 2
    amplitude = 0.03
    psi0 = gaussian_wavepacket_2D(grid_x, grid_y, x0, y0, kx, ky, sigma_x, sigma_y, amplitude)

    # Perform time evolution
    PSI_t = CrankNicolson_2D(psi0, potential, grid_x, grid_y, dt, N, print_norm=False)

    # Data logger for x=40
    x_logger_index = np.argmin(np.abs(grid_x - 40))  # Find the index of x=40
    logged_data = np.zeros(len(grid_y))  # 1D array to store accumulated logged data for each y

    # Animation setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    wavefunction_magnitude = np.abs(PSI_t[:, :, 0])**2

    # Set color range
    vmin = 0  # Minimum value for the colormap
    vmax = 0.000007  # Maximum value for the colormap (adjust based on your data)

    # Plot the initial wave function
    surface = ax.plot_surface(X, Y, wavefunction_magnitude, cmap='viridis', edgecolor='none', alpha=0.8, vmin=vmin, vmax=vmax)

    # Add barriers as red hollow sections
    for barrier in barriers:
        x_start, x_end = barrier['x_range']
        y_start, y_end = barrier['y_range']
        ax.plot([x_start, x_end, x_end, x_start, x_start],
                [y_start, y_start, y_end, y_end, y_start],
                [0, 0, 0, 0, 0], color='red', linewidth=2)

    # Add a line plot for the logger graph on the x=40 plane
    logger_line, = ax.plot([40] * len(grid_y), grid_y, [0] * len(grid_y), color='red', linewidth=2, label="Logger")

    ax.set_title("Time Evolution of |ψ|² with Logger")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|ψ|²")
    ax.set_zlim(0, 0.0006)
    ax.legend()

    def update(frame):
        global surface, logger_line  # Use the global surface and logger_line objects
        wavefunction_magnitude = np.abs(PSI_t[:, :, frame])**2

        # Accumulate the wavefunction magnitude at x=40 for all y points
        logged_data[:] += wavefunction_magnitude[x_logger_index, :]*10e15

        # Remove the previous surface
        surface.remove()

        # Plot the updated surface
        surface = ax.plot_surface(X, Y, wavefunction_magnitude, cmap='viridis', edgecolor='none', alpha=0.8, vmin=vmin, vmax=vmax)

        # Update the logger line on the x=40 plane
        logger_line.set_data([40] * len(grid_y), grid_y)  # Keep x=40 and y unchanged
        logger_line.set_3d_properties(logged_data)  # Update the z-values with the accumulated data

        # Re-add barriers as red hollow sections
        for barrier in barriers:
            x_start, x_end = barrier['x_range']
            y_start, y_end = barrier['y_range']
            ax.plot([x_start, x_end, x_end, x_start, x_start],
                    [y_start, y_start, y_end, y_end, y_start],
                    [0, 0, 0, 0, 0], color='red', linewidth=2)

        ax.set_title(f"Time Evolution of |ψ|² (Frame {frame})")
        return surface, logger_line

    # Create the animation
    anim = FuncAnimation(fig, update, frames=N, interval=20, blit=False)

    # Show the animation
    plt.show()

    # Plot the accumulated intensity distribution
    plt.figure()
    plt.plot(grid_y, logged_data, label="Accumulated Intensity")
    plt.title("Accumulated Wavefunction Intensity at x=40")
    plt.xlabel("y")
    plt.ylabel("Total Intensity")
    plt.legend()
    plt.grid()
    plt.show()

    