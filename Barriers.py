import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import pandas as pd

# 1. Define helper functions
def create_potential_barriers(num_barriers, barrier_widths, spacings, total_width, barrier_heights):
    x = np.linspace(0, total_width, 1000)
    potential = np.zeros_like(x)
    for i in range(num_barriers):
        start = sum(spacings[:i+1]) + sum(barrier_widths[:i])
        end = start + barrier_widths[i]
        potential[(x >= start) & (x < end)] = barrier_heights[i % len(barrier_heights)]
    potential[0] = 1e10  # Boundary conditions
    potential[-1] = 1e10
    return x, potential

def gaussian_wavepacket(x, x0, k, sigma=0.1, amplitude=1.0):
    g = np.sqrt(1 / np.sqrt(np.pi) / sigma) * np.exp(-(x - x0)**2 / 2 / sigma**2) * amplitude
    return np.exp(1j * k * (x - x0)) * g

def CrankNicolson(psi0, V, x, dt, N=100, print_norm=False):
    J = x.size - 1
    dx = x[1] - x[0]
    V = diags(V)
    O = np.ones(J+1)
    T = (-1 / 2 / dx**2) * diags([O, -2*O, O], [-1, 0, 1], shape=(J+1, J+1))
    U2 = diags([1], [0], shape=(J+1, J+1)) + (1j * 0.5 * dt) * (T + V)
    U1 = diags([1], [0], shape=(J+1, J+1)) - (1j * 0.5 * dt) * (T + V)
    U2 = U2.tocsc()
    LU = splu(U2)
    PSI_t = np.zeros((J+1, N), dtype=complex)
    PSI_t[:, 0] = psi0
    for n in range(N-1):
        b = U1.dot(PSI_t[:, n])
        PSI_t[:, n+1] = LU.solve(b)
        norm = np.trapz(np.abs(PSI_t[:, n+1])**2, x)
        PSI_t[:, n+1] /= np.sqrt(norm)  # Normalize wavefunction
        if print_norm:
            print(n, np.trapz(np.abs(PSI_t[:, n+1])**2, x))
    return PSI_t

def compute_coefficients(PSI_t, x, barrier_positions):
    psi_final = np.abs(PSI_t[:, -1])**2
    reflection_region = x < barrier_positions[0]
    transmission_region = x > barrier_positions[-1]
    R = np.trapz(psi_final[reflection_region], x[reflection_region])
    T = np.trapz(psi_final[transmission_region], x[transmission_region])
    return R, T

def analytical_transmission_multiple_barriers(barrier_heights, barrier_widths, spacings, E, m=1.0, hbar=1.0):
    """
    Generalized analytical transmission for multiple barriers.
    """
    k1 = np.sqrt(2 * m * E) / hbar  # Wave number in free space
    M_total = np.array([[1, 0], [0, 1]])  # Initialize total transfer matrix

    for i in range(len(barrier_heights)):
        # Wave number inside the barrier
        V0 = barrier_heights[i]
        a = barrier_widths[i]
        if E < V0:
            # Energy is less than the barrier height
            B = np.sqrt(2 * m * (V0 - E)) / hbar
            T_single = (16 * E * (V0 - E) / V0**2) * np.exp(-2 * B * a)
            return T_single  # Return immediately for a single barrier case
        else:
            # Energy is greater than the barrier height
            k2 = np.sqrt(2 * m * (E - V0)) / hbar

            # Transfer matrix for the current barrier
            M_barrier = np.array([
                [np.cosh(k2 * a), (1j / (2 * k2)) * np.sinh(k2 * a)],
                [(2 * 1j * k2) * np.sinh(k2 * a), np.cosh(k2 * a)]
            ])

            # Propagation matrix for the spacing after the barrier (if not the last barrier)
            if i < len(spacings):
                d = spacings[i]
                P = np.array([
                    [np.exp(1j * k1 * d), 0],
                    [0, np.exp(-1j * k1 * d)]
                ])
                M_total = M_total @ M_barrier @ P
            else:
                M_total = M_total @ M_barrier

    # Analytical transmission coefficient for multiple barriers
    T_analytical = 1 / np.abs(M_total[0, 0])**2
    return T_analytical

# 2. Define constants and parameters
m = 1.0
hbar = 1.0
num_barriers = 1
barrier_widths = [0.2]
spacings = [5, 3]
barrier_heights = [90]
total_width = sum(spacings) + sum(barrier_widths)

# 3. Create potential barriers
x, potential = create_potential_barriers(num_barriers, barrier_widths, spacings, total_width, barrier_heights)
barrier_positions = [sum(spacings[:i+1]) + sum(barrier_widths[:i]) for i in range(num_barriers)]
barrier_positions.append(barrier_positions[-1] + barrier_widths[-1])

# 4. Initialize wave packet
x0 =2
k0 = 10
sigma = 0.3
amplitude = 0.9
psi0 = gaussian_wavepacket(x, x0, k0, sigma, amplitude)

# 5. Time evolution
total_time = 0.8
dt = 10e-4
N = int(total_time / dt) + 1
PSI_t = CrankNicolson(psi0, potential, x, dt, N)

# 6. Compute coefficients
R_numeric, T_numeric = compute_coefficients(PSI_t, x, barrier_positions)
E = hbar**2 * k0**2 / (2 * m)
T_analytical = analytical_transmission_multiple_barriers(barrier_heights, barrier_widths, spacings, E)

# 7. Compare results
data = {
    "Coefficient": ["Reflection (R)", "Transmission (T)"],
    "Numerical": [R_numeric, T_numeric],
    "Analytical": [1 - T_analytical, T_analytical],
    "Difference (%)": [
        abs((R_numeric - (1 - T_analytical)) / (1 - T_analytical)) * 100,
        abs((T_numeric - T_analytical) / T_analytical) * 100
    ]
}
df = pd.DataFrame(data)

# Format values to 2 decimal places
df = df.round(2)

print(df)

# 8. Plot results
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)  # Adjust the font size as needed
table.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust column widths
plt.show()

# 9. Animate wave packet
fig, ax = plt.subplots()
ax.plot(x, potential, label='Potential Barriers')
line, = ax.plot(x, np.abs(PSI_t[:, 0]), label='Wave Packet')
ax.set_ylim(-amplitude*0.5, amplitude*1.5)
ax.set_xlabel('Position')
ax.set_ylabel('Amplitude')
ax.set_title('Potential Barriers and Gaussian Wave Packet')
ax.legend()

for i in range(num_barriers):
    start = sum(spacings[:i+1]) + sum(barrier_widths[:i])
    end = start + barrier_widths[i]
    height = barrier_heights[i % len(barrier_heights)]
    ax.text((start + end) / 2, amplitude*0.8, f'V = {height}', ha='center', va='bottom')

def animate(i):
    line.set_ydata(np.abs(PSI_t[:, i]))
    return line,

ani = FuncAnimation(fig, animate, frames=N, interval=15, blit=True)
plt.show()