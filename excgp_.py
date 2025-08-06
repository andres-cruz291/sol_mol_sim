import numpy as np
import matplotlib.pyplot as plt

def compute_forces(positions, potential='harmonic', epsilon=1.0, sigma=1.0, k=1.0, alpha=1.0):
    """Compute forces for different potentials in a vectorized way."""
    n = len(positions)
    rij = positions[:, None] - positions  # Pairwise distances
    np.fill_diagonal(rij, np.inf)  # Avoid self-interactions

    if potential == 'harmonic':
        forces = -k * rij
    elif potential == 'lennard-jones':
        r6 = (sigma / np.abs(rij)) ** 6
        r12 = r6 ** 2
        force_mag = 24 * epsilon * (2 * r12 - r6) / np.abs(rij) ** 2
        forces = force_mag * np.sign(rij)
    elif potential == 'yukawa':
        r = np.abs(rij)
        force_mag = epsilon * (alpha + 1/r) * np.exp(-alpha * r) / r**2
        forces = force_mag * np.sign(rij)
    else:
        raise ValueError("Unknown potential")

    return np.sum(forces, axis=1)

def potential_energy(positions, potential='harmonic', epsilon=1.0, sigma=1.0, k=1.0, alpha=1.0):
    """Compute total potential energy."""
    rij = positions[:, None] - positions
    np.fill_diagonal(rij, np.inf)
    r = np.abs(rij)

    if potential == 'harmonic':
        energy = 0.5 * k * r**2
    elif potential == 'lennard-jones':
        r6 = (sigma / r) ** 6
        r12 = r6 ** 2
        energy = 4 * epsilon * (r12 - r6)
    elif potential == 'yukawa':
        energy = epsilon * np.exp(-alpha * r) / r
    else:
        raise ValueError("Unknown potential")

    return np.sum(energy) / 2  # divide by 2 to avoid double counting

def simulate(positions, velocities, dt, steps, potential='harmonic', **kwargs):
    """Run simulation and return total energies."""
    energies = []
    pos = positions.copy()
    vel = velocities.copy()

    for _ in range(steps):
        forces = compute_forces(pos, potential=potential, **kwargs)
        pos += vel * dt + 0.5 * forces * dt**2
        new_forces = compute_forces(pos, potential=potential, **kwargs)
        vel += 0.5 * (forces + new_forces) * dt

        ke = 0.5 * np.sum(vel**2)
        pe = potential_energy(pos, potential=potential, **kwargs)
        energies.append(ke + pe)

    return np.array(energies)

def run_analysis(potential='harmonic'):
    """Run simulation for multiple time steps and plot scaled energy deviation."""
    n_particles = 5
    np.random.seed(0)
    initial_positions = np.random.rand(n_particles)
    initial_velocities = np.random.randn(n_particles)

    timesteps = np.array([0.01, 0.005, 0.0025, 0.00125])
    order = 2  # Velocity Verlet is 2nd order
    max_steps = 1000

    plt.figure(figsize=(8, 5))
    for dt in timesteps:
        energies = simulate(initial_positions, initial_velocities, dt, max_steps, potential=potential)
        energy_deviation = np.abs(energies - energies[0])
        scaled_deviation = energy_deviation / dt**order
        plt.plot(scaled_deviation, label=f"dt={dt}")

    plt.title(f"Scaled Energy Deviation ({potential.capitalize()} Potential)")
    plt.xlabel("Time Step")
    plt.ylabel("Energy Deviation / dt²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Choose one of: 'harmonic', 'lennard-jones', 'yukawa'
run_analysis(potential='harmonic')
