import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Simulation parameters
N = 3  # Number of particles
L = 6.8399037867  # Box length
rho = N / L**3
timesteps = [0.001, 0.002, 0.004, 0.008]  # Different timesteps to test
#steps = 500  # Simulation steps
max_time = timesteps[0]*5000

# Yukawa potential parameters
A = 1.0
k = 1.0
rcut = L / 2.0  # Cutoff distance

def minimum_image(rij, box_size):
    return rij - box_size * np.round(rij / box_size)

def compute_forces(positions, box_size):
    forces = np.zeros_like(positions)
    forces_t = np.zeros_like(positions)
    potential_energy = 0.0

    # Compute all pairwise displacement vectors using broadcasting
    rij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    rij = minimum_image(rij, box_size)
    dist = np.linalg.norm(rij, axis=-1)
    #print(rij)
    #print(dist)

    # Mask out self-interactions and apply cutoff
    #mask = (dist < rcut) & (dist > 0)
    mask = dist > 0
    # Yukawa potential and force calculations
    r = dist[mask]
    r_vec = rij[mask]
    exp_term = np.exp(-k * r)
    #print(r)
    #print(exp_term)
    force_scalar = A * exp_term * (k + 1) / r**3
    #print(force_scalar)
    force_vec = (r_vec.T * force_scalar).T
    #print(force_vec)
    # Indices of interacting pairs
    idx_i, idx_j = np.where(mask)
    
    # Accumulate forces (Newton's third law)
    np.add.at(forces, idx_i, force_vec)
    np.subtract.at(forces, idx_j, force_vec)
    #print("forces tmp", forces)
    
    #print("forces tmp 2", forces_t)
    #forces += forces_t
    

    #print(dist.shape, mask.shape, r.shape, r_vec.shape, idx_i.shape, idx_j.shape)
    #print(mask)
    #print(idx_i)
    #print(idx_j)
    #print(forces)
    #quit()

    # Potential energy (avoid double counting)
    potential_energy += np.sum(A * exp_term / r) * 0.5

    return forces, potential_energy

def compute_forces_tmp(positions, box_size):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            rij = positions[i] - positions[j]
            rij = minimum_image(rij, box_size)
            r = np.linalg.norm(rij)

            if r > 0:
                exp_term = np.exp(-k * r)
                force_scalar = A * exp_term * (k + 1 ) / r**3
                force_vec = rij * force_scalar
                forces[i] += force_vec
                forces[j] -= force_vec
                potential_energy += A * exp_term / r
    print("forces tmp", forces)
    return forces, potential_energy

def compute_forces_tmp_2(positions, box_size):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(len(positions) - 1):
        rij = positions[i] - positions[i+1:]
        rij = minimum_image(rij, box_size)
        dist = np.linalg.norm(rij, axis=1)

        within_cutoff = dist < rcut
        r = dist[within_cutoff]
        r_vec = rij[within_cutoff]

        # Yukawa potential and force
        exp_term = np.exp(-k * r)
        force_scalar = A * exp_term * (k + 1 ) / r**3
        force_vec = (r_vec.T * force_scalar).T

        forces[i] += np.sum(force_vec, axis=0)
        forces[i+1:][within_cutoff] -= force_vec

        potential_energy += np.sum(A * exp_term / r)
    return forces, potential_energy

def velocity_verlet(positions, velocities, dt, box_size):
    forces, potential_energy = compute_forces(positions, box_size)
    positions += velocities * dt + 0.5 * forces * dt**2
    positions %= box_size  # Apply periodic boundaries
    new_forces, potential_energy = compute_forces(positions, box_size)
    velocities += 0.5 * (forces + new_forces) * dt
    return positions, velocities, new_forces, potential_energy

def simulate(dt, steps):
    positions = positions_init.copy()
    velocities = velocities_init.copy()
    forces, potential_energy = compute_forces(positions, L)
    forces_tmp, potential_energy_tmp = compute_forces_tmp(positions, L)
    forces_tmp2, potential_energy_tmp2 = compute_forces_tmp_2(positions, L)
    #print(np.allclose(forces, forces_tmp), np.allclose(potential_energy, potential_energy_tmp))
    print(forces-forces_tmp)
    print(forces_tmp-forces_tmp2)
    quit()

    kinetic_energy = 0.5 * np.sum(velocities**2)
    total_energy_initial = kinetic_energy + potential_energy
    energies = []
    time = []

    for step in trange(steps, desc="Integrating"):
        positions, velocities, forces, potential_energy = velocity_verlet(positions, velocities, dt, L)
        kinetic_energy = 0.5 * np.sum(velocities**2)
        total_energy = kinetic_energy + potential_energy
        energies.append(total_energy - total_energy_initial)  # Energy deviation
        time.append(step * dt)

    return np.array(energies), np.array(time)

# Run simulation for each timestep and plot energy deviation
plt.figure(figsize=(8, 6))
positions_init = np.random.rand(N, 3) * L
#positions_init = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
velocities_init = np.random.rand(N, 3)
for dt in timesteps:
    steps = int(max_time / dt)
    energy_dev,time = simulate(dt, steps)
    
    #plt.plot(time, energy_dev/dt**2, label=f'dt = {dt}')
    plt.plot(time, energy_dev, label=f'dt = {dt}')

plt.xlabel('Step')
plt.ylabel('Energy Deviation')
plt.title('Energy Deviation vs Time for Different Timesteps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
