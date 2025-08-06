import numpy as np
from utils.initial_conditions import epsilon, a, k, m, A_yukawa
        
def harmonic_force(r):
    f = np.zeros_like(r)
    N = len(r)
    for i in range(1, N - 1):
        f[i] += -k * (r[i] - r[i - 1])
        f[i] += -k * (r[i] - r[i + 1])
    f[0] += -k * (r[0] - r[1])
    f[N - 1] += -k * (r[N - 1] - r[N - 2])
    return f

def fene_force(r):
    f = np.zeros_like(r)
    N = len(r)
    if N != 2:
        raise ValueError("FENE force currently only implemented for N=2")
    r12 = r[0] - r[1]
    r2_sq = np.dot(r12, r12)
    denom = a**2 - r2_sq
    if denom <= 1e-12:
        denom = 1e-12  # prevent divergence
    f12 = -epsilon * r12 / denom
    f[0] = f12
    f[1] = -f12
    return f

def yukawa_force_loop(positions):
    N = positions.shape[0]
    forces = np.zeros((N, 3))

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r == 0:
                continue  # avoid division by zero
            f_mag = A_yukawa * np.exp(-k * r) * (k * r + 1) / r**3
            f_vec = f_mag * r_vec
            forces[i] += f_vec
            forces[j] -= f_vec  # Newton's third law

    return forces

def yukawa_force(positions):
    """f = np.zeros_like(r)
    N = len(r)
    # Compute all pairwise differences
    rij = r[:, np.newaxis, :] - r[np.newaxis, :, :]  # shape (N, N, 3)
    dist = np.linalg.norm(rij, axis=2)  # shape (N, N)
    # Avoid self-interaction
    np.fill_diagonal(dist, np.inf)
    mask = dist > 1e-12
    mask &= np.isfinite(dist)
    # Compute force magnitude for valid pairs
    force_mag = np.zeros_like(dist)
    force_mag[mask] = A_yukawa * np.exp(-k * dist[mask]) * (1 + k * dist[mask]) / (dist[mask]**3)
    # Expand force_mag for broadcasting
    force_mag_expanded = force_mag[..., np.newaxis]  # shape (N, N, 1)
    # Compute force vectors
    f = np.sum(force_mag_expanded * rij, axis=1)
    return f"""
    """N = positions.shape[0]
    forces = np.zeros((N, 3))

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r == 0:
                continue  # avoid division by zero
            f_mag = A_yukawa * np.exp(-k * r) * (k * r + 1) / r**3
            f_vec = f_mag * r_vec
            forces[i] += f_vec
            forces[j] -= f_vec  # Newton's third law

    return forces"""
    f = np.zeros_like(positions)
    N = len(positions)
    # Compute all pairwise differences
    rij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape (N, N, 3)
    dist = np.linalg.norm(rij, axis=2)  # shape (N, N)
    # Avoid self-interaction
    np.fill_diagonal(dist, np.inf)
    mask = dist > 1e-12
    mask &= np.isfinite(dist)
    # Compute force magnitude for valid pairs
    force_mag = np.zeros_like(dist)
    force_mag[mask] = A_yukawa * np.exp(-k * dist[mask]) * (1 + k * dist[mask]) / (dist[mask]**3)
    # Expand force_mag for broadcasting
    force_mag_expanded = force_mag[..., np.newaxis]  # shape (N, N, 1)
    #print(force_mag_expanded)
    #quit()
    # Compute force vectors
    f = np.sum(force_mag_expanded * rij, axis=1)
    return f

def kinetic_energy(particles):
    return sum(0.5 * p['mass'] * np.dot(p['v'], p['v']) for p in particles)

def potential_energy(particles, type="harmonic"):
    r1, r2 = particles[0]['r'], particles[1]['r']
    if type == "harmonic":
        return 0.5 * np.sum((r1 - r2)**2)
    elif type == "fene":
        r2_sq = np.sum((r1 - r2)**2)
        arg = 1 - r2_sq / a**2
        arg = np.clip(arg, 1e-12, 1.0)
        return -0.5 * epsilon * np.log(arg)
    else:
        raise ValueError("Unknown potential type")

def potential_energy_per_particle(positions_traj, potential_fn, mass=1.0):
    N = positions_traj.shape[1]
    Epot_list = []
    for pos in positions_traj:
        particles = [{'r': r, 'v': np.zeros(3), 'f': np.zeros(3), 'mass': mass} for r in pos]
        Epot = potential_fn(particles, return_energy=True)
        Epot_list.append(Epot / N)
    return np.array(Epot_list)

def harmonic_chain_force(particles, return_energy=False):
    """
    Applies harmonic forces along the chain: between each pair of neighbors.
    Optionally returns total potential energy.
    """
    N = len(particles)
    # Zero all forces
    for p in particles:
        p['f'][:] = 0.0

    E_pot = 0.0
    for i in range(N - 1):
        r_i = particles[i]['r']
        r_j = particles[i + 1]['r']
        diff = r_i - r_j
        f = -k * diff

        particles[i]['f'] += f
        particles[i + 1]['f'] -= f

        if return_energy:
            E_pot += 0.5 * k * np.sum(diff**2)

    if return_energy:
        return E_pot
