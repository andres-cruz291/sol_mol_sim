import numpy as np
from tqdm import tqdm, trange
from forces.potentials import harmonic_force, fene_force, kinetic_energy, potential_energy, yukawa_force
from utils.initial_conditions import (
    #initialize_particles,
    initialize_two_particles,
    #linear_chain,
    #collapsed_chain,
    set_maxwell_boltzmann
)


def velocity_verlet_random_momenta(N, dt, t_sim, k=1.0, m=1.0, epsilon=1.0, seed=None, config="linear"):
    if seed is not None:
        np.random.seed(seed)

    steps = int(t_sim / dt)
    a = 1.0

    r = np.zeros((N, 3))
    v = np.zeros((N, 3))

    if config == "linear":
        for i in range(N):
            r[i, 0] = i * a
    elif config == "collapsed":
        for i in range(N):
            r[i, 0] = 0
    else:
        print("configuration not recognized.")
        

    # Random initial velocities (3d)
    v[:] = np.random.normal(0, np.sqrt(epsilon / m), size=(N, 3))

    r_traj = np.zeros((steps, N, 3))

    def compute_forces(r):
        f = np.zeros_like(r)
        for i in range(1, N - 1):
            f[i] += -k * (r[i] - r[i - 1])
            f[i] += -k * (r[i] - r[i + 1])
        f[0] += -k * (r[0] - r[1])
        f[N - 1] += -k * (r[N - 1] - r[N - 2])
        return f

    f = compute_forces(r)

    for step in trange(steps, desc="integration process"):
        r_traj[step] = r.copy()
        v += 0.5 * dt * f / m
        r += dt * v
        f = compute_forces(r)
        v += 0.5 * dt * f / m

    return r_traj

def velocity_verlet_simulation(N=2, dt=0.01, t_sim=10.0, k=1.0, m=1.0, beta=1.0, 
                                config="linear", random_momenta=False, 
                                return_energy=False, track_observables=False, 
                                seed=None, potential_type="harmonic"):

    if seed is not None:
        np.random.seed(seed)

    steps = int(t_sim / dt)
    a = 1.0
    epsilon = 1 / beta

    # --- Particle Initialization ---
    if N == 2:
        r, v = initialize_two_particles()
    else:
        r = np.zeros((N, 3))
        if config == "linear":
            for i in range(N):
                r[i, 0] = i * a
        elif config == "collapsed":
            pass  # already zero
        else:
            raise ValueError("Unknown configuration")

        v = np.zeros((N, 3))
        if random_momenta:
            v[:] = np.random.normal(0, np.sqrt(epsilon / m), size=(N, 3))
        else:
            v[0, 0] = +1.0
            v[-1, 0] = -1.0
    
    if potential_type == "harmonic":
        force_fn = harmonic_force
    elif potential_type == "fene":
        force_fn = fene_force
    else:
        raise ValueError("Unknown potential type")

    f = force_fn(r)

    # --- Data storage ---
    r_traj = np.zeros((steps, N, 3))
    v_traj = np.zeros((steps, N, 3)) if return_energy else None
    kin_energy = np.zeros(steps) if return_energy else None
    P_traj, J_traj, E_traj, time = [], [], [], []

    # --- Time Integration ---
    for step in trange(steps, desc="Integrating"):
        r_traj[step] = r.copy()
        if return_energy:
            v_traj[step] = v.copy()
            kin_energy[step] = 0.5 * m * np.sum(v**2)

        v += 0.5 * dt * f / m
        r += dt * v
        f = force_fn(r)
        v += 0.5 * dt * f / m

        if track_observables:
            KE = 0.5 * m * np.sum(v**2)
            PE = 0.5 * k * np.sum((r[1:] - r[:-1])**2)
            ptot = np.sum(m * v, axis=0)
            J = np.sum(np.cross(r, v), axis=0)
            P_traj.append(ptot)
            J_traj.append(J)
            E_traj.append(KE + PE)
            time.append((step + 1) * dt)

    # --- Return values ---
    result = {"r_traj": r_traj}
    if return_energy:
        result["v_traj"] = v_traj
        result["kin_energy"] = kin_energy
    if track_observables:
        result.update({
            "P_traj": np.array(P_traj),
            "J_traj": np.array(J_traj),
            "E_traj": np.array(E_traj),
            "time": np.array(time),
        })
    if N == 2:
        result["r1_traj"] = r_traj[:, 0]
        result["r2_traj"] = r_traj[:, 1]

    return result


def velocity_verlet_simulation_pbc(N=11, dt=0.01, t_sim=10.0, k=1.0, m=1.0, beta=1.0, 
                                    config="linear", random_momenta=False, return_energy=False, 
                                    seed=None, L=None, apply_pbc=True, close_chain=False):

    if seed is not None:
        np.random.seed(seed)

    steps = int(t_sim / dt)
    a = 1.0
    epsilon = 1 / beta
    if L is None:
        L = N * a

    r = np.zeros((N, 3))
    r[:, 0] = np.arange(N) * a - L / 2  # centered initialization
    v = np.zeros((N, 3))

    if random_momenta:
        v[:] = np.random.normal(0, np.sqrt(epsilon / m), size=(N, 3))
    else:
        v[N // 2] = np.array([-3/7, 6/7, -2/7]) * np.sqrt(epsilon / m)

    def apply_pbc_fn(r):
        return (r + L / 2) % L - L / 2 if apply_pbc else r

    def compute_forces(r):
        return yukawa_force(r)
        f = np.zeros_like(r)
        for i in range(N - 1):
            rij = r[i + 1] - r[i]
            if apply_pbc:
                rij -= L * np.round(rij / L)
            dist = np.linalg.norm(rij)
            if dist > 0:
                f_ij = k * (dist - a) * rij / dist
                f[i] += f_ij
                f[i + 1] -= f_ij
        if close_chain:
            rij = r[0] - r[-1]
            if apply_pbc:
                rij -= L * np.round(rij / L)
            dist = np.linalg.norm(rij)
            if dist > 0:
                f_ij = k * (dist - a) * rij / dist
                f[-1] += f_ij
                f[0] -= f_ij
        return f

    def compute_potential_energy(r):
        U = 0.0
        for i in range(N - 1):
            rij = r[i + 1] - r[i]
            if apply_pbc:
                rij -= L * np.round(rij / L)
            dist = np.linalg.norm(rij)
            U += 0.5 * k * (dist)**2
        if close_chain:
            rij = r[0] - r[-1]
            if apply_pbc:
                rij -= L * np.round(rij / L)
            dist = np.linalg.norm(rij)
            U += 0.5 * k * (dist - a)**2
        return U

    # Storage
    r_traj = np.zeros((steps, N, 3))
    energy_traj = np.zeros(steps) if return_energy else None
    energy_pot = np.zeros(steps) if return_energy else None

    # Initial force
    f = compute_forces(r)
    return 0
    P_traj, J_traj, E_traj, time = [], [], [], []
    for step in trange(steps, desc="Integrating"):
        r_traj[step] = r.copy()

        v += 0.5 * dt * f / m
        r += dt * v
        r = apply_pbc_fn(r)
        f = compute_forces(r)
        v += 0.5 * dt * f / m

        if return_energy:
            KE = 0.5 * m * np.sum(v**2)
            PE = compute_potential_energy(r)
            energy_traj[step] = KE 
            energy_pot[step] = PE
            ptot = np.sum(m * v, axis=0)
            J = np.sum(np.cross(r, v), axis=0)
            P_traj.append(ptot)
            J_traj.append(J)
            E_traj.append(KE + PE)
            time.append((step + 1) * dt)


    result = {"r_traj": r_traj}

    if return_energy:
        result["energy_traj"] = energy_traj
        result["energy_pot"] = energy_pot
        result.update({
            "P_traj": np.array(P_traj),
            "J_traj": np.array(J_traj),
            "E_traj": np.array(E_traj),
            "time": np.array(time),
        })
    return result

