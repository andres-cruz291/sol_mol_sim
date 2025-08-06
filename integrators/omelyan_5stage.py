import numpy as np
from forces.potentials import harmonic_force, fene_force, potential_energy, kinetic_energy, yukawa_force


# --- 5-Stage Integrator ---
def simulate_5stage(dt, t_sim=10.0, potential_type="harmonic", lambda5=0.1931833275037836, 
                    k=1.0, m=1.0, beta=1.0, a=1.0, epsilon=None):
    if epsilon is None:
        epsilon = 1 / beta

    n_steps = int(t_sim / dt)
    particles = [
        {'r': np.array([a/3, 0.0, 0.0]), 'v': np.array([-6/7, 3/7, -2/7]) * np.sqrt(m * epsilon), 'f': np.zeros(3), 'mass': m},
        {'r': np.array([-a/3, 0.0, 0.0]), 'v': np.array([6/7, -3/7, 2/7]) * np.sqrt(m * epsilon), 'f': np.zeros(3), 'mass': m}
    ]

    # Select appropriate force function
    if potential_type == "harmonic":
        def force_fn(particles):
            r = np.array([p['r'] for p in particles])
            f = harmonic_force(r)
            for i, p in enumerate(particles):
                p['f'] = f[i]
    elif potential_type == "fene":
        def force_fn(particles):
            r = np.array([p['r'] for p in particles])
            f = fene_force(r)
            for i, p in enumerate(particles):
                p['f'] = f[i]
    elif potential_type == "yukawa":
        def force_fn(particles):
            r = np.array([p['r'] for p in particles])
            f = yukawa_force(r)
            for i, p in enumerate(particles):
                p['f'] = f[i]
    else:
        raise ValueError("Unknown potential type")

    # Initial force evaluation
    force_fn(particles)

    r1_traj, r2_traj = [particles[0]['r'].copy()], [particles[1]['r'].copy()]
    P_traj, J_traj, E_traj = [], [], []
    time = [0.0]

    for step in range(n_steps):
        # Stage 1: First kick
        for p in particles:
            p['v'] += lambda5 * dt * p['f'] / p['mass']

        # Stage 2: First drift
        for p in particles:
            p['r'] += 0.5 * dt * p['v']

        # Stage 3: Middle kick
        force_fn(particles)
        for p in particles:
            p['v'] += (1 - 2 * lambda5) * dt * p['f'] / p['mass']

        # Stage 4: Second drift
        for p in particles:
            p['r'] += 0.5 * dt * p['v']

        # Stage 5: Final kick
        force_fn(particles)
        for p in particles:
            p['v'] += lambda5 * dt * p['f'] / p['mass']

        # Record observables
        KE = 0.5 * sum(p['mass'] * np.dot(p['v'], p['v']) for p in particles)
        if potential_type == "harmonic":
            PE = 0.5 * k * np.dot(particles[0]['r'] - particles[1]['r'], particles[0]['r'] - particles[1]['r'])
        elif potential_type == "fene":
            r12 = particles[0]['r'] - particles[1]['r']
            r2_sq = np.dot(r12, r12)
            PE = -0.5 * epsilon * a**2 * np.log(1 - r2_sq / a**2 + 1e-12)
        else:
            r12 = particles[0]['r'] - particles[1]['r']
            r2_sq = np.dot(r12, r12)
            r_mag = np.sqrt(r2_sq)
            PE = a*np.exp(-k * r_mag) / r_mag if r_mag > 0 else 0.0

        E = KE + PE
        ptot = sum(p['mass'] * p['v'] for p in particles)
        J = sum(np.cross(p['r'], p['v']) for p in particles)

        r1_traj.append(particles[0]['r'].copy())
        r2_traj.append(particles[1]['r'].copy())
        P_traj.append(ptot)
        J_traj.append(J)
        E_traj.append(E)
        time.append((step + 1) * dt)

    return {
        "r1_traj": np.array(r1_traj),
        "r2_traj": np.array(r2_traj),
        "P_traj": np.array(P_traj),
        "J_traj": np.array(J_traj),
        "E_traj": np.array(E_traj),
        "time": np.array(time)
    }