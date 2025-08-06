import numpy as np
from utils.initial_conditions import initialize_particles, epsilon, a, k, m
from forces.potentials import kinetic_energy, potential_energy

# Omelyan 11-stage BA5B coefficients
theta = 0.08398315262876693
rho = 0.2539785108410595
lambda_ = 0.6822365335719091
mu = -0.03230286765269967

def harmonic_force(particles, return_energy=False):
    r1, r2 = particles[0]['r'], particles[1]['r']
    f = -1.0 * (r1 - r2)
    particles[0]['f'] = f
    particles[1]['f'] = -f
    
    if return_energy:
        return 0.5*np.sum((r1-r2)**2)


def fene_force(particles):
    r1, r2 = particles[0]['r'], particles[1]['r']
    r12 = r1 - r2
    r2_sq = np.dot(r12, r12)
    denom = a**2 - r2_sq
    if denom <= 1e-12:
        denom = 1e-12  # prevent divergence
    f = -epsilon * r12 / denom
    particles[0]['f'] = f
    particles[1]['f'] = -f

def yukawa_force(particles):
    r1, r2 = particles[0]['r'], particles[1]['r']
    r12 = r1 - r2
    r2_sq = np.dot(r12, r12)

    f = np.exp(-k * r2_sq) * (1 + k * r2_sq) / (r2_sq**3)
    f *= r12
    particles[0]['f'] = f
    particles[1]['f'] = -f


def omelyan_BA5B_step(particles, dt, force_fn, box=None):
    sub = [
        theta * dt,
        rho * dt,
        lambda_ * dt,
        mu * dt,
        (1 - 2 * (lambda_ + theta)) * dt / 2,
        (1 - 2 * (mu + rho)) * dt,
        (1 - 2 * (lambda_ + theta)) * dt / 2,
        mu * dt,
        lambda_ * dt,
        rho * dt,
        theta * dt
    ]

    for k in range(0, 10, 2):
        dt0, dt1 = sub[k], sub[k + 1]
        for p in particles:
            p['v'] += dt0 * p['f'] / p['mass']
            p['r'] += dt1 * p['v']
        if box:
            apply_periodic(particles, box)
        force_fn(particles)

    dt_last = sub[-1]
    for p in particles:
        p['v'] += dt_last * p['f'] / p['mass']


def simulate_11stage(dt, t_sim=10.0, potential_type="harmonic"):
    steps = int(t_sim / dt)
    particles = initialize_particles()

    # Choose force function
    force_fn = harmonic_force if potential_type == "harmonic" else  fene_force if potential_type == "fene" else yukawa_force

    # Initialize force
    force_fn(particles)

    r1_traj, r2_traj = [particles[0]['r'].copy()], [particles[1]['r'].copy()]
    P_traj, J_traj, E_traj = [], [], []
    time = [0.0]

    for step in range(steps):
        KE = kinetic_energy(particles)
        PE = potential_energy(particles)
        E_traj.append(KE + PE)

        ptot = particles[0]['mass'] * particles[0]['v'] + particles[1]['mass'] * particles[1]['v']
        J = np.cross(particles[0]['r'], particles[0]['v']) + np.cross(particles[1]['r'], particles[1]['v'])

        P_traj.append(ptot)
        J_traj.append(J)

        omelyan_BA5B_step(particles, dt, force_fn)

        r1_traj.append(particles[0]['r'].copy())
        r2_traj.append(particles[1]['r'].copy())
        time.append((step + 1) * dt)

    return {
        "r1_traj": np.array(r1_traj),
        "r2_traj": np.array(r2_traj),
        "P_traj": np.array(P_traj),
        "J_traj": np.array(J_traj),
        "E_traj": np.array(E_traj),
        "time": np.array(time)
    }
