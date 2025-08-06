import numpy as np

a = 1.0
k = 1.0
m = 1.0
epsilon = k * a**2
A_yukawa = 1.0


def initialize_particles():
    r1 = np.array([a/3, 0.0, 0.0])
    r2 = -r1
    v1 = np.array([-6/7, 3/7, -2/7])* np.sqrt(m * epsilon)
    v2 = -v1
    mass = m

    return [
        {'r': r1.copy(), 'v': v1.copy(), 'f': np.zeros(3), 'mass': mass},
        {'r': r2.copy(), 'v': v2.copy(), 'f': np.zeros(3), 'mass': mass}
    ]


def initialize_two_particles(a=1.0, m=1.0, epsilon=epsilon):
    r1 = np.array([a / 3, 0.0, 0.0])
    r2 = -r1
    v1 = np.array([-6 / 7, 3 / 7, -2 / 7]) * np.sqrt(m * epsilon)
    v2 = -v1
    return np.stack([r1, r2]), np.stack([v1, v2])

    
    
def set_maxwell_boltzmann(particles, temperature=1.0):
    """
    Assign velocities from a Maxwell-Boltzmann distribution at given temperature.
    Ensures net momentum is zero.
    """
    N = len(particles)
    dim = 3
    m = particles[0]['mass']
    stddev = np.sqrt(temperature / m)
    velocities = np.random.normal(loc=0.0, scale=stddev, size=(N, dim))

    # Remove center-of-mass velocity to ensure net momentum = 0
    v_cm = np.mean(velocities, axis=0)
    velocities -= v_cm

    for i, p in enumerate(particles):
        p['v'] = velocities[i]
