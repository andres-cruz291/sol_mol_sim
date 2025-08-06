import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from utils.initial_conditions import A_yukawa
from forces.potentials import yukawa_force, yukawa_force_loop
#from integrators.velocity_verlet import velocity_verlet_simulation_pbc
from utils.plotting import plot_energy_deviation, plot_scaled_energy, plot_energy_deviation_comparison


def velocity_verlet_simulation_pbc(N=11, dt=0.01, t_sim=10.0, k=1.0, m=1.0, beta=1.0, 
                                    config="linear", random_momenta=False, return_energy=False, 
                                    seed=None, L=None, apply_pbc=True, close_chain=False, r=None, v=None, i_loop=False):

    if seed is not None:
        np.random.seed(seed)

    steps = int(t_sim / dt)
    a = 1.0
    epsilon = 1 / beta
    if L is None:
        L = N * a

    """r = np.zeros((N, 3))
    r = np.random.uniform(0, L, size=(N, 3))
    v = np.zeros((N, 3))
    v[:] = np.random.normal(0, np.sqrt(epsilon / m), size=(N, 3))"""

    def apply_pbc_fn(r):
        return (r + L / 2) % L - L / 2 if apply_pbc else r

    def compute_forces(r):
        if i_loop:
            return yukawa_force_loop(r)
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
            #U += 0.5 * k * (dist)**2
            U += A_yukawa * np.exp(-k * dist) / dist
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
    r = apply_pbc_fn(r)
    f = compute_forces(r)
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


omega_0_inv = 1.0
dts = [0.0005*(2**(i)) * omega_0_inv for i in range(0, 4)]
#dts = [0.001]
#print(dts)

orders = [1, 2, 3, 4]

#for potential in ["harmonic", "fene"]:
energy_data = {}

N=128
epsilon = 1.0
m = 1.0
L=6.8399037867

r = np.zeros((N, 3))
r = np.random.uniform(0, L, size=(N, 3))
v = np.zeros((N, 3))
v[:] = np.random.normal(0, np.sqrt(epsilon / m), size=(N, 3))

dts_filtered = dts#[1:] #if potential == 'harmonic' else dts[1:]  # removing larger timestep in the case of anharmonic potential
for dt in dts_filtered:
    #res = simulate_11stage(dt=dt, potential_type=potential)
    res = velocity_verlet_simulation_pbc(N, dt, dts[0]*8000, 1.,m,1.,"linear", True, True, None, L, True, False, r.copy(), v.copy())
    energy_data[dt] = (res["time"], res["E_traj"])
    #print(res["time"])
    #print(res["energy_pot"])
    #quit()
    #plt.plot(res["time"], (res["energy_pot"]-res["energy_pot"][0])*dt**(-2), label=f"dt={dt:.4f}")
    plt.plot(res["time"], (res["energy_traj"]-res["energy_traj"][0])/dt**2, label=f"dt={dt:.4f}")
    #plot_energy_deviation(res["time"], res["E_traj"], label="Yukawa")
#plot_energy_deviation_comparison(energy_data[dts[0]], energy_data[dts[1]])
plt.xlabel("Time")
plt.ylabel("Potential Energy")
plt.title("Potential Energy Evolution (Yukawa)")
plt.legend()
plt.tight_layout()
plt.show()

#plot_scaled_energy(dts, energy_data, orders, title=f"Velocity verlet: Yukawa potential")
