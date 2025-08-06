import numpy as np
import matplotlib.pyplot as plt
from integrators.velocity_verlet import velocity_verlet_simulation
from tqdm import trange


def analyze_temperature(N_values, k=1.0, m=1.0, epsilon=1.0, dt=0.01, t_sim=300):

    omega_0_inv = np.sqrt(m / k)
    dt = dt * omega_0_inv
    t_sim = t_sim * omega_0_inv
    kB = 1.0  # Assume kB = 1

    steps = int(t_sim / dt)
    time = np.arange(steps) * dt

    plt.figure(figsize=(10, 6))
    for i, N in enumerate(N_values, 1):
        res = velocity_verlet_simulation(
            N=N,
            dt=dt,
            t_sim=t_sim,
            k=k,
            m=m,
            beta=1.0 / epsilon,
            potential_type="harmonic",
            return_energy=True,
            track_observables=False,
        )
        kin_energy = res["kin_energy"]
        avg_kinetic = np.mean(kin_energy)
        T_numerical = avg_kinetic / (N * kB)
        T_predicted = epsilon / (4 * N * kB)
        rel_error = abs(T_numerical - T_predicted) / T_predicted

        print(f"N={N:3d} | T_pred = {T_predicted:.4f}, T_num = {T_numerical:.4f}, "
              f"rel.error = {rel_error:.4f}")

        plt.plot(time, kin_energy, label=f"N={N}, T_pred={T_predicted:.2f}")

    plt.xlabel(r"Time $[\sqrt{m/k}]$", fontsize=13)
    plt.ylabel("Kinetic Energy [$\\epsilon$]", fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_end_to_end(N_values, repeats=3, k=1.0, m=1.0, epsilon=1.0, dt=0.01, t_sim=300):
    omega_0_inv = np.sqrt(m / k)
    dt = dt * omega_0_inv
    t_sim = t_sim * omega_0_inv
    time = np.arange(int(t_sim / dt)) * dt
    steps = len(time)

    # Lists to store results
    N_list = []
    Re_means = []
    Re_stds = []

    for N in N_values:
        plt.figure(figsize=(10, 4))
        print(f"--- N={N} ---")
        avg_Re_list = []

        for rep in range(repeats):
            res = velocity_verlet_simulation(
                N=N,
                dt=dt,
                t_sim=t_sim,
                k=k,
                m=m,
                beta=1.0 / epsilon,
                config="linear",
                random_momenta=True,
                return_energy=False,
                track_observables=False,
                seed=rep
            )
            r_traj = res["r_traj"]
            R_e = np.linalg.norm(r_traj[:, -1] - r_traj[:, 0], axis=1)

            if N in [2, 3]:
                plot_steps = steps // 2
                plt.plot(time[:plot_steps], R_e[:plot_steps], label=f"Run {rep+1}")
            else:
                plt.plot(time, R_e, label=f"Run {rep+1}")

            avg_Re_list.append(np.mean(R_e))

        mean_Re = np.mean(avg_Re_list)
        std_Re = np.std(avg_Re_list)
        print(f"Average ⟨Re⟩ = {mean_Re:.3f} ± {std_Re:.3f}")

        # Store for return
        N_list.append(N)
        Re_means.append(mean_Re)
        Re_stds.append(std_Re)

        plt.xlabel(r"Time $[\sqrt{m/k}]$", fontsize=13)
        plt.ylabel(r"$R_e(t)$ [$\ell$]", fontsize=13)
        plt.title(f"N={N}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Return results
    return np.array(N_list), np.array(Re_means), np.array(Re_stds)


                

def recurrence_distance_with_integrator(N, dt, t_sim, t_ref, k=1.0, m=1.0): # FIX ME
    # Run simulation 
    r_traj, v_traj = velocity_verlet_chain(N=N, dt=dt, t_sim=t_sim, k=k, m=m)

    steps_ref = int(t_ref / dt)
    r_ref = r_traj[steps_ref]
    v_ref = v_traj[steps_ref]

    # Compute phase-space distance from t_ref onward
    r_diff_sq = (r_traj[steps_ref:] - r_ref)**2
    v_diff_sq = (v_traj[steps_ref:] - v_ref)**2

    dists = np.sqrt(
        np.sum(k * r_diff_sq + v_diff_sq / m, axis=(1, 2)) / (6 * N)
    )
    time = np.arange(steps_ref, r_traj.shape[0]) * dt
    return time, dists


def analyze_recurrence_integrator(N_values, k=1.0, m=1.0): # FIX ME
    omega_0_inv = np.sqrt(m / k)
    dt = 0.01 * omega_0_inv
    t_ref = 10 * omega_0_inv
    t_sim = 20000 * omega_0_inv

    for N in N_values:
        print(f"\n--- Recurrence analysis for N = {N} ---")
        time, dist = recurrence_distance_with_integrator(N, dt, t_sim, t_ref, k=k, m=m)

        min_dist = np.min(dist)
        min_time = time[np.argmin(dist)]
        print(f"Minimum d(t) = {min_dist:.4e} at t = {min_time:.5f} √(m/k)")

        plt.figure(figsize=(10, 4))
        plt.plot(time[:len(time)//4], dist[:len(time)//4], label=f"N = {N}")
        plt.xlabel(r"Time $[\sqrt{m/k}]$", fontsize=13)
        plt.ylabel(r"$d(t) [\ell]$", fontsize=13)
        plt.title(f"N = {N}")
        plt.legend()
        plt.show()
        
def compute_mean_and_se(data):
    data = np.array(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(len(data) - 1)
    return mean, se