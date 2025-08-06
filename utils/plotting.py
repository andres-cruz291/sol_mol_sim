import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(res, title="trajectory"):
    
    r1_traj = res['r1_traj']
    r2_traj = res['r2_traj']
    plt.figure()
    plt.plot(r1_traj[:, 0], r1_traj[:, 1], linewidth=2.5, label='$r1(t)$')
    plt.plot(r2_traj[:, 0], r2_traj[:, 1], '--', label='$r2(t)$')
    plt.xlabel("$x(t)$ [a] ")
    plt.ylabel("$y(t)$ [a]")
    plt.title(title)
    plt.legend()
    plt.show()    
    
def plot_deviations(res):
    
    E0 = res["E_traj"][0]
    P0 = res["P_traj"][0]
    J0 = res["J_traj"][0]

    dP = np.linalg.norm(res["P_traj"] - P0, axis=1)
    dJ_abs = np.linalg.norm(res["J_traj"] - J0, axis=1)
    dE_abs = np.abs(res["E_traj"] - E0)

    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(res["time"][:], dP)
    axs[0].set_ylabel('$|P(t)-P(0)|$ $[m \epsilon]^{-1}$')
    axs[0].set_title("Momentum deviation")
    axs[0].set_ylim(-1e-25, 1e-25)  


    axs[1].plot(res["time"][:], dJ_abs)
    axs[1].set_ylabel('$|J(t)-J(0)|$ $[m \omega_0 a^2]$')
    axs[1].set_title("Angular momentum deviation")

    axs[2].plot(res["time"][:], dE_abs)
    axs[2].set_ylabel('$|E(t)-E(0)|$ $[\epsilon]$')
    axs[2].set_xlabel("Time $[\omega_0]^{-1}$")
    axs[2].set_title("Energy deviation")

    plt.tight_layout()
    plt.show()



def plot_angular_direction_deviation(res, title="Angular momentum direction deviation"):
    J_traj = res["J_traj"]
    time=res["time"]
    J0 = J_traj[0]
    dJ_dir = 1 - np.einsum('ij,j->i', J_traj, J0) / (np.linalg.norm(J_traj, axis=1) * np.linalg.norm(J0))

    plt.figure()
    plt.plot(time[:], dJ_dir[:])
    plt.ylabel('$1 - (J(t)·J(0)) / (|J(t)||J(0)|)$')
    plt.xlabel('Time $[\omega_0]^{-1}$')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_energy_deviation_comparison(res_dt, res_4dt):
    
    
    dE_dt = np.abs(res_dt["E_traj"] - res_dt["E_traj"][0]) / np.abs(res_dt["E_traj"][0])
    dE_4dt = np.abs(res_4dt["E_traj"] - res_4dt["E_traj"][0]) / np.abs(res_4dt["E_traj"][0])

    plt.figure()
    plt.plot(res_dt["time"][:], dE_dt[:], label=r"$\Delta t$")
    plt.plot(res_4dt["time"][:], dE_4dt[:], '--', label=r"$4\Delta t$")
    plt.xlabel("Time $[\omega_0^{-1}]$")
    plt.ylabel("$|\Delta E(t)/E(0)|$")
    #plt.title(f"Energy Deviation Comparison ({label})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()



def plot_energy_deviation(time, energy, label=None, ref=None, dt=None, n=None):
    
    delta_E = np.abs(energy - energy[0])
    if ref is not None:
        scaled = delta_E / dt**n
    else:
        scaled = delta_E
    plt.plot(time, scaled, label=label)

    
def plot_scaled_energy(dts, energy_data, orders, title="Energy scaling"):
    plt.figure(figsize=(8, 4))
    for n in orders:
        plt.figure()
        for dt, (time, energy) in energy_data.items():
            delta_E = energy - energy[0]
            scaled = delta_E / dt**n
            plt.plot(time, scaled, label=f"$\Delta t = {dt:.1}$")
        plt.title(f"{title} — n = {n}")
        plt.xlabel("Time $[\omega_0]^{-1}$")
        plt.ylabel(f"$\Delta E(t) / (\Delta t)^{{n}}$")
        plt.legend()
        plt.tight_layout()
        plt.show()
