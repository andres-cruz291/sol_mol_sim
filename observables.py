import numpy as np

def compute_Re(r_traj):
    """
    Compute Re(t) from a trajectory array of shape (steps, N, 3)
    """
    return np.linalg.norm(r_traj[:, -1] - r_traj[:, 0], axis=1)

def compute_Rg2(r_traj):
    """
    Compute Rg^2(t) from a trajectory array of shape (steps, N, 3)
    """
    r_cm = np.mean(r_traj, axis=1)  # shape (steps, 3)
    diffs = r_traj - r_cm[:, np.newaxis, :]  # (steps, N, 3)
    Rg2 = np.mean(np.sum(diffs**2, axis=2), axis=1)  # mean over N
    return Rg2

def compute_U(r_traj, k=1.0):
    """
    Compute potential energy per particle from harmonic chain trajectory.
    """
    dr = r_traj[:, 1:] - r_traj[:, :-1]           # shape (steps, N-1, 3)
    bond_energy = 0.5 * k * np.sum(dr**2, axis=2) # shape (steps, N-1)
    U_ = np.mean(bond_energy, axis=1)        # average over N-1 bonds
    return U_

def time_average_observables(r_traj, k=1.0, t_cutoff=1000, dt=0.01):

    start_idx = int(t_cutoff / dt)
    Re2 = compute_Re(r_traj[start_idx:])**2
    Rg2 = compute_Rg2(r_traj[start_idx:])
    U   = compute_U(r_traj[start_idx:], k=k)

    return np.mean(Rg2), np.mean(Re2), np.mean(U)
