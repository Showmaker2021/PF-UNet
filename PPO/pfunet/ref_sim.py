import numpy as np
from collections import namedtuple

RefObs = namedtuple('RefObs', ['xy', 'feats', 'raw', 'mask_pair', 'const', 'alive_jobs', 'S',
                                'fut_xy', 'fut_feats', 'fut_mask'])

class RefSim:
    """
    Pure-Python Reference Simulator for Flexible Job Shop Problem (FJSP).
    Tracks:
      - R[j]: ready time of job j
      - A[k]: available time of machine k
      - ptr[j]: current operation index for job j
    """
    def __init__(self, dur: np.ndarray, jobs: list):
        """
        Args:
            dur: np.ndarray of shape (total_ops, num_machines). Incompatible entries are inf.
            jobs: list of lists, jobs[j] is the sequence of global operation indices for job j.
        """
        self.dur = dur.astype(float)
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = dur.shape[1]
        self.R = np.zeros(self.num_jobs, dtype=float)
        self.A = np.zeros(self.num_machines, dtype=float)
        self.ptr = [0] * self.num_jobs

    def heads(self):
        """Returns list of (job_idx, global_op_idx) for active jobs."""
        return [(j, self.jobs[j][self.ptr[j]]) for j in range(self.num_jobs) if self.ptr[j] < len(self.jobs[j])]

    def is_done(self):
        return all(self.ptr[j] >= len(self.jobs[j]) for j in range(self.num_jobs))

    def step(self, j: int, k: int) -> float:
        """
        Schedule head operation of job j on machine k.
        """
        assert self.ptr[j] < len(self.jobs[j]), f"Job {j} has already completed all operations."
        op = self.jobs[j][self.ptr[j]]
        dur_val = self.dur[op, k]
        assert not np.isinf(dur_val) and dur_val > 0, f"Operation {op} cannot run on machine {k}."
        
        start = max(self.R[j], self.A[k])
        finish = start + dur_val
        self.R[j] = finish
        self.A[k] = finish
        self.ptr[j] += 1
        return finish

    def makespan(self) -> float:
        return float(self.R.max())


def to_ref(env, batch_idx: int = 0):
    """
    Extracts dur matrix and job lists from FJSPEnv for a specific batch element.
    """
    raw_dur = env.problem.duration[batch_idx].astype(float)
    # 0 in ReSched duration indicates incompatible machine
    dur = np.where(raw_dur == 0, np.inf, raw_dur)
    
    job_ids = env.problem.job_idx[batch_idx]
    unique_jobs = np.unique(job_ids)
    jobs = []
    for j in unique_jobs:
        op_indices = np.where(job_ids == j)[0]
        jobs.append(list(op_indices))
    return dur, jobs


def ref_obs(sim: RefSim, eps: float = 1e-6, job_order: list = None) -> RefObs:
    """
    Computes reference observation matching Section 5.3 of the guide.
    If job_order is provided, use that specific ordering of active jobs;
    otherwise use natural sorted order of active jobs.
    """
    M = sim.num_machines
    alive = [j for j in range(sim.num_jobs) if sim.ptr[j] < len(sim.jobs[j])]
    if job_order is not None:
        alive = [j for j in job_order if j in alive]
    
    J = len(alive)
    if J == 0:
        return None

    # 1. Remaining work W and remaining op count n
    W = np.zeros(J, dtype=float)
    n = np.zeros(J, dtype=float)
    R = np.zeros(J, dtype=float)
    p_min = np.zeros(J, dtype=float)
    p_hd = np.zeros((J, M), dtype=float)
    ok_hd = np.zeros((J, M), dtype=bool)
    nM_hd = np.zeros(J, dtype=float)

    for idx, j in enumerate(alive):
        curr_p = sim.ptr[j]
        rem_ops = sim.jobs[j][curr_p:]
        n[idx] = len(rem_ops)
        R[idx] = sim.R[j]
        
        # total remaining work = sum of min dur over remaining ops
        w_tot = 0.0
        for op in rem_ops:
            op_durs = sim.dur[op]
            valid = op_durs[op_durs < np.inf]
            w_tot += valid.min() if len(valid) > 0 else 0.0
        W[idx] = w_tot

        # head op
        hd_op = sim.jobs[j][curr_p]
        hd_durs = sim.dur[hd_op]
        valid_hd = hd_durs[hd_durs < np.inf]
        p_min[idx] = valid_hd.min() if len(valid_hd) > 0 else eps
        for k in range(M):
            if hd_durs[k] < np.inf and hd_durs[k] > 0:
                ok_hd[idx, k] = True
                p_hd[idx, k] = hd_durs[k]
        nM_hd[idx] = max(float(ok_hd[idx].sum()), 1.0)

    # 2. Normalization bounds (ignore disconnected/dead machines)
    alive_m = np.zeros(M, dtype=bool)
    for j in alive:
        for op in sim.jobs[j][sim.ptr[j]:]:
            op_valid = (sim.dur[op] < np.inf) & (sim.dur[op] > 0)
            alive_m |= op_valid
    if not alive_m.any():
        alive_m[:] = True
    Ma = max(float(alive_m.sum()), 1.0)

    A = sim.A.copy()  # [M]
    t0 = A[alive_m].min()
    Rt = np.maximum(R, t0)
    LBj = (Rt + W).max()
    LB = max(LBj, A[alive_m].max())
    S = max(LB - t0, eps)

    # 3. Features layout [M, J]
    P = p_hd.T  # [M, J]
    OK = ok_hd.T  # [M, J]
    ST = np.maximum(R[None, :], A[:, None])  # [M, J]
    
    x = np.clip((ST - t0) / S, 0.0, 1.0)
    y = np.clip(np.broadcast_to(W[None, :] / S, (M, J)), 0.0, 1.0)
    wait = np.clip((ST - A[:, None]) / S, 0.0, 1.0)
    job_wait = np.clip((ST - R[None, :]) / S, 0.0, 1.0)
    max_n = max(n.max(), 1.0)
    prog = np.broadcast_to((n / max_n)[None, :], (M, J))
    flex = np.clip(np.broadcast_to((nM_hd / Ma)[None, :], (M, J)), 0.0, 1.0)
    p_norm = np.clip(P / S, 0.0, 1.0)
    p_fast = np.clip(np.maximum(p_min[None, :], eps) / np.maximum(P, eps), 0.0, 1.0) * OK.astype(float)

    # 3b. Lookahead & critical-path features [M, J]
    W_tail = np.maximum(W - p_min, 0.0)
    lb_push = np.clip(((ST + P + W_tail[None, :]) - LB) / S, -1.0, 1.0)
    next_p_norm = np.zeros((M, J), dtype=float)
    next_aff = np.zeros((M, J), dtype=float)
    tail_aff = np.zeros((M, J), dtype=float)

    for idx, j in enumerate(alive):
        curr_p = sim.ptr[j]
        rem_ops = sim.jobs[j][curr_p:]
        if len(rem_ops) > 1:
            nxt_op = rem_ops[1]
            nxt_durs = sim.dur[nxt_op]
            nxt_valid = (nxt_durs < np.inf) & (nxt_durs > 0)
            nxt_min = nxt_durs[nxt_valid].min() if nxt_valid.any() else 0.0
            next_p_norm[:, idx] = np.clip(nxt_min / S, 0.0, 1.0)
            for k in range(M):
                if nxt_valid[k]:
                    next_aff[k, idx] = np.clip(max(nxt_min, eps) / max(nxt_durs[k], eps), 0.0, 1.0)
        if W_tail[idx] > eps:
            w_m = np.zeros(M, dtype=float)
            for op in rem_ops[1:]:
                op_durs = sim.dur[op]
                valid_mask = (op_durs < np.inf) & (op_durs > 0)
                if valid_mask.any():
                    op_min = op_durs[valid_mask].min()
                    nM_op = float(valid_mask.sum())
                    w_m[valid_mask] += op_min / nM_op
            tail_aff[:, idx] = np.clip(w_m / (W_tail[idx] + eps), 0.0, 1.0)

    xy = np.stack([x, y], axis=-1)  # [M, J, 2]
    feats = np.stack([np.ones_like(x), p_norm, flex, wait, prog], axis=-1)  # [M, J, 5]
    raw = np.stack([
        p_norm, p_fast, wait, job_wait, x, y, flex, prog,
        lb_push, next_p_norm, next_aff, tail_aff,
    ], axis=-1)  # [M, J, 12]

    # 4. Machine constant channels: c6, c7
    c6 = np.maximum((A - t0) / S, 0.0) * alive_m.astype(float)  # [M]
    
    # c7: future load for operations strictly after current head + future field
    L = np.zeros(M, dtype=float)
    fut_entries = []
    for j in alive:
        curr_p = sim.ptr[j]
        rem_ops = sim.jobs[j][curr_p:]
        r_acc = sim.R[j]
        # precompute min durations along remaining ops
        op_mins = []
        for op in rem_ops:
            od = sim.dur[op]
            vm = (od < np.inf) & (od > 0)
            op_mins.append(od[vm].min() if vm.any() else 0.0)
        total_w = sum(op_mins)
        w_left = total_w
        for pos_idx, op in enumerate(rem_ops):
            op_min = op_mins[pos_idx]
            if pos_idx > 0:
                op_durs = sim.dur[op]
                valid_mask = (op_durs < np.inf) & (op_durs > 0)
                if valid_mask.any():
                    nM_op = float(valid_mask.sum())
                    for k in range(M):
                        if valid_mask[k]:
                            L[k] += op_durs[k] / nM_op
                    fut_entries.append((r_acc, w_left, valid_mask.copy(), nM_op))
            r_acc += op_min
            w_left -= op_min

    c7 = (Ma * L / (L.sum() + eps)) * alive_m.astype(float)
    const = np.stack([c6, c7], axis=-1)  # [M, 2]

    N_fut = max(len(fut_entries), 1)
    K_fut = len(fut_entries)
    fut_xy = np.zeros((M, K_fut, 2), dtype=float)
    fut_feats = np.zeros((M, K_fut, 2), dtype=float)
    fut_mask = np.zeros((M, K_fut), dtype=bool)
    for fi, (r_op, w_op, vmask, nM_op) in enumerate(fut_entries):
        st_op = np.maximum(r_op, A)
        fut_xy[:, fi, 0] = np.clip((st_op - t0) / S, 0.0, 1.0)
        fut_xy[:, fi, 1] = np.clip(w_op / S, 0.0, 1.0)
        fut_mask[:, fi] = vmask
        dem = (Ma / N_fut) / nM_op
        crit = np.clip((r_op + w_op - t0) / S, 0.0, 1.0)
        fut_feats[:, fi, 0] = dem
        fut_feats[:, fi, 1] = dem * crit

    return RefObs(xy, feats, raw, OK, const, alive, S, fut_xy, fut_feats, fut_mask)
