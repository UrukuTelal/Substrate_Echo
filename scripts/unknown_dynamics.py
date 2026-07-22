#!/usr/bin/env python3
"""
Unknown Dynamics Recovery.

Generate data from an unseen system. The learner has NO knowledge of:
- how many attractors exist
- where they are
- what the potential looks like

It only sees trajectories. From those, it must:
1. Learn V(x)
2. Discover attractors (fixed points)
3. Classify stability (eigenvalues)
4. Map basin boundaries
5. Predict transition probabilities between basins
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.learned_geometry import LocalLinearField


# ──────────────────────────────────────────────────────────────
# Unknown dynamics: asymmetric 4-basin system with different depths
# ──────────────────────────────────────────────────────────────

class UnknownDynamics:
    """4 basins with different shapes, depths, and asymmetric coupling.
    
    The learner does NOT know:
    - number of basins (4)
    - their locations
    - their depths (varying)
    - the asymmetric coupling between basins 2 and 3
    """
    
    def __init__(self):
        # Asymmetric: different depths and widths
        self.wells = [
            {'center': np.array([-2.0, -1.5]), 'depth': 1.0, 'sigma': 0.6},
            {'center': np.array([ 2.0, -1.0]), 'depth': 0.8, 'sigma': 0.7},
            {'center': np.array([ 0.5,  2.0]), 'depth': 1.2, 'sigma': 0.5},
            {'center': np.array([-1.0,  1.5]), 'depth': 0.4, 'sigma': 0.9},
        ]
        # Asymmetric coupling: basin 2 pulls basin 3 slightly
        self.coupling = np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0.3],
                                   [0, 0, 0.1, 0]])
        self.dt = 0.02
    
    def potential(self, x, y):
        V = 0.0
        for w in self.wells:
            dx, dy = x - w['center'][0], y - w['center'][1]
            r2 = dx**2 + dy**2
            V -= w['depth'] * np.exp(-r2 / (2 * w['sigma']**2))
        return V
    
    def velocity(self, x, y, noise=0.0):
        gx, gy = 0.0, 0.0
        for w in self.wells:
            dx, dy = x - w['center'][0], y - w['center'][1]
            r2 = dx**2 + dy**2
            coeff = w['depth'] / (w['sigma']**2) * np.exp(-r2 / (2 * w['sigma']**2))
            gx += coeff * dx
            gy += coeff * dy
        
        # Asymmetric coupling: slight drift in x based on y proximity to basin 2
        gx += 0.2 * self.coupling[2, 3] * np.exp(-((x-0.5)**2 + (y-2.0)**2) / 2)
        
        vx = -gx + np.random.randn() * np.sqrt(noise) * 0.1
        vy = -gy + np.random.randn() * np.sqrt(noise) * 0.1
        return vx, vy
    
    def true_basin_of(self, x, y):
        """Ground truth: which basin is (x,y) in?"""
        dists = []
        for w in self.wells:
            d = np.sqrt(np.sum((np.array([x, y]) - w['center'])**2)) / w['sigma']
            dists.append(d)
        return int(np.argmin(dists))
    
    def generate_trajectory(self, x0, y0, n_steps, noise=0.01):
        traj = [(x0, y0)]
        velocities = []
        x, y = x0, y0
        for _ in range(n_steps):
            vx, vy = self.velocity(x, y, noise)
            x_new = np.clip(x + self.dt * vx, -4, 4)
            y_new = np.clip(y + self.dt * vy, -4, 4)
            traj.append((x_new, y_new))
            velocities.append((vx, vy))
            x, y = x_new, y_new
        return np.array(traj), np.array(velocities)


def discover_attractors_by_clustering(model, n_samples=200, n_steps=300, dt=0.02):
    """Discover attractors by integrating from random starts and clustering endpoints.
    
    This is more robust than fixed-point search because it doesn't depend on
    the model being locally accurate — only that trajectories converge.
    """
    rng = np.random.RandomState(42)
    endpoints = []
    
    # Random starts
    for _ in range(n_samples):
        x0, y0 = rng.uniform(-3, 3), rng.uniform(-3, 3)
        x, y = x0, y0
        for _ in range(n_steps):
            vx, vy = model.predict_velocity(x, y)
            x = np.clip(x + dt * vx, -4, 4)
            y = np.clip(y + dt * vy, -4, 4)
        endpoints.append([x, y])
    
    # Also sample from a grid to ensure coverage
    for gx in np.linspace(-2.5, 2.5, 10):
        for gy in np.linspace(-2.5, 2.5, 10):
            x, y = gx, gy
            for _ in range(n_steps):
                vx, vy = model.predict_velocity(x, y)
                x = np.clip(x + dt * vx, -4, 4)
                y = np.clip(y + dt * vy, -4, 4)
            endpoints.append([x, y])
    
    endpoints = np.array(endpoints)
    
    # Simple clustering: find dense regions
    # Use a grid-based approach
    grid_size = 50
    xs = np.linspace(-3.5, 3.5, grid_size)
    ys = np.linspace(-3.5, 3.5, grid_size)
    density = np.zeros((grid_size, grid_size))
    
    for ep in endpoints:
        ix = int((ep[0] - xs[0]) / (xs[1] - xs[0]))
        iy = int((ep[1] - ys[0]) / (ys[1] - ys[0]))
        if 0 <= ix < grid_size and 0 <= iy < grid_size:
            density[ix, iy] += 1
    
    # Smooth density
    from scipy.ndimage import gaussian_filter
    density_smooth = gaussian_filter(density, sigma=1)
    
    # Find peaks (local maxima)
    attractors = []
    min_density = max(3, np.percentile(density_smooth[density_smooth > 0], 50))
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            if density_smooth[i, j] >= min_density:
                is_peak = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if density_smooth[i+di, j+dj] > density_smooth[i, j]:
                            is_peak = False
                            break
                    if not is_peak:
                        break
                if is_peak:
                    attractors.append((xs[i], ys[j]))
    
    # Deduplicate
    unique = []
    for a in attractors:
        if not any(np.sqrt((a[0]-u[0])**2 + (a[1]-u[1])**2) < 0.5 for u in unique):
            unique.append(a)
    
    return unique, endpoints


def classify_by_integration(model, fx, fy, n_directions=8, perturb_dist=0.5, n_steps=100, dt=0.02):
    """Classify a fixed point by integrating outward.
    
    If ALL perturbations return -> attractor
    If NONE return -> repellor
    If SOME return -> saddle
    """
    returns = 0
    for angle in np.linspace(0, 2*np.pi, n_directions, endpoint=False):
        x = fx + perturb_dist * np.cos(angle)
        y = fy + perturb_dist * np.sin(angle)
        x, y = np.clip(x, -4, 4), np.clip(y, -4, 4)
        
        for _ in range(n_steps):
            vx, vy = model.predict_velocity(x, y)
            x = np.clip(x + dt * vx, -4, 4)
            y = np.clip(y + dt * vy, -4, 4)
        
        dist_to_fp = np.sqrt((x - fx)**2 + (y - fy)**2)
        if dist_to_fp < perturb_dist * 0.5:
            returns += 1
    
    if returns == n_directions:
        return "ATTRACTOR"
    elif returns == 0:
        return "REPELLOR"
    else:
        return "SADDLE"


def run_unknown_dynamics():
    print("=" * 60)
    print("UNKNOWN DYNAMICS RECOVERY")
    print("=" * 60)
    print("Learner has NO knowledge of basin count, locations, or depths.")
    
    world = UnknownDynamics()
    
    # ── Phase 1: Generate training data ──
    print(f"\n--- Phase 1: Generate Trajectories ---")
    rng = np.random.RandomState(42)
    all_states, all_velocities = [], []
    
    # Random starts
    for i in range(40):
        x0, y0 = rng.uniform(-3, 3), rng.uniform(-3, 3)
        traj, vels = world.generate_trajectory(x0, y0, 300, noise=0.005)
        for t in range(3, len(vels)):
            all_states.append(traj[t])
            all_velocities.append(vels[t])
    
    # Uniform grid for coverage
    for gx in np.linspace(-3, 3, 25):
        for gy in np.linspace(-3, 3, 25):
            vx, vy = world.velocity(gx, gy, noise=0.001)
            all_states.append([gx, gy])
            all_velocities.append([vx, vy])
    
    all_states = np.array(all_states)
    all_velocities = np.array(all_velocities)
    print(f"  {len(all_states)} training samples")
    
    # ── Phase 2: Learn vector field (blind) ──
    print(f"\n--- Phase 2: Learn Vector Field ---")
    model = LocalLinearField(k=25, bandwidth=0.4)
    model.fit(all_states, all_velocities)
    print(f"  Model: k-NN local linear, k={model.k}, bw={model.bandwidth}")
    
    # ── Phase 3: Discover structure (blind) ──
    print(f"\n--- Phase 3: Discover Structure ---")
    
    # 3a. Discover attractors by clustering trajectory endpoints
    print(f"\n  3a. Attractor discovery (clustering):")
    attractors, endpoints = discover_attractors_by_clustering(model)
    print(f"  Discovered {len(attractors)} attractors from {len(endpoints)} trajectories")
    for i, (ax, ay) in enumerate(attractors):
        print(f"    Attractor {i}: ({ax:+.2f}, {ay:+.2f})")
    
    # 3b. Compare to ground truth
    print(f"\n  3b. Ground truth comparison:")
    true_centers = [w['center'] for w in world.wells]
    print(f"  True attractors: {[(c[0], c[1]) for c in true_centers]}")
    
    matched = 0
    for i, tc in enumerate(true_centers):
        if attractors:
            dists = [np.sqrt((tc[0]-a[0])**2 + (tc[1]-a[1])**2) for a in attractors]
            best = min(dists)
            best_idx = np.argmin(dists)
            status = "MATCHED" if best < 0.5 else "MISSED"
            if best < 0.5:
                matched += 1
            print(f"    True {i} ({tc[0]:+.1f}, {tc[1]:+.1f}) -> "
                  f"Learned {best_idx} (dist={best:.3f}) {status}")
    
    print(f"  Geometry recovered: {matched}/{len(true_centers)} basins")
    
    # 3c. Classify stability of discovered attractors
    print(f"\n  3c. Stability classification:")
    for i, (ax, ay) in enumerate(attractors):
        label = classify_by_integration(model, ax, ay, perturb_dist=0.3)
        eigvals = model.eigenvalues_at(ax, ay)
        print(f"    Attractor {i} ({ax:+.2f}, {ay:+.2f}): {label}, "
              f"eig={[f'{ev:.2f}' for ev in eigvals]}")
    
    # 3d. Basin assignment: which endpoint belongs to which attractor?
    print(f"\n  3d. Basin assignment:")
    if attractors:
        basin_counts = {i: 0 for i in range(len(attractors))}
        for ep in endpoints:
            dists = [np.sqrt((ep[0]-a[0])**2 + (ep[1]-a[1])**2) for a in attractors]
            best = np.argmin(dists)
            basin_counts[best] += 1
        
        for i, count in basin_counts.items():
            pct = count / len(endpoints) * 100
            print(f"    Basin {i}: {count} endpoints ({pct:.1f}%)")
    else:
        print(f"    No attractors discovered")
    
    # ── Phase 4: Prediction test ──
    print(f"\n--- Phase 4: Prediction ---")
    horizons = [1, 5, 10, 25, 50]
    results = {h: {'persistence': [], 'learned': []} for h in horizons}
    
    for i in range(20):
        x0, y0 = rng.uniform(-3, 3), rng.uniform(-3, 3)
        true_traj, _ = world.generate_trajectory(x0, y0, 60, noise=0.0)
        pred_traj = model.predict_trajectory(x0, y0, 60, dt=world.dt)
        
        for h in horizons:
            if h < len(true_traj) and h < len(pred_traj):
                pers_err = np.linalg.norm(true_traj[h] - true_traj[0])
                pred_err = np.linalg.norm(true_traj[h] - pred_traj[h])
                results[h]['persistence'].append(pers_err)
                results[h]['learned'].append(pred_err)
    
    print(f"  {'Horizon':>8s} {'Persistence':>12s} {'Learned V':>12s} {'Better?':>8s}")
    for h in horizons:
        p = np.mean(results[h]['persistence'])
        l = np.mean(results[h]['learned'])
        better = "YES" if l < p else "no"
        print(f"  t+{h:<5d} {p:>12.4f} {l:>12.4f} {better:>8s}")
    
    # ── Phase 5: Transition prediction ──
    print(f"\n--- Phase 5: Transition Prediction ---")
    
    if attractors:
        transitions = {i: {} for i in range(len(attractors))}
        
        for i, (ax, ay) in enumerate(attractors):
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x0 = np.clip(ax + 1.5 * np.cos(angle), -3.5, 3.5)
                y0 = np.clip(ay + 1.5 * np.sin(angle), -3.5, 3.5)
                
                x, y = x0, y0
                final_basin = -1
                for step in range(200):
                    vx, vy = model.predict_velocity(x, y)
                    x = np.clip(x + 0.02 * vx, -4, 4)
                    y = np.clip(y + 0.02 * vy, -4, 4)
                    
                    for k, (ax2, ay2) in enumerate(attractors):
                        if np.sqrt((x-ax2)**2 + (y-ay2)**2) < 0.3:
                            final_basin = k
                            break
                    if final_basin >= 0:
                        break
                
                if final_basin >= 0:
                    transitions[i][final_basin] = transitions[i].get(final_basin, 0) + 1
        
        print(f"  Transition matrix:")
        print(f"  {'From/To':>10s}", end="")
        for j in range(len(attractors)):
            print(f"  {'A'+str(j):>6s}", end="")
        print()
        for i in sorted(transitions.keys()):
            print(f"  {'A'+str(i):>10s}", end="")
            total = sum(transitions[i].values())
            for j in range(len(attractors)):
                count = transitions[i].get(j, 0)
                pct = count / max(total, 1) * 100
                print(f"  {pct:>5.0f}%", end="")
            print()
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  True system: {len(world.wells)} basins")
    print(f"  Discovered: {len(attractors)} attractors")
    
    matched = 0
    for i, tc in enumerate(true_centers):
        if attractors:
            dists = [np.sqrt((tc[0]-a[0])**2 + (tc[1]-a[1])**2) for a in attractors]
            if min(dists) < 0.5:
                matched += 1
    print(f"  Geometry recovered: {matched}/{len(world.wells)} basins")
    
    return {
        'n_true_basins': len(world.wells),
        'n_discovered_attractors': len(attractors),
        'attractors': [(a[0], a[1]) for a in attractors],
        'true_basins': [(w['center'][0], w['center'][1]) for w in world.wells],
    }


if __name__ == '__main__':
    run_unknown_dynamics()
