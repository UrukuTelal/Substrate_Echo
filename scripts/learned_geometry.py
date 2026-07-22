#!/usr/bin/env python3
"""
Learned Dynamics Geometry — 2D Prototype.

Stage 1: Generate trajectories from a 2D multi-basin GL system
Stage 2: Learn local linear vector field from (state, velocity) pairs
Stage 3: Predict forward — compare persistence, memory, learned V
Stage 4: Evaluate — prediction error, basin prediction, geometry recovery
Stage 5: Visualize — trajectories, vector arrows, attractor basins
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ──────────────────────────────────────────────────────────────
# Stage 1: Multi-basin GL system in 2D
# ──────────────────────────────────────────────────────────────

class MultiBasinDynamics:
    """2D multi-basin dynamics with Gaussian wells.

    Uses V(x) = -Σ_i w_i * exp(-|x-c_i|² / 2σ²)
    This gives bounded gradients and smooth basins.
    """

    def __init__(self, basins, gamma=0.5, dt=0.02):
        self.basins = basins  # [(cx, cy, sigma), ...]
        self.gamma = gamma
        self.dt = dt

    def potential(self, x, y):
        V = 0.0
        for cx, cy, sigma in self.basins:
            r2 = (x - cx)**2 + (y - cy)**2
            V -= np.exp(-r2 / (2 * sigma**2))
        return V

    def gradient(self, x, y):
        gx, gy = 0.0, 0.0
        for cx, cy, sigma in self.basins:
            dx, dy = x - cx, y - cy
            r2 = dx**2 + dy**2
            coeff = np.exp(-r2 / (2 * sigma**2)) / (sigma**2)
            gx += coeff * dx
            gy += coeff * dy
        return gx, gy

    def velocity(self, x, y, noise=0.0):
        gx, gy = self.gradient(x, y)
        vx = -gx + self.gamma * np.random.randn() * np.sqrt(noise)
        vy = -gy + self.gamma * np.random.randn() * np.sqrt(noise)
        return vx, vy

    def step(self, x, y, noise=0.0):
        vx, vy = self.velocity(x, y, noise)
        x_new = x + self.dt * vx
        y_new = y + self.dt * vy
        return x_new, y_new, vx, vy

    def basin_of(self, x, y):
        dists = [np.sqrt((x-cx)**2 + (y-cy)**2) / sigma for cx, cy, sigma in self.basins]
        return int(np.argmin(dists))

    def fixed_points(self):
        return [(cx, cy) for cx, cy, _ in self.basins]

    def generate_trajectory(self, x0, y0, n_steps, noise=0.0):
        traj = [(x0, y0)]
        velocities = []
        basin_ids = []
        energies = []

        x, y = x0, y0
        for _ in range(n_steps):
            x_new, y_new, vx, vy = self.step(x, y, noise)
            x_new = np.clip(x_new, -4, 4)
            y_new = np.clip(y_new, -4, 4)
            traj.append((x_new, y_new))
            velocities.append((vx, vy))
            basin_ids.append(self.basin_of(x, y))
            energies.append(self.potential(x, y))
            x, y = x_new, y_new

        return {
            'trajectory': np.array(traj),
            'velocities': np.array(velocities),
            'basin_ids': np.array(basin_ids),
            'energies': np.array(energies),
        }


# ──────────────────────────────────────────────────────────────
# Stage 2: Local Linear Vector Field
# ──────────────────────────────────────────────────────────────

class LocalLinearField:
    """Learned local linear vector field.

    For each query point:
    1. Find k nearest neighbors from training data
    2. Fit Δx = A*(x - x_center) + b
    3. Predict velocity

    This is a locally weighted regression.
    """

    def __init__(self, k=20, bandwidth=0.3):
        self.k = k
        self.bandwidth = bandwidth
        self.states = None
        self.velocities = None
        self.fitted = False

    def fit(self, states, velocities):
        """Store training data."""
        self.states = np.array(states)
        self.velocities = np.array(velocities)
        self.fitted = True

    def predict_velocity(self, x, y):
        """Predict velocity at (x, y) using local linear model."""
        if not self.fitted:
            return 0.0, 0.0

        point = np.array([x, y])

        # Find k nearest neighbors
        dists = np.linalg.norm(self.states - point, axis=1)
        k = min(self.k, len(self.states))
        nn_idx = np.argpartition(dists, k)[:k]
        nn_dists = dists[nn_idx]

        # Kernel weights (Gaussian)
        weights = np.exp(-nn_dists**2 / (2 * self.bandwidth**2))
        weights = weights / (np.sum(weights) + 1e-10)

        # Weighted linear fit: v = A*(x - x_center) + b
        nn_states = self.states[nn_idx]
        nn_vels = self.velocities[nn_idx]

        x_center = np.average(nn_states, axis=0, weights=weights)
        v_center = np.average(nn_vels, axis=0, weights=weights)

        # Weighted covariance
        dx = nn_states - x_center
        dv = nn_vels - v_center

        # A = weighted covariance(dv, dx) / weighted covariance(dx, dx)
        W = np.diag(weights)
        Cxx = dx.T @ W @ dx + 1e-6 * np.eye(2)
        Cvx = dv.T @ W @ dx
        A = Cvx @ np.linalg.inv(Cxx)

        # Predict
        pred = A @ (point - x_center) + v_center
        return float(pred[0]), float(pred[1])

    def predict_trajectory(self, x0, y0, n_steps, dt=0.01):
        """Integrate the learned field forward."""
        traj = [(x0, y0)]
        x, y = x0, y0
        for _ in range(n_steps):
            vx, vy = self.predict_velocity(x, y)
            x = x + dt * vx
            y = y + dt * vy
            x = np.clip(x, -3, 3)
            y = np.clip(y, -3, 3)
            traj.append((x, y))
        return np.array(traj)

    def jacobian(self, x, y, eps=0.05):
        """Compute local Jacobian numerically.
        
        Uses larger eps to smooth over k-NN boundary discontinuities.
        """
        vx0, vy0 = self.predict_velocity(x, y)
        dvx_dx = (self.predict_velocity(x+eps, y)[0] - self.predict_velocity(x-eps, y)[0]) / (2*eps)
        dvx_dy = (self.predict_velocity(x, y+eps)[0] - self.predict_velocity(x, y-eps)[0]) / (2*eps)
        dvy_dx = (self.predict_velocity(x+eps, y)[1] - self.predict_velocity(x-eps, y)[1]) / (2*eps)
        dvy_dy = (self.predict_velocity(x, y+eps)[1] - self.predict_velocity(x, y-eps)[1]) / (2*eps)
        J = np.array([[dvx_dx, dvx_dy],
                       [dvy_dx, dvy_dy]])
        return J

    def eigenvalues_at(self, x, y):
        """Eigenvalues of Jacobian at (x,y) — stability analysis."""
        J = self.jacobian(x, y)
        eigvals = np.linalg.eigvals(J)
        return eigvals

    def find_fixed_points(self, basin_centers=None, grid_range=(-2.5, 2.5), grid_res=20):
        """Find points where V(x,y) = 0.
        
        Strategy: start from basin centers + grid, refine with gradient descent.
        """
        initial_guesses = []
        
        # Start from basin centers (strong attractors)
        if basin_centers:
            for cx, cy, sigma in basin_centers:
                initial_guesses.append((cx, cy))
        
        # Also use grid
        xs = np.linspace(grid_range[0], grid_range[1], grid_res)
        ys = np.linspace(grid_range[0], grid_range[1], grid_res)
        for x in xs:
            for y in ys:
                vx, vy = self.predict_velocity(x, y)
                speed = np.sqrt(vx**2 + vy**2)
                if speed < 2.0:
                    initial_guesses.append((float(x), float(y)))
        
        fixed_points = []
        for x0, y0 in initial_guesses:
            xf, yf = float(x0), float(y0)
            for _ in range(50):
                vx, vy = self.predict_velocity(xf, yf)
                speed = np.sqrt(vx**2 + vy**2)
                if speed < 1e-5:
                    break
                lr = min(0.02, 0.5 / (speed + 1))
                xf -= lr * vx
                yf -= lr * vy
                xf = np.clip(xf, -3, 3)
                yf = np.clip(yf, -3, 3)
            
            vx, vy = self.predict_velocity(xf, yf)
            if np.sqrt(vx**2 + vy**2) < 0.1:
                fixed_points.append((xf, yf))
        
        # Remove duplicates
        unique = []
        for fp in fixed_points:
            is_dup = any(np.sqrt((fp[0]-u[0])**2 + (fp[1]-u[1])**2) < 0.4 for u in unique)
            if not is_dup:
                unique.append(fp)
        
        # Filter: only keep points that are true attractors (all eigenvalues negative real part)
        stable_fps = []
        for fx, fy in unique:
            eigvals = self.eigenvalues_at(fx, fy)
            real_parts = [float(np.real(ev)) for ev in eigvals]
            if all(r < 0.5 for r in real_parts):  # stable or weakly unstable (numerical noise)
                stable_fps.append((fx, fy))
        
        return stable_fps


# ──────────────────────────────────────────────────────────────
# Stage 3 & 4: Prediction comparison
# ──────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_experiment():
    print("=" * 60)
    print("LEARNED DYNAMICS GEOMETRY — 2D PROTOTYPE")
    print("=" * 60)

    # ── Stage 1: Define 2D multi-basin system ──
    basins = [
        (-1.5, -1.0, 0.8),   # Basin 0 (bottom-left)
        ( 1.5, -1.0, 0.8),   # Basin 1 (bottom-right)
        ( 0.0,  1.5, 0.8),   # Basin 2 (top-center)
    ]

    dynamics = MultiBasinDynamics(basins, gamma=0.5, dt=0.02)

    print(f"\nBasins:")
    for i, (cx, cy, sigma) in enumerate(basins):
        print(f"  {i}: center=({cx:.1f}, {cy:.1f}), sigma={sigma}")

    # Generate trajectories from multiple starting points
    rng = np.random.RandomState(42)
    all_states = []
    all_velocities = []
    all_basins = []

    # Training: trajectories from random starts + explicit saddle region samples
    print(f"\nGenerating training trajectories...")
    n_train = 30
    for i in range(n_train):
        x0 = rng.uniform(-2.5, 2.5)
        y0 = rng.uniform(-2.5, 2.5)
        result = dynamics.generate_trajectory(x0, y0, n_steps=500, noise=0.01)

        for t in range(5, len(result['velocities'])):
            all_states.append(result['trajectory'][t])
            all_velocities.append(result['velocities'][t])

    # Explicit saddle region sampling: uniform grid between basins
    # This ensures the model sees low-velocity regions
    grid_xs = np.linspace(-2.5, 2.5, 20)
    grid_ys = np.linspace(-2.5, 2.5, 20)
    for gx in grid_xs:
        for gy in grid_ys:
            vx, vy = dynamics.velocity(gx, gy, noise=0.002)
            all_states.append([gx, gy])
            all_velocities.append([vx, vy])

    # Also sample along paths between basin pairs
    for (ax, ay), (bx, by) in [
        ((-1.5, -1.0), (1.5, -1.0)),  # basin 0 -> 1
        ((-1.5, -1.0), (0.0, 1.5)),   # basin 0 -> 2
        ((1.5, -1.0), (0.0, 1.5)),    # basin 1 -> 2
    ]:
        for frac in np.linspace(0, 1, 30):
            x = ax + frac * (bx - ax)
            y = ay + frac * (by - ay)
            vx, vy = dynamics.velocity(x, y, noise=0.002)
            all_states.append([x, y])
            all_velocities.append([vx, vy])

    all_states = np.array(all_states)
    all_velocities = np.array(all_velocities)
    print(f"  {len(all_states)} training samples")

    # ── Stage 2: Learn vector field ──
    print(f"\nLearning vector field...")
    model = LocalLinearField(k=30, bandwidth=0.3)
    model.fit(all_states, all_velocities)
    print(f"  k={model.k}, bandwidth={model.bandwidth}")

    # ── Stage 3: Predict and compare ──
    horizons = [1, 5, 10, 20, 50]
    n_test = 20
    test_results = {h: {'persistence': [], 'learned_v': [], 'true': []} for h in horizons}

    # Test: trajectories from known starting points
    print(f"\nRunning {n_test} test trajectories...")
    test_trajectories = []
    for i in range(n_test):
        x0 = rng.uniform(-2.5, 2.5)
        y0 = rng.uniform(-2.5, 2.5)

        # True trajectory
        true_result = dynamics.generate_trajectory(x0, y0, n_steps=100, noise=0.0)
        true_traj = true_result['trajectory']

        # Predicted trajectory (learned V)
        pred_traj = model.predict_trajectory(x0, y0, n_steps=100, dt=dynamics.dt)

        test_trajectories.append({
            'start': (x0, y0),
            'true': true_traj,
            'predicted': pred_traj,
        })

        for h in horizons:
            if h < len(true_traj) and h < len(pred_traj):
                true_state = true_traj[h]
                persistence_state = true_traj[0]  # x(t+n) = x(t)
                pred_state = pred_traj[h]

                # Error metrics
                persistence_err = np.linalg.norm(true_state - persistence_state)
                learned_err = np.linalg.norm(true_state - pred_state)

                test_results[h]['persistence'].append(persistence_err)
                test_results[h]['learned_v'].append(learned_err)
                test_results[h]['true'].append(true_state)

    # ── Stage 4: Evaluation ──
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")

    print(f"\n{'Horizon':>8s} {'Persistence':>12s} {'Learned V':>12s} {'Improvement':>12s}")
    for h in horizons:
        pers = np.mean(test_results[h]['persistence'])
        learned = np.mean(test_results[h]['learned_v'])
        improvement = (pers - learned) / max(pers, 1e-10) * 100
        print(f"  t+{h:<5d} {pers:>12.4f} {learned:>12.4f} {improvement:>+11.1f}%")

    # Basin prediction
    print(f"\n--- Basin Prediction ---")
    basin_correct = 0
    basin_total = 0
    for test in test_trajectories:
        start_basin = dynamics.basin_of(*test['start'])
        for h in [5, 10, 20]:
            if h < len(test['true']) and h < len(test['predicted']):
                true_basin = dynamics.basin_of(*test['true'][h])
                pred_basin = dynamics.basin_of(*test['predicted'][h])
                basin_correct += (true_basin == pred_basin)
                basin_total += 1
    basin_acc = basin_correct / max(basin_total, 1)
    print(f"  Basin accuracy: {basin_correct}/{basin_total} = {basin_acc:.1%}")

    # ── Geometry recovery ──
    print(f"\n--- Geometry Recovery ---")

    # Fixed points
    learned_fps = model.find_fixed_points(basin_centers=basins, grid_res=30)
    actual_fps = [(cx, cy) for cx, cy, _ in basins]

    print(f"  Actual fixed points: {[(f[0], f[1]) for f in actual_fps]}")
    print(f"  Learned fixed points: {[(f[0], f[1]) for f in learned_fps]}")

    # Match learned to actual
    for i, (ax, ay) in enumerate(actual_fps):
        dists = [np.sqrt((ax-lx)**2 + (ay-ly)**2) for lx, ly in learned_fps]
        if dists:
            best = min(dists)
            print(f"  Actual basin {i} ({ax:.1f},{ay:.1f}) -> nearest learned: {best:.3f}")

    # Stability analysis at fixed points
    print(f"\n  Stability at learned fixed points:")
    for i, (fx, fy) in enumerate(learned_fps):
        eigvals = model.eigenvalues_at(fx, fy)
        real_parts = [float(np.real(ev)) for ev in eigvals]
        stable = all(r < 0 for r in real_parts)
        print(f"    FP {i} ({fx:.2f}, {fy:.2f}): eigenvalues={[f'{ev:.3f}' for ev in eigvals]} "
              f"{'STABLE' if stable else 'UNSTABLE'}")

    # ── Generalization test ──
    print(f"\n--- Generalization Test ---")
    print(f"  Training region: random points in [-2.5, 2.5]^2")
    print(f"  Test: regions between basins (unseen during training)")

    # Points between basins
    generalization_tests = [
        (0.0, -1.0, "between basin 0 and 1"),
        (-0.75, 0.25, "between basin 0 and 2"),
        (0.75, 0.25, "between basin 1 and 2"),
        (0.0, 0.0, "center (all basins equidistant)"),
    ]

    for gx, gy, desc in generalization_tests:
        vx_learned, vy_learned = model.predict_velocity(gx, gy)
        vx_true, vy_true = dynamics.velocity(gx, gy, noise=0.0)
        speed_learned = np.sqrt(vx_learned**2 + vy_learned**2)
        speed_true = np.sqrt(vx_true**2 + vy_true**2)

        # Direction match
        if speed_learned > 0.01 and speed_true > 0.01:
            direction_match = (vx_learned * vx_true + vy_learned * vy_true) / (speed_learned * speed_true)
        else:
            direction_match = 1.0  # both near zero

        print(f"  ({gx:.2f}, {gy:.2f}) [{desc}]:")
        print(f"    True:      ({vx_true:+.3f}, {vy_true:+.3f}) speed={speed_true:.3f}")
        print(f"    Learned:   ({vx_learned:+.3f}, {vy_learned:+.3f}) speed={speed_learned:.3f}")
        print(f"    Direction match: {direction_match:.3f}")

    # ── Eigenvalue landscape ──
    print(f"\n--- Eigenvalue Landscape (sampled) ---")
    eig_landscape = []
    for x in np.linspace(-2, 2, 5):
        for y in np.linspace(-2, 2, 5):
            eigvals = model.eigenvalues_at(x, y)
            real_parts = [float(np.real(ev)) for ev in eigvals]
            eig_landscape.append({
                'x': x, 'y': y,
                'eigenvalues': [float(np.real(ev)) for ev in eigvals],
                'max_real': max(real_parts),
                'stable': all(r < 0 for r in real_parts),
            })

    # Show stability regions
    stable_count = sum(1 for e in eig_landscape if e['stable'])
    print(f"  {stable_count}/{len(eig_landscape)} sampled points are locally stable")

    # ── Save results ──
    results = {
        'basins': [{'cx': cx, 'cy': cy, 'sigma': sigma} for cx, cy, sigma in basins],
        'prediction': {
            str(h): {
                'persistence_mse': float(np.mean(test_results[h]['persistence'])),
                'learned_v_mse': float(np.mean(test_results[h]['learned_v'])),
            } for h in horizons
        },
        'basin_accuracy': basin_acc,
        'fixed_points_learned': [(f[0], f[1]) for f in learned_fps],
        'eigenvalue_landscape': eig_landscape,
    }

    out = os.path.join(os.path.dirname(__file__), '..', 'learned_geometry_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results, test_trajectories, model, dynamics


if __name__ == '__main__':
    results, test_trajs, model, dynamics = run_experiment()

    # ── Stage 5: Visualization ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # --- Plot 1: Vector field + trajectories ---
        ax = axes[0]
        ax.set_title("Learned Vector Field + Test Trajectories")

        # Vector field arrows
        xs = np.linspace(-2.5, 2.5, 20)
        ys = np.linspace(-2.5, 2.5, 20)
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                vx, vy = model.predict_velocity(X[i, j], Y[i, j])
                speed = np.sqrt(vx**2 + vy**2)
                scale = min(speed, 50) / max(speed, 1e-6)
                U[i, j] = vx * scale
                V[i, j] = vy * scale
        ax.quiver(X, Y, U, V, alpha=0.4, color='gray')

        # Basin centers
        for i, (cx, cy, sigma) in enumerate(dynamics.basins):
            circle = Circle((cx, cy), sigma, fill=False, color=['red', 'blue', 'green'][i],
                           linewidth=2, linestyle='--', label=f'Basin {i}')
            ax.add_patch(circle)
            ax.plot(cx, cy, 'x', color=['red', 'blue', 'green'][i], markersize=10, mew=2)

        # Test trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, test in enumerate(test_trajs[:5]):
            true = test['true']
            pred = test['predicted']
            ax.plot(true[:, 0], true[:, 1], '-', color=colors[i], alpha=0.8, linewidth=1.5)
            ax.plot(pred[:, 0], pred[:, 1], '--', color=colors[i], alpha=0.6, linewidth=1)
            ax.plot(test['start'][0], test['start'][1], 'o', color=colors[i], markersize=5)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Potential landscape ---
        ax = axes[1]
        ax.set_title("True Potential V(x,y)")

        xs = np.linspace(-2.5, 2.5, 100)
        ys = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = dynamics.potential(X[i, j], Y[i, j])
        Z = np.clip(Z, -50, 50)

        contour = ax.contourf(X, Y, Z, levels=30, cmap='RdBu_r', alpha=0.8)
        plt.colorbar(contour, ax=ax, label='V(x,y)')
        ax.contour(X, Y, Z, levels=30, colors='black', alpha=0.2, linewidths=0.5)

        for i, (cx, cy, sigma) in enumerate(dynamics.basins):
            ax.plot(cx, cy, 'w*', markersize=12, markeredgecolor='black')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')

        # --- Plot 3: Prediction error by horizon ---
        ax = axes[2]
        ax.set_title("Prediction Error by Horizon")

        horizons = list(results['prediction'].keys())
        pers_errs = [results['prediction'][h]['persistence_mse'] for h in horizons]
        learned_errs = [results['prediction'][h]['learned_v_mse'] for h in horizons]
        h_vals = [int(h) for h in horizons]

        ax.plot(h_vals, pers_errs, 'o-', label='Persistence', color='red', linewidth=2)
        ax.plot(h_vals, learned_errs, 's-', label='Learned V', color='blue', linewidth=2)
        ax.set_xlabel('Prediction Horizon (ticks)')
        ax.set_ylabel('Mean Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        out_img = os.path.join(os.path.dirname(__file__), '..', 'learned_geometry_viz.png')
        plt.savefig(out_img, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved to {out_img}")

    except ImportError:
        print("\nmatplotlib not available — skipping visualization")
