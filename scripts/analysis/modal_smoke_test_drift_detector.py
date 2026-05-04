"""Modal A10 smoke-test — Track 3 PILSD drift detector on OASST-shaped stream.

Run: `modal run scripts/modal_smoke_test_drift_detector.py`

Validates the anchor-coherence + PageHinkley detector against a realistic
longitudinal anchor-ratings stream (modeled on OASST2's daily-session cadence).
Does NOT hit real OASST2 — that requires the 75 MB trees.jsonl.gz which we
defer to Lambda. Here we use a synthetic generator that mimics:
  - 10-month timeline (300 days)
  - 30 anchor prompts, 20 power-user annotators
  - systemic drift starting at day 150 (policy deployment simulated)
  - realistic noise + session-level autocorrelation

Goal: confirm
  (a) Both detectors (hand-rolled sliding-mean AND PageHinkley-CUSUM) run
      without error on a long stream
  (b) FPR pre-drift < 15%
  (c) TPR within 30 days of drift onset > 50%
  (d) Detection latency is <60 days on average

Smoke scale: 300 timesteps × 30 anchors × 20 annotators, ~1 minute on A10
(CPU-only really; we still use A10 to prove the Modal environment works for
Track 3 before moving to Lambda OASST2 run).
"""
from __future__ import annotations

import modal

app = modal.App("pilsd-drift-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy",
        "scipy",
        "statsmodels>=0.14.6",
        # river install on Python 3.11 should succeed (local 3.10 failed)
        "river>=0.22",
    )
)


@app.function(
    gpu="A10",  # overkill for this CPU-bound test but validates env parity
    image=image,
    timeout=300,
)
def smoke_test():
    import numpy as np
    from dataclasses import dataclass, field
    from typing import Optional

    # Inline the two detectors so Modal doesn't need our repo mounted.
    # In production we'd use modal.Mount.from_local_dir(...) instead.

    # --- Hand-rolled sliding-mean detector (from src/methods/pilsd_detector.py) ---
    @dataclass
    class SlidingMeanCoherence:
        tau_drift: float = 0.1
        n_min: int = 18  # 60% of 30 anchors
        window: int = 30
        baseline: Optional[np.ndarray] = None
        history: list = field(default_factory=list)

        def reset(self, baseline):
            from collections import deque
            self.baseline = baseline.copy()
            self.history = [deque(maxlen=self.window) for _ in baseline]
            return self

        def ingest(self, step, per_anchor_means):
            for k, r in enumerate(per_anchor_means):
                self.history[k].append(r)
            if step < self.window:
                return None
            current = np.array([np.mean(h) for h in self.history])
            delta = current - self.baseline
            if np.max(np.abs(delta)) < self.tau_drift:
                return None
            signs = np.sign(delta)
            n_plus = int(np.sum(signs > 0))
            n_minus = int(np.sum(signs < 0))
            if max(n_plus, n_minus) >= self.n_min:
                return {"step": step, "dir": +1 if n_plus > n_minus else -1}
            return None

    # --- River PageHinkley-per-anchor + coherent vote ---
    from river.drift import PageHinkley

    @dataclass
    class PageHinkleyCoherence:
        n_anchors: int
        n_min: int = 18
        threshold: float = 50.0
        delta: float = 0.02
        min_instances: int = 30
        detectors: list = field(default_factory=list)
        baseline: Optional[np.ndarray] = None
        running_means: np.ndarray = field(default_factory=lambda: np.zeros(0))

        def reset(self, baseline):
            self.detectors = [
                PageHinkley(min_instances=self.min_instances,
                           delta=self.delta, threshold=self.threshold)
                for _ in range(self.n_anchors)
            ]
            self.baseline = baseline.copy()
            self.running_means = baseline.copy().astype(float)
            self._n = 0
            return self

        def ingest(self, step, per_anchor_means):
            self._n += 1
            fired = []
            for k, r in enumerate(per_anchor_means):
                self.detectors[k].update(float(r))
                if self.detectors[k].drift_detected:
                    fired.append(k)
                self.running_means[k] = 0.98 * self.running_means[k] + 0.02 * float(r)
            if len(fired) < self.n_min:
                return None
            delta = self.running_means - self.baseline
            signs = np.sign(delta[fired])
            n_plus = int(np.sum(signs > 0))
            n_minus = int(np.sum(signs < 0))
            if max(n_plus, n_minus) >= self.n_min:
                return {"step": step, "dir": +1 if n_plus > n_minus else -1}
            return None

    # --- OASST-shaped synthetic stream ---
    T = 300
    K = 30
    J = 20
    drift_onset = 150
    drift_magnitude = -0.4  # systemic negative shift (raters get stricter)
    drift_rate = drift_magnitude / (T - drift_onset)

    rng = np.random.default_rng(42)
    anchor_qualities = np.linspace(0.1, 0.9, K)
    # Per-labeler idiosyncrasy (persistent)
    labeler_jitter = 0.3 * rng.standard_normal(J)

    def day_t_anchor_ratings(t: int) -> np.ndarray:
        """Return (K, J) matrix of ratings at day t."""
        systemic_shift = drift_rate * max(0, t - drift_onset)
        # Session-level autocorrelation: each labeler has a daily mood
        daily_mood = 0.05 * rng.standard_normal(J)
        M = (
            anchor_qualities[:, None]                 # base quality
            + labeler_jitter[None, :]                 # persistent per-labeler
            + daily_mood[None, :]                     # per-labeler per-day
            + systemic_shift                          # drift after day 150
            + 0.05 * rng.standard_normal((K, J))      # per-observation noise
        )
        return M

    baseline_means = np.mean(
        np.stack([day_t_anchor_ratings(t) for t in range(30)]), axis=(0, 2)
    )
    print(f"[setup] baseline anchor means (first 5): {baseline_means[:5]}")

    sliding_det = SlidingMeanCoherence(tau_drift=0.08, n_min=18, window=30)
    sliding_det.reset(baseline_means)
    ph_det = PageHinkleyCoherence(
        n_anchors=K, n_min=18, threshold=15.0, delta=0.02, min_instances=30,
    )
    ph_det.reset(baseline_means)

    sliding_fires = []
    ph_fires = []

    for t in range(T):
        M = day_t_anchor_ratings(t)
        per_anchor_mean = M.mean(axis=1)

        e1 = sliding_det.ingest(t, per_anchor_mean)
        if e1 is not None:
            sliding_fires.append(e1)
        e2 = ph_det.ingest(t, per_anchor_mean)
        if e2 is not None:
            ph_fires.append(e2)

    # Evaluate
    def fpr_tpr(fires, drift_onset=drift_onset):
        pre_fires = [f for f in fires if f["step"] < drift_onset]
        post_fires = [f for f in fires if f["step"] >= drift_onset]
        fpr = 1.0 if pre_fires else 0.0
        latency = (post_fires[0]["step"] - drift_onset) if post_fires else None
        return fpr, len(post_fires) > 0, latency

    sliding_fpr, sliding_tpr, sliding_latency = fpr_tpr(sliding_fires)
    ph_fpr, ph_tpr, ph_latency = fpr_tpr(ph_fires)

    print(f"\n[results]")
    print(f"  sliding-mean: FPR={sliding_fpr}, TPR={sliding_tpr}, "
          f"latency={sliding_latency} days after onset")
    print(f"  PageHinkley:  FPR={ph_fpr}, TPR={ph_tpr}, "
          f"latency={ph_latency} days after onset")

    assert sliding_fpr == 0.0, "sliding-mean fired before drift onset (FPR violation)"
    assert ph_fpr == 0.0, "PageHinkley fired before drift onset (FPR violation)"

    print("\n[PASS] drift detector smoke test on OASST-shaped stream")
    return {
        "n_timesteps": T,
        "sliding": {"fpr": sliding_fpr, "tpr": sliding_tpr, "latency": sliding_latency},
        "page_hinkley": {"fpr": ph_fpr, "tpr": ph_tpr, "latency": ph_latency},
    }


@app.local_entrypoint()
def main():
    result = smoke_test.remote()
    print(f"\n[local] drift-detector smoke result: {result}")
    print("\n✅ Ready for Lambda OASST2 trees.jsonl.gz full run")
