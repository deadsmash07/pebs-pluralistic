# PEBS: Per-rater Empirical-Bayes Shrinkage for RLHF Reward-Model Calibration

Code release for the ICML 2026 Workshop on Pluralistic Alignment submission
**"PEBS: Per-rater Empirical-Bayes Shrinkage for RLHF Reward-Model Calibration"**.

This repository is shared anonymously for double-blind review via the
anonymous.4open.science mirror.

## Repository layout

```
paper/                                 LaTeX submission package + figure source
├── pluralistic_ICML2026.tex
├── pluralistic_ICML2026.pdf           compiled paper (15 pp; body 8/8)
├── references.bib
├── icml2026.sty / .bst                official ICML 2026 template
├── algorithm.sty / algorithmic.sty / fancyhdr.sty
└── figures/                           9 figure PDFs + matplotlib source

scripts/                               experiment code referenced in the paper
├── q1_helpsteer2_replication.py       §3.4 cross-corpus replication on HelpSteer2
├── q2_f23_mistral_multiseed.py        §3.5 F23 Mistral cross-base panel
├── q3_half_pilsd_cross_corpus.py      §3.5 alpha-only / beta-only ablation
├── q4_pref_lore_full_embedding.py     §3.7 PReF / LoRe head-to-head
├── q5_halpern_plus_pilsd_pipeline.py  §3.6 Halpern composition
├── q6_cross_backbone.py               §3.5 cross-backbone PEBS
├── q6_dpo_downstream_impact.py        §3.7 single-seed DPO downstream eval
├── q6_ms3_dpo_multiseed.py            §3.7 4-seed Mistral DPO multi-seed
├── q7_backbone_cross_architecture.py  §3.5 cross-architecture transfer
├── q9_pooled_4_corpus_pilsd.py        §3.4 pooled 4-corpus PEBS
├── q10_sample_efficiency_curve.py     §3.6 sample-efficiency curve
├── wave_a_*                           §3.1 / §3.9 stability + null controls
├── wave_b/                            5-seed CV + half-PEBS
├── wave_c/                            preWardBench Prop-1 demo + OASST2 replication
├── falsifiers/                        F1 to F4 internal falsification tests
├── baselines/                         LoRe / PReF / P-GenRM reimplementations on PRISM
├── calibration/                       per-rater alpha,beta fit + RMSE/pair-acc eval routines
│   ├── fit_user_calibrators*.py       core PEBS fit (least-squares + EB shrinkage)
│   ├── eval_user_score_mse*.py        within-user RMSE (pop / shrunk / quadratic / RF)
│   ├── eval_helpsteer2_pilsd_attribute*.py    HelpSteer2 5-axis REML eval
│   ├── eval_pluriharms_pilsd*.py      PluriHarms in-family + 3-backbone eval
│   ├── eval_oasst2_pilsd_calibrator.py        OASST2 cross-corpus eval
│   ├── eval_cross_user_transfer.py    leave-user-out generalization
│   ├── morris_g_*validate.py          Morris empirical-Bayes g-function forecaster
│   ├── t1_alpha_beta_*                alpha,beta cluster + semantic correlation analyses
│   ├── build_3backbone_2corpus_matrix.py      6-cell deployment-map builder
│   ├── pluriharms_practical_baselines.py      practical comparison
│   ├── run_ppo_arm_comparison.py      arm-A / arm-B PPO downstream
│   └── eval_rewardbench2.py           external RewardBench-v2 eval
├── analysis/                          post-fit analyses + drift detectors
│   ├── pilsd_*.py                     gain by demographic / cohort / conv-type / RMSE / pair-acc
│   ├── multipref_*.py                 LOCO + sign-flip stress tests
│   ├── pluriharms_robust_reanalysis.py
│   ├── ebpo_h2h_3backbone_2corpus.py  EBPO head-to-head matrix
│   ├── lore_slice_pair_acc_h2h.py     per-slice LoRe head-to-head
│   ├── compound_stress_test*.py       multi-axis stress test
│   ├── run_drift_on_*.py              prism / multipref / multiaxis drift runs
│   ├── detector_comparison_extended.py   8-detector comparison
│   ├── heteroskedastic_robustness.py  variance-misspecification check
│   ├── multiple_comparisons_correction.py    BH-FDR over claim suite
│   ├── analyze_power_curve.py
│   ├── check_paper_claims_artifacts.py        per-claim artifact existence check
│   └── compile_t3_realdata_scorecard.py
├── cohorts/                           dataset preparation
│   ├── build_multipref_cohort.py / build_multipref_multiaxis_cohort.py
│   ├── build_oasst1_author_cohort.py / build_oasst2_multiaxis_cohort.py
│   ├── build_shp_cohort.py
│   └── build_author_trajectory_cohort.py
└── plots/                             standalone plotting (paper/figures has main 9)
```

**135 Python files total**: 14 top-level (q-suite + wave_a) plus 36 analysis,
52 calibration, 6 cohort builders, 5 plots, 3 baselines, 4 falsifiers, and
4 wave_b/c.

## Naming note (PILSD to PEBS)

The method was renamed late in development. The paper text consistently uses
**PEBS** (Per-rater Empirical-Bayes Shrinkage). Many script filenames and
internal identifiers retain the historical `pilsd` token (e.g.
`scripts/q3_half_pilsd_cross_corpus.py`, `scripts/calibration/eval_pluriharms_pilsd.py`,
column names like `pilsd_rmse`). These implement the same method described as
PEBS in the paper. No Python module named `pilsd` is imported anywhere; the
rename does not break any code path. We intentionally did **not** mass-rename
internal identifiers because the historical names appear in committed result
artifacts (parquet / JSON) referenced by the analyses; renaming would have
severed result reproducibility.

## Reproducing results

### Environment
Python 3.10+, CUDA 12.x recommended for backbone-eval and DPO scripts.

```bash
pip install torch transformers datasets trl peft accelerate \
            numpy scipy pandas scikit-learn matplotlib statsmodels \
            seaborn pyarrow tqdm
```

### Data
PRISM, PluriHarms, HelpSteer2, MultiPref, and OASST2 are not redistributed
here. Obtain them from the original sources cited in `paper/references.bib`.

### Expected paths

Many scripts use argparse defaults that point at the original development
infrastructure (RunPod `/workspace/...` or laptop
`<DATA_ROOT>/...`). To run on your own machine, override
via CLI flags or set environment variables:

| Script class | Default path | Override |
|---|---|---|
| HuggingFace dataset cache | `/workspace/.hf_cache/...` | `export HF_HOME=/your/cache` |
| PRISM calibrators | `.../1_Causal_RLHF/data/prism_user_calibrators.parquet` | `--calibrators-path /your/...` |
| PluriHarms results | `.../1_Causal_RLHF/results/...` | `--results-dir /your/...` |
| Output / report dir | `.../1_Causal_RLHF/results/<exp>/` | `--output-dir /your/...` |

Every script supports `--help` for its argument list.

### Primary experiment (PRISM RMSE, ~10 minutes on a CPU)
```bash
python scripts/calibration/fit_user_calibrators.py \
  --rm-scored /your/path/prism_rm_scored.parquet \
  --output /your/path/prism_user_calibrators.parquet

python scripts/calibration/eval_user_score_mse_shrunk.py \
  --calibrators /your/path/prism_user_calibrators.parquet \
  --rm-scored /your/path/prism_rm_scored.parquet
```

### Cross-corpus replication (PluriHarms, ~20 minutes per backbone, 3 backbones)
```bash
python scripts/calibration/eval_pluriharms_pilsd_3backbones.py
```

### DPO downstream (multi-seed Mistral, ~3.5 h per seed on H100, 4 seeds)
```bash
for seed in 20260420 42 123 7777; do
  python scripts/q6_ms3_dpo_multiseed.py --seed $seed \
    --base-model-id mistralai/Mistral-7B-Instruct-v0.3
done
```

### Falsifiability suite (F1 to F4)
```bash
python scripts/falsifiers/F1_synthetic_user_recovery.py
python scripts/falsifiers/F2_rm_signal_ablation.py
python scripts/falsifiers/F3_anti_calibrator.py
python scripts/falsifiers/F4_adversarial_user_injection.py
```

Per-experiment expected wall times and verdict-class predictions are
documented in script docstrings.

## License

Code released under MIT for review purposes. See paper for citations.

## Contact

Reviewer questions go through OpenReview. Authors are anonymous during review.
