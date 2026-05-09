# Reviewer guide

Thank you for reviewing this submission. This document is a quick map of the
repository, oriented around the claims in the paper.

## Where to start

1. **Read** the compiled submission PDF on the workshop's OpenReview page
   (15 pp: 8 pp body + refs + appendix).
2. **Skim** the appendix in the same PDF for proofs (Appendix A) and dataset
   cards (Appendix D).
3. If you want to see code behind a specific claim, use the table below.

## Claim-to-code map

| Paper location | Claim | Script(s) |
|---|---|---|
| Abstract / §3.1 | PRISM RMSE -8.58 % within-user | `scripts/calibration/eval_user_score_mse_shrunk.py` + `scripts/wave_a_W_A3_cluster_boot.py` |
| §3.2 | Cohen d / per-user gain distribution | `scripts/calibration/analyze_rmse_improvement_distribution.py` + `scripts/calibration/compute_cohen_d_bt_ll.py` |
| §3.4 | PluriHarms +9.66 % matched-procedure | `scripts/calibration/eval_pluriharms_pilsd_3backbones.py` + `scripts/q1_helpsteer2_replication.py` (PluriHarms branch) |
| §3.4 | HelpSteer2 cross-corpus replication | `scripts/calibration/eval_helpsteer2_pilsd_attribute_REML.py` + `scripts/q1_helpsteer2_replication.py` |
| §3.4 | Pooled 4-corpus PEBS | `scripts/q9_pooled_4_corpus_pilsd.py` |
| §3.5 | Cross-backbone (Mistral / Phi / Yi / Llama / Qwen) | `scripts/q2_f23_mistral_multiseed.py` + `scripts/q6_cross_backbone.py` + `scripts/calibration/build_3backbone_2corpus_matrix.py` |
| §3.5 | alpha-only / beta-only half-PEBS ablation | `scripts/q3_half_pilsd_cross_corpus.py` + `scripts/wave_b/wave_b_W_B5_half_pilsd.py` |
| §3.5 | Cross-architecture transfer | `scripts/q7_backbone_cross_architecture.py` |
| §3.6 | Sample-efficiency curve (Morris g) | `scripts/q10_sample_efficiency_curve.py` + `scripts/calibration/morris_g_validate.py` + `scripts/calibration/morris_g_2param_validate.py` |
| §3.6 | Theorem 1 oracle inequality (Halpern composition) | `scripts/q5_halpern_plus_pilsd_pipeline.py` |
| §3.7 | Single-seed DPO downstream | `scripts/q6_dpo_downstream_impact.py` |
| §3.7 | 4-seed Mistral DPO multi-seed | `scripts/q6_ms3_dpo_multiseed.py` |
| §3.7 | LoRe / PReF / P-GenRM head-to-head | `scripts/q4_pref_lore_full_embedding.py` + `scripts/baselines/` + `scripts/analysis/lore_slice_pair_acc_h2h.py` |
| §3.7 | EBPO head-to-head matrix | `scripts/analysis/ebpo_h2h_3backbone_2corpus.py` |
| §3.8 | Stress tests (heteroskedastic / compound / multipref-LOCO) | `scripts/analysis/heteroskedastic_robustness.py` + `scripts/analysis/compound_stress_test.py` + `scripts/analysis/multipref_loco_reanalysis.py` |
| §3.9 | Test-retest stability | `scripts/wave_a_W_A1_test_retest.py` |
| §3.9 | Demographic null + sign-flip stress | `scripts/wave_a_W_A4_demographic_balance.py` + `scripts/analysis/multipref_sign_flip_stress.py` |
| §3.9 | Verbosity null | `scripts/calibration/eval_user_score_mse_shrunk.py` (`--verbosity-null`) |
| Appendix A (proof) | Theorem 1 stated; proof in PDF | `scripts/q5_halpern_plus_pilsd_pipeline.py` (numerical check) |
| Falsifiability suite | F1 to F4 internal nulls | `scripts/falsifiers/F1_…F4_…py` |

## Hardware / data requirements

Most alpha,beta-fitting and analysis scripts are **CPU-only** and finish in
seconds to minutes on a laptop. The only GPU-bound code is:

- DPO downstream (`scripts/q6_ms3_dpo_multiseed.py` /
  `scripts/q6_dpo_downstream_impact.py`): about 14 GB H100 VRAM, about 3.5 h
  per seed.
- Score-PRISM-utterances (`scripts/calibration/score_prism_utterances.py`):
  about 40 GB GPU for 7B reward models.
- Cross-base panels (`scripts/q2_f23_mistral_multiseed.py` etc.) use
  pre-scored RM outputs (parquet); the fit step itself is CPU.

PRISM, PluriHarms, HelpSteer2, MultiPref, and OASST2 are not redistributed
here. Obtain them from the original sources (see paper bibliography).

## Naming note

The method was renamed late: the paper says **PEBS**, while many script
filenames and internal column names retain the historical `pilsd` token. They
implement the same method.

## Hardcoded paths

Scripts default to RunPod (`/workspace/...`) or laptop
(`<DATA_ROOT>/...`) paths. Override via CLI flags;
each script accepts `--help`.

## Anonymity

This is a private GitHub repository surfaced via anonymous.4open.science.
Please do not attempt to deanonymize; the redirect strips git metadata.
