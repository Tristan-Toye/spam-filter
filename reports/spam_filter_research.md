# Spam Filter Research Notes

## Metrics to optimize
- Precision/Recall & F1: balance user trust vs catch rate; report both micro and per-class when imbalance is high.
- False Positive Rate (FPR): key UX metric—keeps ham safe; watch absolute FP counts at volume.
- False Negative Rate (FNR): missed spam; often weighted higher for security-driven filters.
- PR-AUC over ROC-AUC: PR highlights performance under skew; ROC can look healthy even with many false positives.
- Calibration (Brier/expected calibration error): supports thresholding and downstream risk scoring.
- Throughput/latency: feature hashing and Count-Min enable O(stream) memory; track tokens/sec alongside accuracy.

## Class imbalance & thresholds
- Use stratified splits and incremental evaluation; track prevalence drift over time.
- Tune decision thresholds per channel (e.g., promotions vs personal) using cost-aware curves on FPR vs FNR.
- Consider double-threshold workflow: auto-block high-confidence spam, queue medium-confidence for secondary checks.
- Apply calibration (Platt/Isotonic) post-Naive Bayes to stabilize probabilities before thresholding.

## Naive Bayes strengths
- Fast, single-pass, low-memory; naturally suited to streaming text with incremental updates.
- Handles large vocabularies when paired with hashing; smoothing (Laplace/Lidstone) combats zero-count issues.
- Transparent contributions: token likelihoods can be surfaced for explainability/appeals.

## Naive Bayes limits and failure modes
- Independence assumption breaks on templated spam bursts; correlated features can inflate confidence.
- Hash collisions (Feature Hashing / Count-Min) smear signal across tokens—watch for recall loss on rare terms.
- Concept drift (new spam campaigns, benign trends) erodes priors; stale priors overweight legacy patterns.
- Class prior sensitivity: imbalance shifts the decision boundary; recalibrate priors when base rates move.
- Non-text signals (links, sender reputation) are absent; purely lexical models struggle with obfuscation/emoji/URL tricks.

## Mitigations and improvements
- Regularly refresh priors with sliding windows; consider decay factors or online smoothing.
- Add character/byte n-grams for obfuscation robustness; cap n-gram size to control collision rates.
- Monitor collision metrics (unique tokens vs buckets) and increase `log_buckets` when saturation rises.
- Use Count-Min for frequency estimates plus a smaller whitelist/blacklist store for high-value tokens.
- Introduce lightweight embeddings (e.g., hashed subwords) or a small linear classifier on top of NB scores for hard cases.
- Deploy canary thresholds and measure user-facing metrics (false alarm reports, missed-spam reports) before full rollout.

## Evaluation workflow checklist
- Track F1, FPR, FNR across traffic slices (locale, device, sender domain).
- Plot calibration and threshold curves each retrain; lock thresholds via validation, not test.
- Keep a drift dashboard: vocabulary churn, token KL divergence, and spam/ham mix.
- Re-run full hyperparameter sweeps when drift or collision warnings appear; validate against a fresh holdout.
