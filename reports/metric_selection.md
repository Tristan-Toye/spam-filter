# Spam Filter Metric Selection: F1 vs F0.5

## Summary

For spam filtering, **F0.5 is the industry standard** metric to optimize, not F1. This is because precision should be weighted more heavily than recall in email spam filtering applications.

## Why Precision > Recall for Spam Filters

| Error Type | Description | Cost |
|------------|-------------|------|
| **False Positive** (FP) | Legitimate email → spam folder | **HIGH** - missed business deals, job offers, important communications (~$3.50/message recovery cost + potential lost opportunities) |
| **False Negative** (FN) | Spam → inbox | **LOW** - user deletes it manually, minor annoyance |

**Key insight**: If an important email goes to spam, the user might miss it entirely. If spam gets through, users can just delete it — annoying but not catastrophic.

## Industry Standard Metrics Comparison

| Metric | β value | Precision:Recall Weight | Use Case |
|--------|---------|------------------------|----------|
| **F0.5** | 0.5 | **Precision weighted 2x** | ✅ **Spam filtering standard** - minimize false positives |
| F1 | 1.0 | Equal weight | General-purpose balanced metric |
| F2 | 2.0 | Recall weighted 2x | Security/fraud detection where missing threats is dangerous |

## The F-Beta Formula

The general F-beta score formula:

```
F_β = (1 + β²) × (precision × recall) / (β² × precision + recall)
```

For **F0.5** specifically:

```
F0.5 = 1.25 × (precision × recall) / (0.25 × precision + recall)
```

## Recommendations for This Project

1. **Primary optimization metric**: **F0.5** (industry standard for email spam filtering)
2. **Secondary constraint**: Keep FPR (False Positive Rate) < 1-2% — users will abandon a spam filter that incorrectly classifies legitimate mail as spam
3. **Report all metrics** for transparency and analysis:
   - Accuracy
   - Precision
   - Recall
   - F1 (for comparison)
   - F0.5 (primary metric)
   - FPR (False Positive Rate)
   - FNR (False Negative Rate)

## References

- Industry spam filter cost analysis estimates ~$3.50 per false positive for message recovery (ComputerWorld)
- F-beta score weighting: GeeksforGeeks Machine Learning documentation
- Email Filtering Effectiveness Scoring: OpenEFA documentation
