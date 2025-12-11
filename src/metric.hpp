#include "email.hpp"
#include <vector>
#include <algorithm>
#include <utility>

namespace bdap {

    // =============================================================================
    // CVMetrics: Result struct for threshold-sweep metrics (ROC/AUC analysis)
    // =============================================================================
    struct CVMetrics {
        static constexpr int NUM_METRICS = 5;
        static constexpr const char* METRIC_NAMES[NUM_METRICS] = {
            "Recall@FPR=0.01", "Recall@FPR=0.001", "AUC", 
            "Recall@Prec=0.99", "Recall@Prec=0.95"
        };

        double recall_at_fpr_001 = 0.0;   // Recall @ FPR = 1%
        double recall_at_fpr_0001 = 0.0;  // Recall @ FPR = 0.1%
        double auc = 0.0;                 // Area Under ROC Curve
        double recall_at_prec_99 = 0.0;   // Best Recall where Precision >= 99%
        double recall_at_prec_95 = 0.0;   // Best Recall where Precision >= 95%

        // Indexed access to metrics (for loops/generic code)
        double get_metric(int idx) const {
            switch (idx) {
                case 0: return recall_at_fpr_001;
                case 1: return recall_at_fpr_0001;
                case 2: return auc;
                case 3: return recall_at_prec_99;
                case 4: return recall_at_prec_95;
                default: return 0.0;
            }
        }
    };

    // =============================================================================
    // ROCMetrics: Collects predictions for threshold-sweep analysis
    // =============================================================================
    struct ROCMetrics {
        std::vector<std::pair<double, bool>> predictions;  // (score, is_spam)

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            double score = clf.predict(email);
            bool is_spam = email.is_spam();
            predictions.emplace_back(score, is_spam);
        }

        CVMetrics get_cv_metrics() const {
            CVMetrics metrics;
            if (predictions.empty()) return metrics;
            
            std::vector<std::pair<double, bool>> sorted_preds = get_sorted_predictions();
            auto [total_pos, total_neg] = count_classes(sorted_preds);
            
            if (total_pos == 0 || total_neg == 0) return metrics;
            
            compute_roc_metrics(sorted_preds, total_pos, total_neg, metrics);
            return metrics;
        }

        // Convenience: return just AUC as single score
        double get_score() const { return get_cv_metrics().auc; }

    private:
        // Sort predictions by score (descending) for threshold sweep
        std::vector<std::pair<double, bool>> get_sorted_predictions() const {
            std::vector<std::pair<double, bool>> sorted = predictions;
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            return sorted;
        }

        // Count total positives (spam) and negatives (ham)
        static std::pair<int, int> count_classes(const std::vector<std::pair<double, bool>>& preds) {
            int pos = 0, neg = 0;
            for (const auto& [score, is_spam] : preds) {
                if (is_spam) ++pos;
                else ++neg;
            }
            return {pos, neg};
        }

        // Linear interpolation for recall at a target FPR
        static double interpolate_recall(double target_fpr, double prev_fpr, double fpr,
                                         double prev_tpr, double tpr) {
            if (fpr > prev_fpr) {
                double ratio = (target_fpr - prev_fpr) / (fpr - prev_fpr);
                return prev_tpr + ratio * (tpr - prev_tpr);
            }
            return tpr;
        }

        // Single-pass ROC curve sweep computing all metrics
        static void compute_roc_metrics(const std::vector<std::pair<double, bool>>& sorted_preds,
                                        int total_pos, int total_neg, CVMetrics& metrics) {
            int tp = 0, fp = 0;
            double prev_fpr = 0.0, prev_tpr = 0.0;
            double auc = 0.0;
            
            // Trackers for recall@FPR targets
            bool found_fpr_001 = false, found_fpr_0001 = false;
            double recall_fpr_001 = 0.0, recall_fpr_0001 = 0.0;
            double best_recall_prec_99 = 0.0, best_recall_prec_95 = 0.0;
            
            for (const auto& [score, is_spam] : sorted_preds) {
                if (is_spam) ++tp;
                else ++fp;
                
                double tpr = static_cast<double>(tp) / total_pos;
                double fpr = static_cast<double>(fp) / total_neg;
                double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 1.0;
                
                // AUC via trapezoidal rule
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
                
                // Recall @ FPR = 1%
                if (!found_fpr_001 && fpr >= 0.01) {
                    recall_fpr_001 = interpolate_recall(0.01, prev_fpr, fpr, prev_tpr, tpr);
                    found_fpr_001 = true;
                }
                
                // Recall @ FPR = 0.1%
                if (!found_fpr_0001 && fpr >= 0.001) {
                    recall_fpr_0001 = interpolate_recall(0.001, prev_fpr, fpr, prev_tpr, tpr);
                    found_fpr_0001 = true;
                }
                
                // Best Recall @ Precision >= 99%
                if (precision >= 0.99)
                    best_recall_prec_99 = std::max(best_recall_prec_99, tpr);
                
                // Best Recall @ Precision >= 95%
                if (precision >= 0.95)
                    best_recall_prec_95 = std::max(best_recall_prec_95, tpr);
                
                prev_fpr = fpr;
                prev_tpr = tpr;
            }
            
            // Fallback if FPR targets never reached
            if (!found_fpr_001) recall_fpr_001 = prev_tpr;
            if (!found_fpr_0001) recall_fpr_0001 = prev_tpr;
            
            metrics.recall_at_fpr_001 = recall_fpr_001;
            metrics.recall_at_fpr_0001 = recall_fpr_0001;
            metrics.auc = auc;
            metrics.recall_at_prec_99 = best_recall_prec_99;
            metrics.recall_at_prec_95 = best_recall_prec_95;
        }
    };

    struct Accuracy {
        int n = 0;
        int correct = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
            ++n;
            correct += static_cast<int>(lab == pred);
        }

        double get_accuracy() const { return static_cast<double>(correct) / n; }
        double get_error() const { return 1.0 - get_accuracy(); }

        double get_score() const { return get_accuracy(); }
    };

    // TODO add your own metrics below here

    struct Precision {
        int tp = 0;  // true positives
        int fp = 0;  // false positives

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool actual_spam = email.is_spam();
            double pr = clf.predict(email);
            bool predicted_spam = clf.classify(pr);
            
            if (predicted_spam && actual_spam) ++tp;
            else if (predicted_spam && !actual_spam) ++fp;
        }

        double get_score() const {
            if (tp + fp == 0) return 1.0;  // no positive predictions
            return static_cast<double>(tp) / (tp + fp);
        }
    };

    struct Recall {
        int tp = 0;  // true positives
        int fn = 0;  // false negatives

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool actual_spam = email.is_spam();
            double pr = clf.predict(email);
            bool predicted_spam = clf.classify(pr);
            
            if (predicted_spam && actual_spam) ++tp;
            else if (!predicted_spam && actual_spam) ++fn;
        }

        double get_score() const {
            if (tp + fn == 0) return 1.0;  // no actual spam
            return static_cast<double>(tp) / (tp + fn);
        }
    };

    struct F1Score {
        int tp = 0;
        int fp = 0;
        int fn = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool actual_spam = email.is_spam();
            double pr = clf.predict(email);
            bool predicted_spam = clf.classify(pr);
            
            if (predicted_spam && actual_spam) ++tp;
            else if (predicted_spam && !actual_spam) ++fp;
            else if (!predicted_spam && actual_spam) ++fn;
        }

        double get_score() const {
            if (tp == 0) return 0.0;
            double precision = static_cast<double>(tp) / (tp + fp);
            double recall = static_cast<double>(tp) / (tp + fn);
            if (precision + recall == 0) return 0.0;
            return 2.0 * precision * recall / (precision + recall);
        }
    };

    struct FalsePositiveRate {
        int fp = 0;  // false positives
        int tn = 0;  // true negatives

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool actual_spam = email.is_spam();
            double pr = clf.predict(email);
            bool predicted_spam = clf.classify(pr);
            
            if (predicted_spam && !actual_spam) ++fp;
            else if (!predicted_spam && !actual_spam) ++tn;
        }

        double get_score() const {
            if (fp + tn == 0) return 0.0;  // no actual ham
            return static_cast<double>(fp) / (fp + tn);
        }
    };

    struct FalseNegativeRate {
        int fn = 0;  // false negatives
        int tp = 0;  // true positives

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool actual_spam = email.is_spam();
            double pr = clf.predict(email);
            bool predicted_spam = clf.classify(pr);
            
            if (!predicted_spam && actual_spam) ++fn;
            else if (predicted_spam && actual_spam) ++tp;
        }

        double get_score() const {
            if (tp + fn == 0) return 0.0;  // no actual spam
            return static_cast<double>(fn) / (tp + fn);
        }
    };

    // Composite metric that tracks all confusion matrix values
    struct ConfusionMetrics {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        // Result struct returned by get_score()
        struct MetricResults {
            double accuracy;
            double precision;
            double recall;
            double f1;
            double f05;  // F0.5 score - industry standard for spam filtering (weights precision 2x)
            double fpr;
            double fnr;
        };

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails) {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool actual_spam = email.is_spam();
            double pr = clf.predict(email);
            bool predicted_spam = clf.classify(pr);
            
            if (predicted_spam && actual_spam) ++tp;
            else if (predicted_spam && !actual_spam) ++fp;
            else if (!predicted_spam && actual_spam) ++fn;
            else ++tn;
        }

        MetricResults get_score() const { 
            return MetricResults{
                get_accuracy(),
                get_precision(),
                get_recall(),
                get_f1(),
                get_f05(),
                get_fpr(),
                get_fnr()
            };
        }
        
        double get_accuracy() const {
            int total = tp + tn + fp + fn;
            if (total == 0) return 0.0;
            return static_cast<double>(tp + tn) / total;
        }

        double get_precision() const {
            if (tp + fp == 0) return 1.0;
            return static_cast<double>(tp) / (tp + fp);
        }

        double get_recall() const {
            if (tp + fn == 0) return 1.0;
            return static_cast<double>(tp) / (tp + fn);
        }

        double get_f1() const {
            double prec = get_precision();
            double rec = get_recall();
            if (prec + rec == 0) return 0.0;
            return 2.0 * prec * rec / (prec + rec);
        }

        // F0.5 score - industry standard for spam filtering
        // Weights precision 2x more than recall (β = 0.5)
        // Formula: (1 + β²) * (prec * rec) / (β² * prec + rec)
        double get_f05() const {
            double prec = get_precision();
            double rec = get_recall();
            if (0.25 * prec + rec == 0) return 0.0;
            return 1.25 * prec * rec / (0.25 * prec + rec);
        }

        double get_fpr() const {
            if (fp + tn == 0) return 0.0;
            return static_cast<double>(fp) / (fp + tn);
        }

        double get_fnr() const {
            if (tp + fn == 0) return 0.0;
            return static_cast<double>(fn) / (tp + fn);
        }
    };

} // namespace bdap
