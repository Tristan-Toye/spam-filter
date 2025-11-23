#include "email.hpp"

namespace bdap {

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
