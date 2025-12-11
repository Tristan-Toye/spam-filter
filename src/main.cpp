/*
 * Copyright 2025 BDAP team.
 *
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "email.hpp"
#include "metric.hpp"
#include "base_classifier.hpp"

#include "naive_bayes_feature_hashing.hpp"
#include "naive_bayes_count_min.hpp"

using namespace bdap;

using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;


// DO NOT CHANGE!
void load_emails(std::vector<Email>& emails, const std::string& fname) {
    std::ifstream f(fname);
    if (!f.is_open()) {
        std::cerr << "Failed to open file `" << fname << "`, skipping..." << std::endl;
    } else {
        steady_clock::time_point begin = steady_clock::now();
        read_emails(f, emails);
        steady_clock::time_point end = steady_clock::now();

        std::cout << "Read " << fname << " in "
            << (duration_cast<milliseconds>(end-begin).count()/1000.0)
            << "s" << std::endl;
    }
}

// DO NOT CHANGE (except paths)!
std::vector<Email> load_emails(int seed) {
    std::vector<Email> emails;

    // Update these paths to your setup
    // Data can be found on the departmental computers in /cw/bdap/assignment1
    load_emails(emails, "C:\\Users\\trist\\OneDrive\\1_files\\2025_2026\\big data\\1-spam_filter\\src\\data\\Enron.txt");
    load_emails(emails, "C:\\Users\\trist\\OneDrive\\1_files\\2025_2026\\big data\\1-spam_filter\\src\\data\\SpamAssasin.txt");
    load_emails(emails, "C:\\Users\\trist\\OneDrive\\1_files\\2025_2026\\big data\\1-spam_filter\\src\\data\\Trec2005.txt");
    load_emails(emails, "C:\\Users\\trist\\OneDrive\\1_files\\2025_2026\\big data\\1-spam_filter\\src\\data\\Trec2006.txt");
    load_emails(emails, "C:\\Users\\trist\\OneDrive\\1_files\\2025_2026\\big data\\1-spam_filter\\src\\data\\Trec2007.txt");

    // Shuffle the emails
    std::default_random_engine g(seed);
    std::shuffle(emails.begin(), emails.end(), g);

    return emails;
}

/**
 * This function emulates a stream of emails. Every `window` examples, the
 * metric is evaluated and the score is recorded. Use the results of this
 * function to plot your learning curves.
 */
 // DO NOT CHANGE!
template <typename Clf, typename Metric>
std::vector<double>
stream_emails(const std::vector<Email> &emails,
              Clf& clf, Metric& metric, int window) {
    std::vector<double> metric_values;
    for (size_t i = 0; i < emails.size(); i+=window) {
        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            metric.evaluate(clf, emails[i+u]);

        double score = metric.get_score();
        metric_values.push_back(score);

        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            clf.update(emails[i+u]);
    }
    return metric_values;
}

/*
################################################################################
################################################################################
HELPER STRUCTS
*/


struct DataSplit {
    std::vector<size_t> dev_indices;   // 80% for development (training + validation)
    std::vector<size_t> test_indices;  // 20% held out for final evaluation
};



struct ParameterResult {
    std::string classifier;
    int ngram;
    int log_buckets;
    int num_hashes;  
    CVMetrics avg_metrics;
    std::vector<CVMetrics> fold_metrics;  
};



struct FinalConfig {
    std::string classifier;
    int ngram;
    int log_buckets;
    int num_hashes;
    double best_threshold;
    double best_f05;
    std::string optimized_for;  // Which metric from Phase 2 this was optimized for
    
    // Additional metrics at best threshold
    double precision;
    double recall;
    double f1;
    double accuracy;
};



// Metrics computed at a single threshold point
struct CurvePoint {
    double threshold;
    int tp, fp, tn, fn;
    double precision, recall, f1, f05, accuracy, tpr, fpr;
};

// Create FinalConfig from ParameterResult and best threshold
FinalConfig create_final_config(
    const ParameterResult& params,
    const CurvePoint& best,
    const std::string& optimized_for
) {
    return FinalConfig{
        params.classifier,
        params.ngram,
        params.log_buckets,
        params.num_hashes,
        best.threshold,
        best.f05,
        optimized_for,
        best.precision,
        best.recall,
        best.f1,
        best.accuracy
    };
}

/*
################################################################################
################################################################################
EVALUATION HELPERS
*/

template <typename Clf>
std::vector<ConfusionMetrics::MetricResults> 
stream_emails(const std::vector<Email> &emails,
                           Clf& clf, int window) {
    std::vector<ConfusionMetrics::MetricResults> all_metrics;
    ConfusionMetrics metric;
    
    for (size_t i = 0; i < emails.size(); i+=window) {
        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            metric.evaluate(clf, emails[i+u]);

        ConfusionMetrics::MetricResults results = metric.get_score();
        all_metrics.push_back(results);

        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            clf.update(emails[i+u]);
    }
    return all_metrics;
}


template <typename Clf>
std::vector<CVMetrics> 
stream_emails_roc(const std::vector<Email> &emails,
                  Clf& clf, int window) {
    std::vector<CVMetrics> all_metrics;
    ROCMetrics metric;
    
    for (size_t i = 0; i < emails.size(); i+=window) {
        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            metric.evaluate(clf, emails[i+u]);

        CVMetrics results = metric.get_cv_metrics();
        all_metrics.push_back(results);

        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            clf.update(emails[i+u]);
    }
    return all_metrics;
}


template <typename Clf>
void train_emails(Clf& clf, const std::vector<Email>& emails, 
                  const std::vector<size_t>& indices) {
    for (size_t idx : indices) {
        clf.update(emails[idx]);
    }
}


template <typename Clf>
CVMetrics evaluate_roc(const Clf& clf, const std::vector<Email>& emails,
                       const std::vector<size_t>& indices) {
    ROCMetrics roc_metric;
    for (size_t idx : indices) {
        roc_metric.evaluate(clf, emails[idx]);
    }
    return roc_metric.get_cv_metrics();
}

template <typename Clf>
ConfusionMetrics::MetricResults evaluate_confusion(const Clf& clf, const std::vector<Email>& emails,
                                                    const std::vector<size_t>& indices) {
    ConfusionMetrics metric;
    for (size_t idx : indices) {
        metric.evaluate(clf, emails[idx]);
    }
    return metric.get_score();
}


template <typename Clf>
std::vector<std::pair<double, bool>> collect_predictions(const Clf& clf, const std::vector<Email>& emails,
                                                          const std::vector<size_t>& indices) {
    std::vector<std::pair<double, bool>> predictions;
    predictions.reserve(indices.size());
    for (size_t idx : indices) {
        double score = clf.predict(emails[idx]);
        predictions.emplace_back(score, emails[idx].is_spam());
    }
    return predictions;
}

// Compute metrics from confusion matrix counts
CurvePoint compute_metrics(double threshold, int tp, int fp, int tn, int fn,
    int total_pos, int total_neg) {
    CurvePoint pt;
    pt.threshold = threshold;
    pt.tp = tp; pt.fp = fp; pt.tn = tn; pt.fn = fn;

    pt.precision = (tp + fp == 0) ? 1.0 : static_cast<double>(tp) / (tp + fp);
    pt.recall = (tp + fn == 0) ? 1.0 : static_cast<double>(tp) / (tp + fn);
    pt.f1 = (pt.precision + pt.recall == 0) ? 0.0 
    : 2.0 * pt.precision * pt.recall / (pt.precision + pt.recall);
    pt.f05 = (0.25 * pt.precision + pt.recall == 0) ? 0.0 
    : 1.25 * pt.precision * pt.recall / (0.25 * pt.precision + pt.recall);
    pt.accuracy = static_cast<double>(tp + tn) / (tp + fp + tn + fn);
    pt.tpr = (total_pos == 0) ? 0.0 : static_cast<double>(tp) / total_pos;
    pt.fpr = (total_neg == 0) ? 0.0 : static_cast<double>(fp) / total_neg;

    return pt;
}

// Train classifier and collect predictions based on ParameterResult
std::vector<std::pair<double, bool>> train_and_predict(
    const ParameterResult& params,
    const std::vector<Email>& emails,
    const std::vector<size_t>& indices
) {
    if (params.classifier == "FeatureHashing") {
        NaiveBayesFeatureHashing clf(params.ngram, params.log_buckets);
        train_emails(clf, emails, indices);
        return collect_predictions(clf, emails, indices);
    } else {
        NaiveBayesCountMin clf(params.ngram, params.num_hashes, params.log_buckets);
        train_emails(clf, emails, indices);
        return collect_predictions(clf, emails, indices);
    }
}


/*
################################################################################
################################################################################
DATA SPLIT HELPER
*/

// I stratified split the data into dev and test sets.
// However, since the data is already shuffled, this is not absolutely necessary.
DataSplit split_data(const std::vector<Email>& emails, double dev_ratio = 0.8) {
    DataSplit split;
    
    size_t spam_count = 0, ham_count = 0;
    for (const Email& email : emails) {
        if (email.is_spam()) ++spam_count;
        else ++ham_count;
    }
    
    size_t spam_dev_size = static_cast<size_t>(spam_count * dev_ratio);
    size_t ham_dev_size = static_cast<size_t>(ham_count * dev_ratio);
    
    split.dev_indices.reserve(spam_dev_size + ham_dev_size);
    split.test_indices.reserve((spam_count - spam_dev_size) + (ham_count - ham_dev_size));
    
    size_t spam_idx = 0, ham_idx = 0;
    for (size_t i = 0; i < emails.size(); ++i) {
        if (emails[i].is_spam()) {
            if (spam_idx++ < spam_dev_size) split.dev_indices.push_back(i);
            else split.test_indices.push_back(i);
        } else {
            if (ham_idx++ < ham_dev_size) split.dev_indices.push_back(i);
            else split.test_indices.push_back(i);
        }
    }
    
    std::default_random_engine g(42);
    std::shuffle(split.dev_indices.begin(), split.dev_indices.end(), g);
    std::shuffle(split.test_indices.begin(), split.test_indices.end(), g);
    
    return split;
}

std::vector<std::vector<size_t>> create_folds(const std::vector<size_t>& indices, int k) {
    std::vector<std::vector<size_t>> folds(k);
    for (size_t i = 0; i < indices.size(); ++i) {
        folds[i % k].push_back(indices[i]);
    }
    return folds;
}

/*
################################################################################
################################################################################
CURVE GENERATION HELPERS
*/

// Generate curve points from sorted predictions (descending by score)
// Only emits points at label boundaries to reduce output size
std::vector<CurvePoint> generate_curve_points(
    const std::vector<std::pair<double, bool>>& sorted_preds
) {
    std::vector<CurvePoint> points;
    
    int total_pos = 0, total_neg = 0;
    for (const auto& [score, is_spam] : sorted_preds) {
        if (is_spam) ++total_pos;
        else ++total_neg;
    }
    
    if (total_pos == 0 || total_neg == 0) return points;
    
    int tp = 0, fp = 0;
    for (size_t i = 0; i < sorted_preds.size(); ++i) {
        const auto& [threshold, actual_spam] = sorted_preds[i];
        
        if (actual_spam) ++tp;
        else ++fp;
        
        int fn = total_pos - tp;
        int tn = total_neg - fp;
        
        // Only emit at boundaries or label changes
        bool at_boundary = (i == 0 || i == sorted_preds.size() - 1);
        bool label_change = (i + 1 < sorted_preds.size() && actual_spam != sorted_preds[i+1].second);
        
        if (at_boundary || label_change) {
            points.push_back(compute_metrics(threshold, tp, fp, tn, fn, total_pos, total_neg));
        }
    }
    
    return points;
}

// Find best threshold by sweeping and optimizing for F0.5
CurvePoint find_best_threshold(
    const std::vector<std::pair<double, bool>>& predictions,
    double start = 0.1, double end = 0.9, double step = 0.01
) {
    int total_pos = 0, total_neg = 0;
    for (const auto& [score, is_spam] : predictions) {
        if (is_spam) ++total_pos;
        else ++total_neg;
    }
    
    CurvePoint best;
    best.f05 = -1.0;
    
    for (double thresh = start; thresh <= end; thresh += step) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (const auto& [score, actual_spam] : predictions) {
            bool predicted_spam = score > thresh;
            if (predicted_spam && actual_spam) ++tp;
            else if (predicted_spam && !actual_spam) ++fp;
            else if (!predicted_spam && !actual_spam) ++tn;
            else ++fn;
        }
        
        CurvePoint pt = compute_metrics(thresh, tp, fp, tn, fn, total_pos, total_neg);
        if (pt.f05 > best.f05) {
            best = pt;
        }
    }
    
    return best;
}

/*
################################################################################
################################################################################
CURVE WRITING HELPERS
*/

// Write PR curve point to stream
void write_pr_point(std::ostream& out, const std::string& clf, int ng, int lb, int nh,
                    const std::string& opt_for, const CurvePoint& pt) {
    out << clf << "," << ng << "," << lb << "," << nh << "," << opt_for << "," << pt.threshold << ","
        << pt.tp << "," << pt.fp << "," << pt.tn << "," << pt.fn << ","
        << pt.precision << "," << pt.recall << "," << pt.f1 << "," << pt.f05 << "," << pt.accuracy << '\n';
}

// Write ROC curve point to stream
void write_roc_point(std::ostream& out, const std::string& clf, int ng, int lb, int nh,
                     const std::string& opt_for, const CurvePoint& pt) {
    out << clf << "," << ng << "," << lb << "," << nh << "," << opt_for << "," << pt.threshold << ","
        << pt.tp << "," << pt.fp << "," << pt.tn << "," << pt.fn << ","
        << pt.tpr << "," << pt.fpr << "," << pt.accuracy << '\n';
}

// Write PR and ROC curves for a parameter set
void write_curves(
    const std::vector<std::pair<double, bool>>& predictions,
    const ParameterResult& params,
    const std::string& optimized_for,
    std::ostream& pr_out,
    std::ostream& roc_out
) {
    auto sorted_preds = predictions;
    std::sort(sorted_preds.begin(), sorted_preds.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (const auto& pt : generate_curve_points(sorted_preds)) {
        write_pr_point(pr_out, params.classifier, params.ngram, params.log_buckets,
                      params.num_hashes, optimized_for, pt);
        write_roc_point(roc_out, params.classifier, params.ngram, params.log_buckets,
                       params.num_hashes, optimized_for, pt);
    }
}


/*
################################################################################
################################################################################
EXPERIMENT STEPS
*/

std::vector<ParameterResult> cross_validation(
    const std::vector<Email>& emails,
    const std::vector<size_t>& dev_indices,
    int k_folds = 5
) {
    std::cout << "Cross-Validation Parameter Tuning" << std::endl;
    
    std::vector<ParameterResult> all_results;
    
    // Create k folds of indices (no email copying)
    std::vector<std::vector<size_t>> folds = create_folds(dev_indices, k_folds);
    std::cout << "Created " << k_folds << " folds from " << dev_indices.size() << " dev emails" << std::endl;
    
    // Parameter grids
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets_vec {10, 12, 14, 16, 18, 20};
    std::vector<int> num_hashes_vec {2, 3, 4, 5};
    
    
    std::ofstream cv_out("cv_results.csv");
    cv_out << "classifier,ngram,log_buckets,num_hashes,fold,"
           << "recall_at_fpr_001,recall_at_fpr_0001,auc,recall_at_prec_99,recall_at_prec_95" << '\n';
    
    // Feature Hashing CV
    std::cout << "\nFeature Hashing cross-validation..." << std::endl;
    for (int ng : ngrams) {
        for (int lb : log_buckets_vec) {
            ParameterResult result;
            result.classifier = "FeatureHashing";
            result.ngram = ng;
            result.log_buckets = lb;
            result.num_hashes = 0;
            result.fold_metrics.resize(k_folds);
            
            CVMetrics sum_metrics;
            
            for (int fold = 0; fold < k_folds; ++fold) {
                NaiveBayesFeatureHashing clf(ng, lb);
                
                // Train on all folds except current
                for (int f = 0; f < k_folds; ++f) {
                    if (f == fold) continue;
                    train_emails(clf, emails, folds[f]);
                }
                
                // Evaluate on validation fold
                CVMetrics fold_metrics = evaluate_roc(clf, emails, folds[fold]);
                result.fold_metrics[fold] = fold_metrics;
                
                sum_metrics.recall_at_fpr_001 += fold_metrics.recall_at_fpr_001;
                sum_metrics.recall_at_fpr_0001 += fold_metrics.recall_at_fpr_0001;
                sum_metrics.auc += fold_metrics.auc;
                sum_metrics.recall_at_prec_99 += fold_metrics.recall_at_prec_99;
                sum_metrics.recall_at_prec_95 += fold_metrics.recall_at_prec_95;
                
                cv_out << "FeatureHashing," << ng << "," << lb << ",0," << fold << ","
                       << fold_metrics.recall_at_fpr_001 << ","
                       << fold_metrics.recall_at_fpr_0001 << ","
                       << fold_metrics.auc << ","
                       << fold_metrics.recall_at_prec_99 << ","
                       << fold_metrics.recall_at_prec_95 << '\n';
            }
            
            // Average across folds
            result.avg_metrics.recall_at_fpr_001 = sum_metrics.recall_at_fpr_001 / k_folds;
            result.avg_metrics.recall_at_fpr_0001 = sum_metrics.recall_at_fpr_0001 / k_folds;
            result.avg_metrics.auc = sum_metrics.auc / k_folds;
            result.avg_metrics.recall_at_prec_99 = sum_metrics.recall_at_prec_99 / k_folds;
            result.avg_metrics.recall_at_prec_95 = sum_metrics.recall_at_prec_95 / k_folds;
            
            all_results.push_back(result);
            
            std::cout << "  FH ng=" << ng << " lb=" << lb 
                      << " | AUC=" << result.avg_metrics.auc
                      << " Rec@FPR.01=" << result.avg_metrics.recall_at_fpr_001
                      << " Rec@FPR.001=" << result.avg_metrics.recall_at_fpr_0001
                      << " Rec@P.95=" << result.avg_metrics.recall_at_prec_95 << std::endl
                      << " Rec@P.99=" << result.avg_metrics.recall_at_prec_99 << std::endl;
        }
    }
    
    // Count-Min CV
    std::cout << "\nCount-Min cross-validation..." << std::endl;
    for (int ng : ngrams) {
        for (int nh : num_hashes_vec) {
            for (int lb : log_buckets_vec) {
                ParameterResult result;
                result.classifier = "CountMin";
                result.ngram = ng;
                result.log_buckets = lb;
                result.num_hashes = nh;
                result.fold_metrics.resize(k_folds);
                
                CVMetrics sum_metrics;
                
                for (int fold = 0; fold < k_folds; ++fold) {
                    NaiveBayesCountMin clf(ng, nh, lb);
                    
                    // Train on all folds except current
                    for (int f = 0; f < k_folds; ++f) {
                        if (f == fold) continue;
                        train_emails(clf, emails, folds[f]);
                    }
                    
                    // Evaluate on validation fold
                    CVMetrics fold_metrics = evaluate_roc(clf, emails, folds[fold]);
                    result.fold_metrics[fold] = fold_metrics;
                    
                    sum_metrics.recall_at_fpr_001 += fold_metrics.recall_at_fpr_001;
                    sum_metrics.recall_at_fpr_0001 += fold_metrics.recall_at_fpr_0001;
                    sum_metrics.auc += fold_metrics.auc;
                    sum_metrics.recall_at_prec_99 += fold_metrics.recall_at_prec_99;
                    sum_metrics.recall_at_prec_95 += fold_metrics.recall_at_prec_95;
                    
                    cv_out << "CountMin," << ng << "," << lb << "," << nh << "," << fold << ","
                           << fold_metrics.recall_at_fpr_001 << ","
                           << fold_metrics.recall_at_fpr_0001 << ","
                           << fold_metrics.auc << ","
                           << fold_metrics.recall_at_prec_99 << ","
                           << fold_metrics.recall_at_prec_95 << '\n';
                }
                
                // Average across folds
                result.avg_metrics.recall_at_fpr_001 = sum_metrics.recall_at_fpr_001 / k_folds;
                result.avg_metrics.recall_at_fpr_0001 = sum_metrics.recall_at_fpr_0001 / k_folds;
                result.avg_metrics.auc = sum_metrics.auc / k_folds;
                result.avg_metrics.recall_at_prec_99 = sum_metrics.recall_at_prec_99 / k_folds;
                result.avg_metrics.recall_at_prec_95 = sum_metrics.recall_at_prec_95 / k_folds;
                
                all_results.push_back(result);
                
                std::cout << "  CM ng=" << ng << " nh=" << nh << " lb=" << lb 
                          << " | AUC=" << result.avg_metrics.auc
                          << " Rec@FPR.01=" << result.avg_metrics.recall_at_fpr_001
                          << " Rec@FPR.001=" << result.avg_metrics.recall_at_fpr_0001
                          << " Rec@P.95=" << result.avg_metrics.recall_at_prec_95 << std::endl
                          << " Rec@P.99=" << result.avg_metrics.recall_at_prec_99 << std::endl;
            }
        }
    }
    
    cv_out.close();
    std::cout << "\nWrote CV results to cv_results.csv" << std::endl;
    
    return all_results;
}


std::vector<ParameterResult> select_best_parameters(
    const std::vector<ParameterResult>& all_results
) {

    std::cout << "Selecting best parameters..." << std::endl;
    
    // Track best for each (classifier, metric) pair in single pass
    // Index: [classifier_idx][metric_idx], classifier: 0=FeatureHashing, 1=CountMin
    constexpr int NUM_CLASSIFIERS = 2;
    const std::string classifier_names[NUM_CLASSIFIERS] = {"FeatureHashing", "CountMin"};
    
    std::array<std::array<const ParameterResult*, CVMetrics::NUM_METRICS>, NUM_CLASSIFIERS> best_ptrs{};
    std::array<std::array<double, CVMetrics::NUM_METRICS>, NUM_CLASSIFIERS> best_scores{};
    
    // Initialize scores to -1
    for (int c = 0; c < NUM_CLASSIFIERS; ++c)
        for (int m = 0; m < CVMetrics::NUM_METRICS; ++m)
            best_scores[c][m] = -1.0;
    
    // Single pass through all results
    for (const auto& result : all_results) {
        int clf_idx = (result.classifier == "FeatureHashing") ? 0 : 1;
        
        for (int m = 0; m < CVMetrics::NUM_METRICS; ++m) {
            double score = result.avg_metrics.get_metric(m);
            if (score > best_scores[clf_idx][m]) {
                best_scores[clf_idx][m] = score;
                best_ptrs[clf_idx][m] = &result;
            }
        }
    }
    
    // Collect results and report
    std::vector<ParameterResult> best_params;
    
    for (int c = 0; c < NUM_CLASSIFIERS; ++c) {
        std::cout << "\n" << classifier_names[c] << ":" << std::endl;
        
        for (int m = 0; m < CVMetrics::NUM_METRICS; ++m) {
            if (best_ptrs[c][m]) {
                const auto& best = *best_ptrs[c][m];
                best_params.push_back(best);
                
                std::cout << "  Best for " << CVMetrics::METRIC_NAMES[m] 
                          << ": ng=" << best.ngram << " lb=" << best.log_buckets;
                if (c == 1) std::cout << " nh=" << best.num_hashes;
                std::cout << " (score=" << best_scores[c][m] << ")" << std::endl;
            }
        }
    }
    
    return best_params;
}



std::vector<FinalConfig> threshold_tuning(
    const std::vector<Email>& emails,
    const std::vector<size_t>& dev_indices,
    const std::vector<ParameterResult>& best_params
) {
    std::cout << "Tuning thresholds on full dev set..." << std::endl;
    
    std::vector<FinalConfig> final_configs;
    
    std::ofstream pr_out("pr_curve.csv");
    pr_out << "classifier,ngram,log_buckets,num_hashes,optimized_for,threshold,"
           << "tp,fp,tn,fn,precision,recall,f1,f05,accuracy" << '\n';
    
    std::ofstream roc_out("roc_curve.csv");
    roc_out << "classifier,ngram,log_buckets,num_hashes,optimized_for,threshold,"
            << "tp,fp,tn,fn,tpr,fpr,accuracy" << '\n';
    
    int config_idx = 0;
    for (const auto& params : best_params) {
        std::string optimized_for = CVMetrics::METRIC_NAMES[config_idx++ % CVMetrics::NUM_METRICS];
        
        std::cout << "\nTraining " << params.classifier << " (ng=" << params.ngram 
                  << " lb=" << params.log_buckets;
        if (params.classifier == "CountMin") std::cout << " nh=" << params.num_hashes;
        std::cout << ") optimized for " << optimized_for << "..." << std::endl;
        
        auto predictions = train_and_predict(params, emails, dev_indices);
        write_curves(predictions, params, optimized_for, pr_out, roc_out);
        
        CurvePoint best = find_best_threshold(predictions);
        final_configs.push_back(create_final_config(params, best, optimized_for));
        
        std::cout << "  Best threshold=" << best.threshold << " F0.5=" << best.f05 
                  << " P=" << best.precision << " R=" << best.recall << std::endl;
    }
    
    pr_out.close();
    roc_out.close();
    
    // Save final configs
    std::ofstream config_out("final_configs.csv");
    config_out << "classifier,ngram,log_buckets,num_hashes,optimized_for,best_threshold,"
               << "f05,precision,recall,f1,accuracy" << '\n';
    for (const auto& cfg : final_configs) {
        config_out << cfg.classifier << "," << cfg.ngram << "," << cfg.log_buckets << ","
                   << cfg.num_hashes << "," << cfg.optimized_for << "," << cfg.best_threshold << ","
                   << cfg.best_f05 << "," << cfg.precision << "," << cfg.recall << ","
                   << cfg.f1 << "," << cfg.accuracy << '\n';
    }
    config_out.close();
    
    std::cout << "\nWrote PR curves to pr_curve.csv" << std::endl;
    std::cout << "Wrote ROC curves to roc_curve.csv" << std::endl;
    std::cout << "Wrote final configs to final_configs.csv" << std::endl;
    
    return final_configs;
}



void evaluation(
    const std::vector<Email>& emails,
    const std::vector<size_t>& test_indices,
    const std::vector<FinalConfig>& configs
) {
    std::cout << "\n=== PHASE 4: Final Test Evaluation ===" << std::endl;
    std::cout << "Evaluating " << configs.size() << " configurations on " 
              << test_indices.size() << " test emails..." << std::endl;
    
    // Extract test emails once for online streaming
    std::vector<Email> test_emails;
    test_emails.reserve(test_indices.size());
    for (size_t idx : test_indices) test_emails.push_back(emails[idx]);
    
    std::ofstream test_out("evalution_results.csv");
    test_out << "classifier,ngram,log_buckets,num_hashes,optimized_for,threshold,"
             << "test_precision,test_recall,test_f1,test_f05,test_accuracy,test_fpr,test_fnr,"
             << "dev_precision,dev_recall,dev_f1,dev_f05,dev_accuracy" << '\n';
    
    for (const auto& cfg : configs) {
        ConfusionMetrics::MetricResults test;
        
        if (cfg.classifier == "FeatureHashing") {
            NaiveBayesFeatureHashing clf(cfg.ngram, cfg.log_buckets, cfg.best_threshold);
            auto results = stream_emails(test_emails, clf, 1);
            test = results.back();  
        } else {
            NaiveBayesCountMin clf(cfg.ngram, cfg.num_hashes, cfg.log_buckets, cfg.best_threshold);
            auto results = stream_emails(test_emails, clf, 1);
            test = results.back();  
        }
        
        test_out << cfg.classifier << "," << cfg.ngram << "," << cfg.log_buckets << ","
                 << cfg.num_hashes << "," << cfg.optimized_for << "," << cfg.best_threshold << ","
                 << test.precision << "," << test.recall << "," << test.f1 << "," << test.f05 << ","
                 << test.accuracy << "," << test.fpr << "," << test.fnr << ","
                 << cfg.precision << "," << cfg.recall << "," << cfg.f1 << "," << cfg.best_f05 << ","
                 << cfg.accuracy << '\n';
        
        std::cout << "\n" << cfg.classifier << " (optimized for " << cfg.optimized_for << "):" << std::endl;
        std::cout << "  Params: ng=" << cfg.ngram << " lb=" << cfg.log_buckets;
        if (cfg.classifier == "CountMin") std::cout << " nh=" << cfg.num_hashes;
        std::cout << " thresh=" << cfg.best_threshold << std::endl;
        std::cout << "  TEST Results: F0.5=" << test.f05 << " P=" << test.precision 
                  << " R=" << test.recall << " Acc=" << test.accuracy << std::endl;
        std::cout << "  DEV Results:  F0.5=" << cfg.best_f05 << " P=" << cfg.precision 
                  << " R=" << cfg.recall << " Acc=" << cfg.accuracy << std::endl;
        
        double f05_diff = test.f05 - cfg.best_f05;
        if (f05_diff < -0.05) {
            std::cout << "  WARNING: Test F0.5 is " << (-f05_diff * 100) << "% lower - possible overfitting!" << std::endl;
        }
    }
    
    test_out.close();
    std::cout << "\nWrote final test results to evaluation_results.csv" << std::endl;
}

/*
################################################################################
################################################################################
EXPERIMENTS
*/

std::vector<FinalConfig> experiment_parameter_search(const std::vector<Email>& emails, int seed) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CROSS-VALIDATION EXPERIMENT" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // stratified split
    std::cout <<  std::string(60, '=') << std::endl;
    std::cout << "Stratified splitting data into dev and test sets..." << std::endl;
    DataSplit split = split_data(emails, 0.8);
    std::cout << "Total emails: " << emails.size() << std::endl;
    std::cout << "Dev set (80%): " << split.dev_indices.size() << std::endl;
    std::cout << "Test set (20%): " << split.test_indices.size() << std::endl;
    size_t dev_spam = 0, test_spam = 0;
    for (size_t idx : split.dev_indices) if (emails[idx].is_spam()) ++dev_spam;
    for (size_t idx : split.test_indices) if (emails[idx].is_spam()) ++test_spam;
    std::cout << "Dev spam: " << dev_spam << " (" << (100.0 * dev_spam / split.dev_indices.size()) << "%)" << std::endl;
    std::cout << "Test spam: " << test_spam << " (" << (100.0 * test_spam / split.test_indices.size()) << "%)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << std::string(60, '=') << std::endl;
    std::vector<ParameterResult> cv_results = cross_validation(emails, split.dev_indices, 5);

    std::cout << std::string(60, '=') << std::endl;
    std::vector<ParameterResult> best_params = select_best_parameters(cv_results);

    std::cout << std::string(60, '=') << std::endl;
    std::vector<FinalConfig> final_configs = threshold_tuning(emails, split.dev_indices, best_params);
    
    
    std::cout << std::string(60, '=') << std::endl;
    evaluation(emails, split.test_indices, final_configs);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CROSS-VALIDATION EXPERIMENT COMPLETE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return final_configs;
}

// Helper: Collect streaming predictions (predict then update)
template <typename Clf>
std::vector<std::pair<double, bool>> collect_streaming_predictions(
    Clf& clf, const std::vector<Email>& emails
) {
    std::vector<std::pair<double, bool>> predictions;
    predictions.reserve(emails.size());
    for (const Email& email : emails) {
        predictions.emplace_back(clf.predict(email), email.is_spam());
        clf.update(email);
    }
    return predictions;
}

void experiment_learning_curves(const std::vector<Email>& emails,
                                const std::vector<FinalConfig>& configs) {
    std::cout << "generating learning curves..." << std::endl;
    
    std::vector<int> window_sizes {200, 500, 1000, 2000};
    std::ofstream lc_out("learning_curves.csv");
    lc_out << "classifier,window,ngram,log_buckets,num_hashes,optimized_for,threshold,step,num_examples,"
           << "accuracy,precision,recall,f1,f05,fpr,fnr" << '\n';

    for (const auto& cfg : configs) {
        std::cout << "\nLearning curves for " << cfg.classifier 
                  << " (optimized for " << cfg.optimized_for << "):" << std::endl;
        
        for (int ws : window_sizes) {
            std::vector<ConfusionMetrics::MetricResults> curve;
            
            if (cfg.classifier == "FeatureHashing") {
                NaiveBayesFeatureHashing clf(cfg.ngram, cfg.log_buckets, cfg.best_threshold);
                curve = stream_emails(emails, clf, ws);
            } else {
                NaiveBayesCountMin clf(cfg.ngram, cfg.num_hashes, cfg.log_buckets, cfg.best_threshold);
                curve = stream_emails(emails, clf, ws);
            }
            
            // Write results for each step
            for (size_t step = 0; step < curve.size(); ++step) {
                size_t num_ex = std::min(static_cast<size_t>((step + 1) * ws), emails.size());
                const auto& r = curve[step];
                
                lc_out << cfg.classifier << "," << ws << "," << cfg.ngram << "," 
                       << cfg.log_buckets << "," << cfg.num_hashes << ","
                       << cfg.optimized_for << "," << cfg.best_threshold << ","
                       << step << "," << num_ex << ","
                       << r.accuracy << "," << r.precision << "," << r.recall << ","
                       << r.f1 << "," << r.f05 << "," << r.fpr << "," << r.fnr << '\n';
            }
            
            const auto& final_r = curve.back();
            std::cout << "  window=" << ws << ": F0.5=" << final_r.f05 
                      << " P=" << final_r.precision << " R=" << final_r.recall << std::endl;
        }
    }
    
    lc_out.close();
    std::cout << "\nWrote learning curves to learning_curves.csv" << std::endl;
}
/*
void experiment_pr_curve(const std::vector<Email>& emails) {
    std::cout << "generating PR curves..." << std::endl;
    
    std::ofstream pr_out("pr_curve.csv");
    pr_out << "classifier,ngram,log_buckets,num_hashes,threshold,tp,fp,tn,fn,"
           << "precision,recall,f1,f05,accuracy" << '\n';
    
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets {10, 12, 14, 16};
    std::vector<int> cm_num_hashes {2, 3, 4};
    
    // Feature Hashing PR curves
    for (int ng : ngrams) {
        for (int lb : log_buckets) {
            std::cout << "Training Feature Hashing (ng=" << ng << ", lb=" << lb << ")..." << std::endl;
            
            NaiveBayesFeatureHashing clf(ng, lb);
            auto predictions = collect_streaming_predictions(clf, emails);
            
            std::sort(predictions.begin(), predictions.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            
            for (const auto& pt : generate_curve_points(predictions)) {
                write_pr_point(pr_out, "FeatureHashing", ng, lb, 0, "", pt);
            }
        }
    }
    
    // Count-Min PR curves
    for (int ng : ngrams) {
        for (int nh : cm_num_hashes) {
            for (int lb : log_buckets) {
                std::cout << "Training Count-Min (ng=" << ng << ", nh=" << nh << ", lb=" << lb << ")..." << std::endl;
                
                NaiveBayesCountMin clf(ng, nh, lb);
                auto predictions = collect_streaming_predictions(clf, emails);
                
                std::sort(predictions.begin(), predictions.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                
                for (const auto& pt : generate_curve_points(predictions)) {
                    write_pr_point(pr_out, "CountMin", ng, lb, nh, "", pt);
                }
            }
        }
    }
    
    pr_out.close();
    std::cout << "Wrote PR curve data to pr_curve.csv" << std::endl;
}

void experiment_roc_curve(const std::vector<Email>& emails) {
    std::cout << "generating ROC curves..." << std::endl;
    
    std::ofstream roc_out("roc_curve.csv");
    roc_out << "classifier,ngram,log_buckets,num_hashes,threshold,tp,fp,tn,fn,"
            << "tpr,fpr,accuracy" << '\n';
    
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets {10, 12, 14, 16};
    std::vector<int> cm_num_hashes {2, 3, 4};
    
    // Feature Hashing ROC curves
    for (int ng : ngrams) {
        for (int lb : log_buckets) {
            std::cout << "Training Feature Hashing (ng=" << ng << ", lb=" << lb << ")..." << std::endl;
            
            NaiveBayesFeatureHashing clf(ng, lb);
            auto predictions = collect_streaming_predictions(clf, emails);
            
            std::sort(predictions.begin(), predictions.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            
            for (const auto& pt : generate_curve_points(predictions)) {
                write_roc_point(roc_out, "FeatureHashing", ng, lb, 0, "", pt);
            }
        }
    }
    
    // Count-Min ROC curves
    for (int ng : ngrams) {
        for (int nh : cm_num_hashes) {
            for (int lb : log_buckets) {
                std::cout << "Training Count-Min (ng=" << ng << ", nh=" << nh << ", lb=" << lb << ")..." << std::endl;
                
                NaiveBayesCountMin clf(ng, nh, lb);
                auto predictions = collect_streaming_predictions(clf, emails);
                
                std::sort(predictions.begin(), predictions.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                
                for (const auto& pt : generate_curve_points(predictions)) {
                    write_roc_point(roc_out, "CountMin", ng, lb, nh, "", pt);
                }
            }
        }
    }
    
    roc_out.close();
    std::cout << "Wrote ROC curve data to roc_curve.csv" << std::endl;
}
*/
void experiment_computational_efficiency(const std::vector<Email>& emails) {
    std::cout << "timing computational efficiency..." << std::endl;
    
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets {10, 12, 14, 16};
    std::vector<int> cm_num_hashes {2, 3, 4};
    int ws = 200;
    
    std::ofstream timing_out("timing_results.csv");
    timing_out << "classifier,ngram,log_buckets,num_hashes,total_time_s,time_per_email_ms" << '\n';
    
    // Feature Hashing timing
    for (int ng : ngrams) {
        for (int lb : log_buckets) {
            NaiveBayesFeatureHashing clf(ng, lb);
            Accuracy metric;
            
            auto start = steady_clock::now();
            stream_emails(emails, clf, metric, ws);
            auto end = steady_clock::now();
            
            double total_time = duration_cast<milliseconds>(end - start).count() / 1000.0;
            double time_per_email = (total_time * 1000.0) / emails.size();
            
            timing_out << "FeatureHashing," << ng << "," << lb << ",0,"
                      << total_time << "," << time_per_email << '\n';
            
            std::cout << "FH timing ngram=" << ng << " log_buckets=" << lb
                     << " time=" << total_time << "s" << std::endl;
        }
    }
    
    // Count-Min timing
    for (int ng : ngrams) {
        for (int nh : cm_num_hashes) {
            for (int lb : log_buckets) {
                NaiveBayesCountMin clf(ng, nh, lb);
                Accuracy metric;
                
                auto start = steady_clock::now();
                stream_emails(emails, clf, metric, ws);
                auto end = steady_clock::now();
                
                double total_time = duration_cast<milliseconds>(end - start).count() / 1000.0;
                double time_per_email = (total_time * 1000.0) / emails.size();
                
                timing_out << "CountMin," << ng << "," << lb << "," << nh << ","
                          << total_time << "," << time_per_email << '\n';
                
                std::cout << "CM timing ngram=" << ng << " num_hashes=" << nh << " log_buckets=" << lb
                         << " time=" << total_time << "s" << std::endl;
            }
        }
    }
    
    timing_out.close();
    std::cout << "Wrote timing results to timing_results.csv" << std::endl;
}

int main(int argc, char *argv[]) { 
    // The arguments can be used for your experiments, specify the correct command(s) in your script.sh

    // Example on how to load the data + statistics on spam/ham
    int seed = 12;
    if (argc > 1) {
        try { seed = std::stoi(argv[1]); } catch (...) {}
    }
    
    std::vector<Email> emails = load_emails(seed);
    std::cout << "#emails: " << emails.size() << std::endl;
    size_t num_spam = 0;
    for (const Email& e : emails)
        num_spam += e.is_spam();
    std::cout << "#spam: " << num_spam << ", "
              << (100.0 * num_spam / emails.size()) << "%"
              << std::endl;

    std::vector<FinalConfig> configs = experiment_parameter_search(emails, seed);
    
    // Run computational efficiency experiment
    experiment_learning_curves(emails, configs);
    experiment_computational_efficiency(emails);
    
    std::cout << "\n=== ALL EXPERIMENTS COMPLETE ===" << std::endl;

    return 0;
}