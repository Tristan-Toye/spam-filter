/*
 * Copyright 2025 BDAP team.
 *
 */

#include <algorithm>
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

// Struct to hold best hyperparameters from grid search
struct BestHyperparameters {
    // Feature Hashing best params
    int fh_ngram;
    int fh_log_buckets;
    double fh_f05;
    double fh_precision;
    double fh_recall;
    double fh_f1;
    double fh_accuracy;
    
    // Count-Min best params
    int cm_ngram;
    int cm_log_buckets;
    int cm_num_hashes;
    double cm_f05;
    double cm_precision;
    double cm_recall;
    double cm_f1;
    double cm_accuracy;
};

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

/**
 * Enhanced version that returns all metrics from ConfusionMetrics.
 * Returns vector of MetricResults structs containing all metrics.
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

void experiment1_learning_curves(const std::vector<Email>& emails) {
    std::cout << "\n=== EXPERIMENT 1: Learning Curves ===" << std::endl;
    
    std::vector<int> window_sizes {200, 500, 1000, 2000};
    std::ofstream lc_out("learning_curves.csv");
    lc_out << "classifier,window,ngram,log_buckets,num_hashes,step,num_examples,"
           << "accuracy,precision,recall,f1,f05,fpr,fnr" << '\n';
    int ngram = 2;
    int log_buckets = 14;
    int num_hashes = 3;

    // Feature Hashing learning curves
    for (int ws : window_sizes) {

        NaiveBayesFeatureHashing fh_clf(ngram, log_buckets);  // good defaults
        std::vector<ConfusionMetrics::MetricResults> fh_curve = stream_emails(emails, fh_clf, ws);
        
        // Write results for each step
        for (size_t step = 0; step < fh_curve.size(); ++step) {
            size_t num_ex = std::min(static_cast<size_t>((step+1)*ws), emails.size());
            
            
            ConfusionMetrics::MetricResults results = fh_curve[step];
            
            lc_out << "FeatureHashing," << ws << "," << ngram << "," << log_buckets << ",1," << step << "," << num_ex << ","
                   << results.accuracy << ","
                   << results.precision << ","
                   << results.recall << ","
                   << results.f1 << ","
                   << results.f05 << ","
                   << results.fpr << ","
                   << results.fnr << '\n';
        }
        std::cout << "FH learning curve window=" << ws << " complete" << std::endl;
        std::cout << "FH parameters: ngram=" << ngram << ", log_buckets=" << log_buckets << std::endl;
        ConfusionMetrics::MetricResults results = fh_curve.back();
        std::cout << "FH last step results: " << std::endl
                  << "Accuracy: " << results.accuracy << std::endl
                  << "Precision: " << results.precision << std::endl
                  << "Recall: " << results.recall << std::endl
                  << "F1: " << results.f1 << std::endl
                  << "F0.5: " << results.f05 << std::endl
                  << "FPR: " << results.fpr << std::endl
                  << "FNR: " << results.fnr << std::endl;
    }
    
    // Count-Min learning curves
    for (int ws : window_sizes) {
        NaiveBayesCountMin cm_clf(ngram, num_hashes, log_buckets);  // good defaults
        std::vector<ConfusionMetrics::MetricResults> cm_curve = stream_emails(emails, cm_clf, ws);
        
        // Write results for each step
        for (size_t step = 0; step < cm_curve.size(); ++step) {
            size_t num_ex = std::min(static_cast<size_t>((step+1)*ws), emails.size());
            
            ConfusionMetrics::MetricResults results = cm_curve[step];
            
            lc_out << "CountMin," << ws << "," << ngram << "," << log_buckets << "," << num_hashes << "," << step << "," << num_ex << ","
                   << results.accuracy << ","
                   << results.precision << ","
                   << results.recall << ","
                   << results.f1 << ","
                   << results.f05 << ","
                   << results.fpr << ","
                   << results.fnr << '\n';

        }
        std::cout << "CM learning curve window=" << ws << " complete" << std::endl;
        std::cout << "CM parameters: ngram=" << ngram << ", num_hashes=" << num_hashes << ", log_buckets=" << log_buckets << std::endl;
        ConfusionMetrics::MetricResults results = cm_curve.back();
        std::cout << "CM last step results: " << std::endl
                  << "Accuracy: " << results.accuracy << std::endl
                  << "Precision: " << results.precision << std::endl
                  << "Recall: " << results.recall << std::endl
                  << "F1: " << results.f1 << std::endl
                  << "F0.5: " << results.f05 << std::endl
                  << "FPR: " << results.fpr << std::endl
                  << "FNR: " << results.fnr << std::endl;
    }
    
    lc_out.close();
    std::cout << "Wrote learning curves to learning_curves.csv" << std::endl;
}

BestHyperparameters experiment2_hyperparameter_grid_search(const std::vector<Email>& emails) {
    std::cout << "\n=== EXPERIMENT 2: Hyperparameter Grid Search ===" << std::endl;
    
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets {10, 12, 14, 16};
    std::vector<int> cm_num_hashes {2, 3, 4};
    int ws = 200;
    
    std::ofstream hyper_out("hyperparameter_results.csv");
    hyper_out << "classifier,ngram,log_buckets,num_hashes,num_examples,"
              << "accuracy,precision,recall,f1,f05,fpr,fnr" << '\n';
    
    // Track best parameters based on F0.5
    BestHyperparameters best = {};
    best.fh_f05 = -1.0;
    best.cm_f05 = -1.0;
    
    // Feature Hashing grid search
    for (int ng : ngrams) {
        for (int lb : log_buckets) {
            NaiveBayesFeatureHashing clf(ng, lb);
            
            std::vector<ConfusionMetrics::MetricResults> fh_curve = stream_emails(emails, clf, ws);

            for (size_t step = 0; step < fh_curve.size(); ++step) {
                size_t num_ex = std::min(static_cast<size_t>((step+1)*ws), emails.size());
                ConfusionMetrics::MetricResults results = fh_curve[step];
                
                hyper_out << "FeatureHashing," << ng << "," << lb << ",0," << num_ex << ","
                         << results.accuracy << ","
                         << results.precision << ","
                         << results.recall << ","
                         << results.f1 << ","
                         << results.f05 << ","
                         << results.fpr << ","
                         << results.fnr << '\n';
            }
            ConfusionMetrics::MetricResults results = fh_curve.back();
            std::cout << "FH ngram=" << ng << " log_buckets=" << lb
                     << " precision=" << results.precision
                     << " recall=" << results.recall    
                     << " f1=" << results.f1
                     << " f0.5=" << results.f05 << std::endl;
            
            // Track best F0.5 for Feature Hashing
            if (results.f05 > best.fh_f05) {
                best.fh_f05 = results.f05;
                best.fh_ngram = ng;
                best.fh_log_buckets = lb;
                best.fh_precision = results.precision;
                best.fh_recall = results.recall;
                best.fh_f1 = results.f1;
                best.fh_accuracy = results.accuracy;
            }
        }
    }
    
    // Count-Min grid search
    for (int ng : ngrams) {
        for (int nh : cm_num_hashes) {
            for (int lb : log_buckets) {
                NaiveBayesCountMin clf(ng, nh, lb);
                std::vector<ConfusionMetrics::MetricResults> cm_curve = stream_emails(emails, clf, ws);
                
                for (size_t step = 0; step < cm_curve.size(); ++step) {
                    size_t num_ex = std::min(static_cast<size_t>((step+1)*ws), emails.size());
                    
                    ConfusionMetrics::MetricResults results = cm_curve[step];
                    
                    hyper_out << "CountMin," << ng << "," << lb << "," << nh << "," << num_ex << ","
                             << results.accuracy << ","
                             << results.precision << ","
                             << results.recall << ","
                             << results.f1 << ","
                             << results.f05 << ","
                             << results.fpr << ","
                             << results.fnr << '\n';
                }
                ConfusionMetrics::MetricResults results = cm_curve.back();
                std::cout << "CM ngram=" << ng << " num_hashes=" << nh << " log_buckets=" << lb
                         << " precision=" << results.precision
                         << " recall=" << results.recall    
                         << " f1=" << results.f1
                         << " f0.5=" << results.f05 << std::endl;
                
                // Track best F0.5 for Count-Min
                if (results.f05 > best.cm_f05) {
                    best.cm_f05 = results.f05;
                    best.cm_ngram = ng;
                    best.cm_log_buckets = lb;
                    best.cm_num_hashes = nh;
                    best.cm_precision = results.precision;
                    best.cm_recall = results.recall;
                    best.cm_f1 = results.f1;
                    best.cm_accuracy = results.accuracy;
                }
            }
        }
    }
    
    hyper_out.close();
    std::cout << "Wrote hyperparameter results to hyperparameter_results.csv" << std::endl;
    
    // Report best configurations based on F0.5
    std::cout << "\n=== BEST CONFIGURATIONS (optimized for F0.5) ===" << std::endl;
    
    std::cout << "\nFeature Hashing best parameters:" << std::endl;
    std::cout << "  ngram=" << best.fh_ngram << ", log_buckets=" << best.fh_log_buckets << std::endl;
    std::cout << "  F0.5=" << best.fh_f05 
              << ", Precision=" << best.fh_precision
              << ", Recall=" << best.fh_recall
              << ", F1=" << best.fh_f1
              << ", Accuracy=" << best.fh_accuracy << std::endl;
    
    std::cout << "\nCount-Min best parameters:" << std::endl;
    std::cout << "  ngram=" << best.cm_ngram << ", num_hashes=" << best.cm_num_hashes 
              << ", log_buckets=" << best.cm_log_buckets << std::endl;
    std::cout << "  F0.5=" << best.cm_f05 
              << ", Precision=" << best.cm_precision
              << ", Recall=" << best.cm_recall
              << ", F1=" << best.cm_f1
              << ", Accuracy=" << best.cm_accuracy << std::endl;
    
    return best;
}

void experiment3_pr_curve(const std::vector<Email>& emails) {
    std::cout << "\n=== EXPERIMENT 3: PR Curve for Threshold Selection ===" << std::endl;
    
    std::ofstream pr_out("pr_curve.csv");
    pr_out << "classifier,ngram,log_buckets,num_hashes,threshold,tp,fp,tn,fn,"
           << "precision,recall,f1,f05,accuracy" << '\n';
    
    // Hyperparameter ranges to sweep
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets {10, 12, 14, 16};
    std::vector<int> cm_num_hashes {2, 3, 4};
    
    // Feature Hashing PR curves for all ngram and log_buckets combinations
    for (int ngram : ngrams) {
        for (int log_num_buckets : log_buckets) {
            std::cout << "Training Feature Hashing (ngram=" << ngram 
                      << ", log_buckets=" << log_num_buckets << ")..." << std::endl;
            
            NaiveBayesFeatureHashing clf(ngram, log_num_buckets);
            std::vector<std::pair<double, bool>> predictions;
            predictions.reserve(emails.size());
            
            for (const Email& email : emails) {
                double score = clf.predict(email);
                predictions.emplace_back(score, email.is_spam());
                clf.update(email);
            }
            
            // Sort predictions by score (descending)
            std::sort(predictions.begin(), predictions.end(), 
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Start with threshold = infinity (all predicted as negative)
            int total_positives = 0;
            int total_negatives = 0;
            for (const auto& pred : predictions) {
                if (pred.second) ++total_positives;
                else ++total_negatives;
            }
            
            int tp = 0, fp = 0;
            
            for (size_t i = 0; i < predictions.size(); ++i) {
                double threshold = predictions[i].first;
                bool actual_spam = predictions[i].second;
                
                // Update counts for this prediction being classified as positive
                if (actual_spam) ++tp;
                else ++fp;
                
                int fn = total_positives - tp;
                int tn = total_negatives - fp;
                
                // Only write when label changes or at boundaries
                if (i == predictions.size() - 1 || i == 0 || actual_spam != predictions[i+1].second) {
                    double precision = (tp + fp == 0) ? 1.0 : static_cast<double>(tp) / (tp + fp);
                    double recall = (tp + fn == 0) ? 1.0 : static_cast<double>(tp) / (tp + fn);
                    double f1 = (precision + recall == 0) ? 0.0 : 2.0 * precision * recall / (precision + recall);
                    double f05 = (0.25 * precision + recall == 0) ? 0.0 : 1.25 * precision * recall / (0.25 * precision + recall);
                    double accuracy = static_cast<double>(tp + tn) / (tp + fp + tn + fn);
                    
                    pr_out << "FeatureHashing," << ngram << "," << log_num_buckets << ",0," << threshold << ","
                           << tp << "," << fp << "," << tn << "," << fn << ","
                           << precision << "," << recall << "," << f1 << "," << f05 << "," << accuracy << '\n';
                }
            }
            
            std::cout << "  Feature Hashing PR curve (ngram=" << ngram 
                      << ", log_buckets=" << log_num_buckets << ") complete" << std::endl;
        }
    }
    
    // Count-Min PR curves for all ngram, num_hashes, and log_buckets combinations
    for (int ngram : ngrams) {
        for (int num_hashes : cm_num_hashes) {
            for (int log_num_buckets : log_buckets) {
                std::cout << "Training Count-Min (ngram=" << ngram << ", num_hashes=" << num_hashes
                          << ", log_buckets=" << log_num_buckets << ")..." << std::endl;
                
                NaiveBayesCountMin clf(ngram, num_hashes, log_num_buckets);
                std::vector<std::pair<double, bool>> predictions;
                predictions.reserve(emails.size());
                
                for (const Email& email : emails) {
                    double score = clf.predict(email);
                    predictions.emplace_back(score, email.is_spam());
                    clf.update(email);
                }
                
                // Sort predictions by score (descending)
                std::sort(predictions.begin(), predictions.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Start with threshold = infinity (all predicted as negative)
                int total_positives = 0;
                int total_negatives = 0;
                for (const auto& pred : predictions) {
                    if (pred.second) ++total_positives;
                    else ++total_negatives;
                }
                
                int tp = 0, fp = 0;
                
                for (size_t i = 0; i < predictions.size(); ++i) {
                    double threshold = predictions[i].first;
                    bool actual_spam = predictions[i].second;
                    
                    // Update counts for this prediction being classified as positive
                    if (actual_spam) ++tp;
                    else ++fp;
                    
                    int fn = total_positives - tp;
                    int tn = total_negatives - fp;
                    
                    // Only write when label changes or at boundaries
                    if (i == predictions.size() - 1 || i == 0 || actual_spam != predictions[i+1].second) {
                        double precision = (tp + fp == 0) ? 1.0 : static_cast<double>(tp) / (tp + fp);
                        double recall = (tp + fn == 0) ? 1.0 : static_cast<double>(tp) / (tp + fn);
                        double f1 = (precision + recall == 0) ? 0.0 : 2.0 * precision * recall / (precision + recall);
                        double f05 = (0.25 * precision + recall == 0) ? 0.0 : 1.25 * precision * recall / (0.25 * precision + recall);
                        double accuracy = static_cast<double>(tp + tn) / (tp + fp + tn + fn);
                        
                        pr_out << "CountMin," << ngram << "," << log_num_buckets << "," << num_hashes << "," << threshold << ","
                               << tp << "," << fp << "," << tn << "," << fn << ","
                               << precision << "," << recall << "," << f1 << "," << f05 << "," << accuracy << '\n';
                    }
                }
                
                std::cout << "  Count-Min PR curve (ngram=" << ngram << ", num_hashes=" << num_hashes
                          << ", log_buckets=" << log_num_buckets << ") complete" << std::endl;
            }
        }
    }
    
    pr_out.close();
    std::cout << "Wrote PR curve data to pr_curve.csv" << std::endl;
}

void experiment5_roc_curve(const std::vector<Email>& emails) {
    std::cout << "\n=== EXPERIMENT 5: ROC Curve for Threshold Selection ===" << std::endl;
    
    std::ofstream roc_out("roc_curve.csv");
    roc_out << "classifier,ngram,log_buckets,num_hashes,threshold,tp,fp,tn,fn,"
            << "tpr,fpr,accuracy" << '\n';
    
    // Hyperparameter ranges to sweep
    std::vector<int> ngrams {1, 2, 3};
    std::vector<int> log_buckets {10, 12, 14, 16};
    std::vector<int> cm_num_hashes {2, 3, 4};
    
    // Feature Hashing ROC curves for all ngram and log_buckets combinations
    for (int ngram : ngrams) {
        for (int log_num_buckets : log_buckets) {
            std::cout << "Training Feature Hashing (ngram=" << ngram 
                      << ", log_buckets=" << log_num_buckets << ")..." << std::endl;
            
            NaiveBayesFeatureHashing clf(ngram, log_num_buckets);
            std::vector<std::pair<double, bool>> predictions;
            predictions.reserve(emails.size());
            
            for (const Email& email : emails) {
                double score = clf.predict(email);
                predictions.emplace_back(score, email.is_spam());
                clf.update(email);
            }
            
            // Sort predictions by score (descending)
            std::sort(predictions.begin(), predictions.end(), 
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Count total positives and negatives
            int total_positives = 0;
            int total_negatives = 0;
            for (const auto& pred : predictions) {
                if (pred.second) ++total_positives;
                else ++total_negatives;
            }
            
            int tp = 0, fp = 0;
            
            for (size_t i = 0; i < predictions.size(); ++i) {
                double threshold = predictions[i].first;
                bool actual_spam = predictions[i].second;
                
                // Update counts for this prediction being classified as positive
                if (actual_spam) ++tp;
                else ++fp;
                
                int fn = total_positives - tp;
                int tn = total_negatives - fp;
                
                // Only write when label changes or at boundaries
                if (i == predictions.size() - 1 || i == 0 || actual_spam != predictions[i+1].second) {
                    double tpr = (total_positives == 0) ? 0.0 : static_cast<double>(tp) / total_positives;
                    double fpr = (total_negatives == 0) ? 0.0 : static_cast<double>(fp) / total_negatives;
                    double accuracy = static_cast<double>(tp + tn) / (tp + fp + tn + fn);
                    
                    roc_out << "FeatureHashing," << ngram << "," << log_num_buckets << ",0," << threshold << ","
                            << tp << "," << fp << "," << tn << "," << fn << ","
                            << tpr << "," << fpr << "," << accuracy << '\n';
                }
            }
            
            std::cout << "  Feature Hashing ROC curve (ngram=" << ngram 
                      << ", log_buckets=" << log_num_buckets << ") complete" << std::endl;
        }
    }
    
    // Count-Min ROC curves for all ngram, num_hashes, and log_buckets combinations
    for (int ngram : ngrams) {
        for (int num_hashes : cm_num_hashes) {
            for (int log_num_buckets : log_buckets) {
                std::cout << "Training Count-Min (ngram=" << ngram << ", num_hashes=" << num_hashes
                          << ", log_buckets=" << log_num_buckets << ")..." << std::endl;
                
                NaiveBayesCountMin clf(ngram, num_hashes, log_num_buckets);
                std::vector<std::pair<double, bool>> predictions;
                predictions.reserve(emails.size());
                
                for (const Email& email : emails) {
                    double score = clf.predict(email);
                    predictions.emplace_back(score, email.is_spam());
                    clf.update(email);
                }
                
                // Sort predictions by score (descending)
                std::sort(predictions.begin(), predictions.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Count total positives and negatives
                int total_positives = 0;
                int total_negatives = 0;
                for (const auto& pred : predictions) {
                    if (pred.second) ++total_positives;
                    else ++total_negatives;
                }
                
                int tp = 0, fp = 0;
                
                for (size_t i = 0; i < predictions.size(); ++i) {
                    double threshold = predictions[i].first;
                    bool actual_spam = predictions[i].second;
                    
                    // Update counts for this prediction being classified as positive
                    if (actual_spam) ++tp;
                    else ++fp;
                    
                    int fn = total_positives - tp;
                    int tn = total_negatives - fp;
                    
                    // Only write when label changes or at boundaries
                    if (i == predictions.size() - 1 || i == 0 || actual_spam != predictions[i+1].second) {
                        double tpr = (total_positives == 0) ? 0.0 : static_cast<double>(tp) / total_positives;
                        double fpr = (total_negatives == 0) ? 0.0 : static_cast<double>(fp) / total_negatives;
                        double accuracy = static_cast<double>(tp + tn) / (tp + fp + tn + fn);
                        
                        roc_out << "CountMin," << ngram << "," << log_num_buckets << "," << num_hashes << "," << threshold << ","
                                << tp << "," << fp << "," << tn << "," << fn << ","
                                << tpr << "," << fpr << "," << accuracy << '\n';
                    }
                }
                
                std::cout << "  Count-Min ROC curve (ngram=" << ngram << ", num_hashes=" << num_hashes
                          << ", log_buckets=" << log_num_buckets << ") complete" << std::endl;
            }
        }
    }
    
    roc_out.close();
    std::cout << "Wrote ROC curve data to roc_curve.csv" << std::endl;
}

void experiment4_computational_efficiency(const std::vector<Email>& emails) {
    std::cout << "\n=== EXPERIMENT 4: Computational Efficiency ===" << std::endl;
    
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

    // Run all experiments
    experiment1_learning_curves(emails);
    BestHyperparameters best_params = experiment2_hyperparameter_grid_search(emails);
    experiment3_pr_curve(emails);
    experiment4_computational_efficiency(emails);
    experiment5_roc_curve(emails);
    
    std::cout << "\n=== ALL EXPERIMENTS COMPLETE ===" << std::endl;

    return 0;
}