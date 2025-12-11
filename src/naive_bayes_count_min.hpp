#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {
private:
    // Decrease the collision rate by using a well-distributed hash functions: https://github.com/HowardHinnant/hash_append/issues/7
    static constexpr size_t GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15;
    
    // Parameters
    int ngram_;
    int num_hashes_;
    size_t num_buckets_;
    
    // Count-Min Sketches for spam and ham
    std::vector<std::vector<int>> spam_sketch_;
    std::vector<std::vector<int>> ham_sketch_;
    
    // Class statistics
    int num_spam_;
    int num_ham_;
    int total_spam_ngrams_;
    int total_ham_ngrams_;

public:
    NaiveBayesCountMin(int ngram, int num_hashes, int log_num_buckets, double threshold = 0.5)
        : BaseClf(threshold)  // threshold for P(spam|email) > threshold
        , ngram_(ngram)
        , num_hashes_(num_hashes)
        , num_buckets_(static_cast<size_t>(1ULL << log_num_buckets))  // 2^log_num_buckets
        , spam_sketch_(num_hashes, std::vector<int>(num_buckets_, 0)) //count-min sketch for spam
        , ham_sketch_(num_hashes, std::vector<int>(num_buckets_, 0)) //count-min sketch for ham
        , num_spam_(0)
        , num_ham_(0)
        , total_spam_ngrams_(0)
        , total_ham_ngrams_(0)
    {
    }

    void update_(const Email &email) {
        bool is_spam = email.is_spam();
        
        // Iterate through n-grams and update the appropriate sketch
        EmailIter iter(email, ngram_);
        if (is_spam) {
            ++num_spam_;
            while (iter) {
                std::string_view ngram = iter.next();
                update_bucket(spam_sketch_, ngram);
                ++total_spam_ngrams_;
            }
        } else {
            ++num_ham_;
            while (iter) {
                std::string_view ngram = iter.next();
                update_bucket(ham_sketch_, ngram);
                ++total_ham_ngrams_;
            }
        }
    }

    double predict_(const Email& email) const {
        // Handle edge case: no training data
        if (num_spam_ == 0 && num_ham_ == 0) {
            return 0.5;
        }
        
        int total_emails = num_spam_ + num_ham_;
        
        //laplace estimate, adding +2 on total emails to compensate for adding 1 email to both spam and ham 
        double log_prior_spam = std::log(static_cast<double>(num_spam_ + 1) / (total_emails + 2)); 
        double log_prior_ham = std::log(static_cast<double>(num_ham_ + 1) / (total_emails + 2));
        
        
        double log_likelihood_spam = 0.0;
        double log_likelihood_ham = 0.0;
        
        
        EmailIter iter(email, ngram_);
        while (iter) {
            std::string_view ngram = iter.next();
            
            // Query counts from Count-Min sketches
            int spam_count = get_count(spam_sketch_, ngram);
            int ham_count = get_count(ham_sketch_, ngram);
            

            
            // laplace estimate, adding num_buckets to compensate for adding each word to both ham and spam
            double prob_ngram_given_spam = static_cast<double>(spam_count + 1) / 
                                          (total_spam_ngrams_ + num_buckets_);
            double prob_ngram_given_ham = static_cast<double>(ham_count + 1) / 
                                         (total_ham_ngrams_ + num_buckets_);
            
            log_likelihood_spam += std::log(prob_ngram_given_spam);
            log_likelihood_ham += std::log(prob_ngram_given_ham);
        }
        
        // Compute log posterior probabilities
        double log_prob_spam = log_prior_spam + log_likelihood_spam;
        double log_prob_ham = log_prior_ham + log_likelihood_ham;
        
        //softmax normalization to get probability
        double log_ham_to_spam_ratio = log_prob_ham - log_prob_spam;
        log_ham_to_spam_ratio = clamp_for_overflow(log_ham_to_spam_ratio);
        return 1.0 / (1.0 + std::exp(log_ham_to_spam_ratio));
    }

    
private:

    size_t get_bucket(std::string_view ngram, size_t i ) const { return get_bucket(hash(ngram, i * GOLDEN_RATIO_64)); }
    size_t get_bucket(size_t hash) const { return efficient_2power_modulo(hash, num_buckets_); }

    // Query Count-Min sketch: return minimum count across all hash functions
    int get_count(const std::vector<std::vector<int>>& sketch, std::string_view ngram) const {
        int min_count = std::numeric_limits<int>::max();
        
        for (int i = 0; i < num_hashes_; ++i) {
            // Use golden ratio to ensure well-distributed seeds
            size_t bucket = get_bucket(ngram, i);
            min_count = std::min(min_count, sketch[i][bucket]);
        }
        
        return min_count;
    }
    
    // Update Count-Min sketch: increment all hash positions
    void update_bucket(std::vector<std::vector<int>>& sketch, std::string_view ngram) {
        for (int i = 0; i < num_hashes_; ++i) {
            // Use golden ratio to ensure well-distributed seeds
            size_t bucket = get_bucket(ngram, i);
            ++sketch[i][bucket];
        }
    }

    size_t efficient_2power_modulo(size_t x, size_t m) const {
        return x & (m - 1ULL);
    }

    double clamp_for_overflow(double value) const { return -50.0 > value ? -50.0 : (50.0 < value ? 50.0 : value); }
};

} // namespace bdap
