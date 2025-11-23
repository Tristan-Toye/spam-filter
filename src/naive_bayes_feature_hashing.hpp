#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
private:
    // Golden ratio constant for well-distributed hash seeds (64-bit)
    static constexpr size_t GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15ULL;
    
    // Parameters
    int seed_;
    int ngram_;
    int log_num_buckets_;
    size_t num_buckets_;
    
    // Feature hash tables for spam and ham
    std::vector<int> spam_counts_;
    std::vector<int> ham_counts_;
    
    // Class statistics
    size_t spam_tokens_ = 0;
    size_t ham_tokens_ = 0;
    int spam_emails_ = 0;
    int ham_emails_ = 0;

public:
    /** Do not change the signature of the constructor! */
    NaiveBayesFeatureHashing(int ngram, int log_num_buckets)
        : BaseClf(0.5)  // threshold for P(spam|email) > 0.5
        , seed_(0xfa4f8cc)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , num_buckets_(static_cast<size_t>(1ULL << log_num_buckets))  // 2^log_num_buckets
        , spam_counts_(num_buckets_, 0)  // feature hash table for spam
        , ham_counts_(num_buckets_, 0)   // feature hash table for ham
    {
    }

    void update_(const Email &email) {
        bool is_spam = email.is_spam();
        
        // Iterate through n-grams and update the appropriate hash table
        EmailIter iter(email, ngram_);
        if (is_spam) {
            ++spam_emails_;
            while (iter) {
                std::string_view ngram = iter.next();
                update_bucket(spam_counts_, ngram);
                ++spam_tokens_;
            }
        } else {
            ++ham_emails_;
            while (iter) {
                std::string_view ngram = iter.next();
                update_bucket(ham_counts_, ngram);
                ++ham_tokens_;
            }
        }
    }

    double predict_(const Email& email) const {
        // Handle edge case: no training data
        if (spam_emails_ == 0 && ham_emails_ == 0) {
            return 0.5;
        }
        
        int total_emails = spam_emails_ + ham_emails_;
        
        //laplace estimate, adding +2 on total emails to compensate for adding 1 email to both spam and ham 
        double log_prior_spam = std::log(static_cast<double>(spam_emails_ + 1) / (total_emails + 2)); 
        double log_prior_ham = std::log(static_cast<double>(ham_emails_ + 1) / (total_emails + 2));
        
        
        double log_likelihood_spam = 0.0;
        double log_likelihood_ham = 0.0;
        
        
        EmailIter iter(email, ngram_);
        while (iter) {
            std::string_view ngram = iter.next();
            
            // Query counts from feature hash tables
            int spam_count = get_count(spam_counts_, ngram);
            int ham_count = get_count(ham_counts_, ngram);
            
            // laplace estimate, adding num_buckets to compensate for adding each word to both ham and spam
            double prob_ngram_given_spam = static_cast<double>(spam_count + 1) / 
                                          (spam_tokens_ + num_buckets_);
            double prob_ngram_given_ham = static_cast<double>(ham_count + 1) / 
                                         (ham_tokens_ + num_buckets_);
            
            log_likelihood_spam += std::log(prob_ngram_given_spam);
            log_likelihood_ham += std::log(prob_ngram_given_ham);
        }
        
        // Compute log posterior probabilities
        double log_prob_spam = log_prior_spam + log_likelihood_spam ;
        double log_prob_ham = log_prior_ham + log_likelihood_ham ;
        
        //softmax normalization to get probability
        double log_ham_to_spam_ratio = log_prob_ham - log_prob_spam;
        log_ham_to_spam_ratio = clamp_for_overflow(log_ham_to_spam_ratio);
        return 1.0 / (1.0 + std::exp(log_ham_to_spam_ratio));
    }

private:
    size_t get_bucket(std::string_view ngram) const { return get_bucket(hash(ngram, seed_)); }
    size_t get_bucket(size_t hash) const { return efficient_2power_modulo(hash, num_buckets_);}
    
    int get_count(const std::vector<int>& counts, std::string_view ngram) const {
        size_t bucket = get_bucket(ngram);
        return counts[bucket];
    }
    
    // Update feature hash table: increment the hash position
    void update_bucket(std::vector<int>& counts, std::string_view ngram) {
        size_t bucket = get_bucket(ngram);
        ++counts[bucket];
    }

    size_t efficient_2power_modulo(size_t x, size_t m) const {
        return x & (m - 1ULL);
    }
    
    double clamp_for_overflow(double value) const {return -50.0 > value ? -50.0 : (50.0 < value ? 50.0 : value); }
};

} // namespace bdap
