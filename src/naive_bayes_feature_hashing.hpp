#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int seed_;
    int ngram_;
    int log_num_buckets_;
    std::vector<int> counts_;

public:
    /** Do not change the signature of the constructor! */
    NaiveBayesFeatureHashing(int ngram, int log_num_buckets)
        : BaseClf(0.0 /* set appropriate threshold */)
        , seed_(0xfa4f8cc)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
    {}

    void update_(const Email &email) {
        // TODO implement this
    }

    double predict_(const Email& email) const {
        // TODO implement this
        return 0.0;
    }

private:
    size_t get_bucket(std::string_view ngram, int is_spam) const {
        return get_bucket(hash(ngram, seed_), is_spam);
    }

    size_t get_bucket(size_t hash, int is_spam) const {
        // TODO limit the range of the hash values here
        return hash;
    }
};

} // namespace bdap
