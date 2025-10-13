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

public:
    NaiveBayesCountMin(int ngram, int num_hashes, int log_num_buckets)
        : BaseClf(0.0 /* set appropriate threshold */)
    {
    }

    void update_(const Email &email) {
        // TODO implement this
    }

    double predict_(const Email& email) const {
        // TODO implement this
        return 0.0;
    }

};

} // namespace bdap
