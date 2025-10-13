#pragma once

/*
 * Copyright 2025 BDAP team.
 *
 */

#include <unordered_map> // std::hash for std::string_view
#include "email.hpp"
#include "murmurhash.hpp"

namespace bdap {

/**
 * A base class for your classifiers.
 * Your implementations should extend this class as follows:
 *
 * ```
 *     class YourClf : public BaseClf<YourClf> {
 *         ...
 *     };
 * ```
 * Your class will fail to compile if it does not implement the methods
 *  - `update(const Email&)`
 *  - `predict(const Email&) const`
 *
 * You must follow this structure for ease of grading.
 *
 * This design pattern is called the 'curiously recurring template pattern'.
 *
 * You should not have to change this class. You do not have to submit this
 * class. If you find issues, contact your TA.
 */
template <typename Derived>
class BaseClf {
public:
    // Statistics
    int num_examples_processed = 0;

private:
    // Classifiers parameters
    double threshold_ = 0.0;

protected:
    // Make sure that the subclass sets an appropriate threshold
    // to make a hard classification (see `predict` and `classify`).
    BaseClf(double threshold) : threshold_(threshold) {}

public:
    /** Update the paramters of the model using the incoming email (online
     * learning). */
    void update(const Email& email) {
        ++num_examples_processed;
        static_cast<Derived *>(this)->update_(email);
    }

    /** Use the current model to make a prediction about the given email. */
    double predict(const Email& email) const {
        return static_cast<const Derived *>(this)->predict_(email);
    }

    /** Threshold the prediction given by `predict` by `threshold` to get a
     * concrete classification. */
    bool classify(const Email& email) const
    { return classify(predict(email)); }

    bool classify(double pr) const
    { return pr > threshold_; }

    /* UTILITY FUNCTIONS */

    static size_t hash(std::string_view key, size_t seed) {
        uint64_t out[2] = {0};
        MurmurHash3_x64_128(key.data(), key.size(), seed, &out);
        return out[0] ^ out[1];
    }

protected:
    /* Implement this method in your subclasses */
    void update_(const Email& email);

    /* Implement this method in your subclasses */
    double predict_(const Email& email) const;
};

} // namespace bdap
