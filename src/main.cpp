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
    load_emails(emails, "data/Enron.txt");
    load_emails(emails, "data/SpamAssasin.txt");
    load_emails(emails, "data/Trec2005.txt");
    load_emails(emails, "data/Trec2006.txt");
    load_emails(emails, "data/Trec2007.txt");

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

int main(int argc, char *argv[]) { 
    // The arguments can be used for your experiments, specify the correct command(s) in your script.sh

    // Example on how to load the data + statistics on spam/ham
    int seed = 12;
    std::vector<Email> emails = load_emails(seed);
    std::cout << "#emails: " << emails.size() << std::endl;
    size_t num_spam = 0;
    for (const Email& e : emails)
        num_spam += e.is_spam();
    std::cout << "#spam: " << num_spam << ", "
              << (100.0 * num_spam / emails.size()) << "%"
              << std::endl;

    // TODO: implement your experiments, write the results to one or more file(s)

    // // Example use:
    // // Given an instantiated classifier clf, you can evaluate a single email as follows:

    // Email email1("EMAIL> label=1", "free try now lot money king rich");
    
    // auto classify = [&clf](const Email& m, const char *name) {
    //     std::cout << "classify(" << name << "): soft label="
    //         << clf.predict(m)
    //         << ", hard label="
    //         << (clf.classify(m) ? "spam" : "ham")
    //         << std::endl;
    // };

    // classify(email1, "email1");

    return 0;
}