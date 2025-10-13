#pragma once

/*
 * Copyright 2025 BDAP team.
 *
 */

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bdap {

class Email {
    std::string header_;
    std::string body_;
    std::vector<size_t> words_; // offsets into `body_`

public:
    Email(const std::string& header, const std::string& body)
        : header_(header)
        , body_(body)
        , words_{}
    {
        // find start indices of words in body
        size_t prev = 0;
        for (size_t i = 0; i < body_.size(); ++i) {
            char c = body_[i];
            if (c == ' ' || c == '\n') {
                words_.push_back(prev);
                prev = i+1;
            }
        }
        if (prev != body_.size()) // omit if last char is space
            words_.push_back(prev);
        words_.push_back(body_.size());
    }

    // careful with return string_view: 
    // https://stackoverflow.com/questions/46032307/how-to-efficiently-get-a-string-view-for-a-substring-of-stdstring
    std::string_view get_ngram(size_t i, size_t k) const {
        // range check
        if (i+k >= words_.size())
            throw std::range_error("ngram out of bounds");

        size_t index0 = words_[i];
        size_t index1 = words_[i+k]-1;

        std::string_view body_view = body_;
        return body_view.substr(index0, index1-index0);
    }

    std::string_view get_word(size_t i) const { return get_ngram(i, 1); }
    size_t num_words() const { return words_.size()-1; }

    const std::string& body() const { return body_; }
    const std::string& header() const { return header_; }

    /** The true label of the email. Do not use this in predict! */
    bool is_spam() const { return header_[13] == '1'; /* EMAIL> label=X */ }
};

class EmailIter {
    int ngram_;
    const Email& email_;

    // Iterator state
    int k_;
    int i_;

public:
  EmailIter(const Email &email, int ngram_)
      : ngram_(ngram_), email_(email), k_{1}, i_{0}
  {
      int num_words = static_cast<int>(email.num_words());
      if (num_words < ngram_)
          ngram_ = num_words;
  }

  operator bool() const
  { return !is_done(); }

  bool is_done() const
  { return k_ > email_.num_words() || k_ > ngram_; }

  std::string_view next() {
      auto out = email_.get_ngram(i_, k_);
      ++i_;
      if (i_ + k_ - 1 == email_.num_words()) {
          i_ = 0;
          ++k_;
      }
      return out;
  }

  size_t size() const {
      size_t s = 0;
      for (int k = 0; k < ngram_; ++k)
          s += email_.num_words() - k;
      return s;
  }
};

static void read_emails(std::ifstream& f, std::vector<Email>& emails) {
    std::stringstream wordsbuf;
    std::string line;
    std::string header;
    int label = -1;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;

        //std::cout << "> '" << l << "'\n";
        if (line.empty() && !header.empty()) // empty newline indicating the end of an email
        {
            //if (header.empty())
            //{
            //    std::cerr << "line number " << lineno << std::endl;
            //    throw std::runtime_error("invalid state");
            //}
            auto body = wordsbuf.str();
            wordsbuf.clear();
            wordsbuf.str("");
            emails.emplace_back(header, body);
            header.clear();
        }
        // header starting with `EMAIL> ` with path to email file
        else if (line.find("EMAIL> ", 0) == 0)
            std::swap(header, line);
        else
            wordsbuf << line; // concat line to email
    }
}


} // namespace bdap
