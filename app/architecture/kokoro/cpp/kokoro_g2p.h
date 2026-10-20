// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

// English text -> Kokoro phonemes. A port of misaki's English G2P (hexgrad/misaki, en.py) for the parts that
// don't need its spaCy tagger or espeak-ng fallback: gold then silver lexicon lookup, -s / -ed / -ing
// stemming, numbers read out, capitalisation stress, and the next word deciding "the" / "to" / "a". Words the
// lexicon can't place are spelled out letter by letter.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace neko {

class KokoroG2P {
public:
    // gold / silver: misaki's us_ or gb_ lexicon JSON files.
    KokoroG2P(const std::string& goldPath, const std::string& silverPath, bool british);

    std::string phonemize(const std::string& text) const;

private:
    struct Token {
        std::u32string text;
        std::u32string phonemes;
        bool word = false;         // looked up in the lexicon (vs punctuation or a pre-resolved number)
        bool spaceAfter = false;
    };

    std::vector<Token> tokenize(const std::u32string& text) const;
    std::u32string wordPhonemes(const std::u32string& word, bool futureVowelKnown, bool futureVowel,
                                bool nextIsWord) const;
    std::u32string numberPhonemes(const std::u32string& number, const std::u32string& currency) const;
    bool lookup(const std::u32string& word, std::u32string& out) const;  // gold, then silver
    bool isKnown(const std::u32string& word) const;
    bool getWord(std::u32string word, double stress, bool hasStress, std::u32string& out) const;
    bool stemS(const std::u32string& word, std::u32string& out) const;
    bool stemEd(const std::u32string& word, std::u32string& out) const;
    bool stemIng(const std::u32string& word, std::u32string& out) const;
    std::u32string spell(const std::u32string& word) const;
    std::u32string words(const std::string& english) const;  // phonemes of plain lowercase words

    std::unordered_map<std::u32string, std::u32string> gold_, silver_;
    bool british_;
};

std::u32string utf8To32(const std::string& s);
std::string utf32To8(const std::u32string& s);

}  // namespace neko
