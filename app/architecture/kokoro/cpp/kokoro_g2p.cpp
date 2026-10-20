// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

#include "kokoro_g2p.h"

#include <algorithm>
#include <cmath>

#include "common.h"
#include "json.h"

namespace neko {

std::u32string utf8To32(const std::string& s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        char32_t cp;
        int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { i++; continue; }
        if (i + size_t(n) > s.size()) break;
        for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + size_t(k)] & 0x3F);
        out.push_back(cp);
        i += size_t(n);
    }
    return out;
}

std::string utf32To8(const std::u32string& s) {
    std::string out;
    for (char32_t c : s) {
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | (c >> 6)); out += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += char(0xE0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
        else { out += char(0xF0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 0x3F)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    }
    return out;
}

namespace {

const char32_t kPrimary = U'ˈ', kSecondary = U'ˌ';
const std::u32string kVowels = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const std::u32string kConsonants = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const std::u32string kPuncts = U";:,.!?—…\"“”";
const std::u32string kNonQuotePuncts = U";:,.!?—…";
const std::u32string kUsTaus = U"AIOWYiuæɑəɛɪɹʊʌ";

bool in(const std::u32string& set, char32_t c) { return set.find(c) != std::u32string::npos; }
bool isAsciiAlpha(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'); }
bool isDigit(char32_t c) { return c >= U'0' && c <= U'9'; }
char32_t lower(char32_t c) { return (c >= U'A' && c <= U'Z') ? c + 32 : c; }
char32_t upper(char32_t c) { return (c >= U'a' && c <= U'z') ? c - 32 : c; }
std::u32string lowerS(std::u32string s) { for (auto& c : s) c = lower(c); return s; }
std::u32string upperS(std::u32string s) { for (auto& c : s) c = upper(c); return s; }
std::u32string capitalize(std::u32string s) {
    s = lowerS(s);
    if (!s.empty()) s[0] = upper(s[0]);
    return s;
}
bool hasVowel(const std::u32string& ps) { return std::any_of(ps.begin(), ps.end(), [](char32_t c) { return in(kVowels, c); }); }
bool hasStressMark(const std::u32string& ps) { return ps.find(kPrimary) != ps.npos || ps.find(kSecondary) != ps.npos; }
bool endsWith(const std::u32string& s, const std::u32string& t) {
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

// Folds common Latin accents so "café" can be looked up as "cafe".
char32_t fold(char32_t c) {
    static const std::u32string from = U"àáâãäåçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ";
    static const std::u32string to = U"aaaaaaceeeeiiiinooooouuuuyyAAAAAACEEEEIIIINOOOOOUUUUY";
    size_t i = from.find(c);
    return i == from.npos ? c : to[i];
}

// Moves each stress mark in front of the vowel it belongs to (misaki's restress).
std::u32string restress(const std::u32string& ps) {
    std::vector<std::pair<double, char32_t>> ips;
    for (size_t i = 0; i < ps.size(); i++) ips.emplace_back(double(i), ps[i]);
    for (size_t i = 0; i < ps.size(); i++) {
        if (ps[i] != kPrimary && ps[i] != kSecondary) continue;
        for (size_t j = i; j < ps.size(); j++)
            if (in(kVowels, ps[j])) { ips[i].first = double(j) - 0.5; break; }
    }
    std::stable_sort(ips.begin(), ips.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::u32string out;
    for (auto& p : ips) out += p.second;
    return out;
}

std::u32string applyStress(std::u32string ps, bool has, double stress) {
    auto replaceAll = [](std::u32string s, char32_t a, const std::u32string& b) {
        std::u32string o;
        for (char32_t c : s) { if (c == a) o += b; else o += c; }
        return o;
    };
    if (!has) return ps;
    if (stress < -1) return replaceAll(replaceAll(ps, kPrimary, U""), kSecondary, U"");
    if (stress == -1 || ((stress == 0 || stress == -0.5) && ps.find(kPrimary) != ps.npos))
        return replaceAll(replaceAll(ps, kSecondary, U""), kPrimary, std::u32string(1, kSecondary));
    if ((stress == 0 || stress == 0.5 || stress == 1) && !hasStressMark(ps)) {
        if (!hasVowel(ps)) return ps;
        return restress(std::u32string(1, kSecondary) + ps);
    }
    if (stress >= 1 && ps.find(kPrimary) == ps.npos && ps.find(kSecondary) != ps.npos)
        return replaceAll(ps, kSecondary, std::u32string(1, kPrimary));
    if (stress > 1 && !hasStressMark(ps)) {
        if (!hasVowel(ps)) return ps;
        return restress(std::u32string(1, kPrimary) + ps);
    }
    return ps;
}

// ---------------------------------------------------------------- numbers as English words (num2words style,
// without "and": misaki drops it)

const char* kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                       "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                       "nineteen"};
const char* kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

std::string below1000(int n) {
    std::string s;
    if (n >= 100) {
        s = std::string(kOnes[n / 100]) + " hundred";
        n %= 100;
        if (n) s += " ";
    }
    if (n >= 20) {
        s += kTens[n / 10];
        if (n % 10) s += std::string(" ") + kOnes[n % 10];
    } else if (n > 0 || s.empty()) {
        s += kOnes[n];
    }
    return s;
}

std::string cardinal(long long n) {
    if (n < 0) return "minus " + cardinal(-n);
    if (n < 1000) return below1000(int(n));
    static const std::pair<long long, const char*> scales[] = {
        {1000000000000LL, "trillion"}, {1000000000LL, "billion"}, {1000000LL, "million"}, {1000LL, "thousand"}};
    std::string s;
    for (auto& sc : scales) {
        if (n >= sc.first) {
            s += (s.empty() ? "" : " ") + cardinal(n / sc.first) + " " + sc.second;
            n %= sc.first;
        }
    }
    if (n) s += " " + below1000(int(n));
    return s;
}

std::string ordinal(long long n) {
    std::string c = cardinal(n);
    size_t sp = c.find_last_of(' ');
    std::string head = sp == std::string::npos ? "" : c.substr(0, sp + 1), last = c.substr(sp == std::string::npos ? 0 : sp + 1);
    static const std::pair<const char*, const char*> irregular[] = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"}, {"eight", "eighth"},
        {"nine", "ninth"}, {"twelve", "twelfth"}};
    for (auto& ir : irregular)
        if (last == ir.first) return head + ir.second;
    if (!last.empty() && last.back() == 'y') return head + last.substr(0, last.size() - 1) + "ieth";
    return head + last + "th";
}

std::string year(int y) {
    if (y % 1000 == 0 || (y >= 2000 && y < 2010)) return cardinal(y);
    int hi = y / 100, lo = y % 100;
    if (lo == 0) return below1000(hi) + " hundred";
    return below1000(hi) + (lo < 10 ? " oh " : " ") + below1000(lo);
}

}  // namespace

// ---------------------------------------------------------------- lexicon

KokoroG2P::KokoroG2P(const std::string& goldPath, const std::string& silverPath, bool british) : british_(british) {
    auto load = [](const std::string& path, std::unordered_map<std::u32string, std::u32string>& dict) {
        Json j = Json::parseFile(path);
        dict.reserve(j.size() * 2);
        for (auto& kv : j.members()) {
            // Entries with part-of-speech variants use their DEFAULT reading (no tagger here).
            const Json& v = kv.second.isObject() ? kv.second["DEFAULT"] : kv.second;
            if (!v.isString()) continue;
            dict[utf8To32(kv.first)] = utf8To32(v.str());
        }
        // misaki's grow_dictionary: "word" also answers "Word", and "Word" answers "word".
        std::vector<std::pair<std::u32string, std::u32string>> extra;
        for (auto& kv : dict) {
            const std::u32string& k = kv.first;
            if (k.size() < 2) continue;
            if (k == lowerS(k)) {
                if (k != capitalize(k)) extra.emplace_back(capitalize(k), kv.second);
            } else if (k == capitalize(k)) {
                extra.emplace_back(lowerS(k), kv.second);
            }
        }
        for (auto& e : extra) dict.emplace(e.first, e.second);  // existing keys win
    };
    load(goldPath, gold_);
    load(silverPath, silver_);
    if (gold_.empty()) fail("empty lexicon: " + goldPath);
}

bool KokoroG2P::lookup(const std::u32string& w, std::u32string& out) const {
    std::u32string word = w;
    if (word == upperS(word) && !gold_.count(word)) word = lowerS(word);
    auto g = gold_.find(word);
    if (g != gold_.end()) { out = g->second; return true; }
    auto s = silver_.find(word);
    if (s != silver_.end()) { out = s->second; return true; }
    return false;
}

bool KokoroG2P::isKnown(const std::u32string& w) const {
    if (gold_.count(w) || silver_.count(w)) return true;
    if (!std::all_of(w.begin(), w.end(), [](char32_t c) { return isAsciiAlpha(c) || c == U'\''; })) return false;
    if (w.size() == 1) return true;
    if (w == upperS(w) && gold_.count(lowerS(w))) return true;
    return false;
}

std::u32string KokoroG2P::spell(const std::u32string& word) const {
    // misaki's get_NNP: each letter's name, one primary stress on the last letter.
    std::u32string ps;
    for (char32_t c : word) {
        if (!isAsciiAlpha(c)) continue;
        auto it = gold_.find(std::u32string(1, upper(c)));
        if (it == gold_.end()) continue;
        ps += it->second;
    }
    ps = applyStress(ps, true, 0);
    size_t pos = ps.rfind(kSecondary);
    if (pos != ps.npos) ps[pos] = kPrimary;
    return ps;
}

bool KokoroG2P::stemS(const std::u32string& w, std::u32string& out) const {
    if (w.size() < 3 || w.back() != U's') return false;
    std::u32string stem;
    if (!endsWith(w, U"ss") && isKnown(w.substr(0, w.size() - 1))) stem = w.substr(0, w.size() - 1);
    else if ((endsWith(w, U"'s") || (w.size() > 4 && endsWith(w, U"es") && !endsWith(w, U"ies"))) &&
             isKnown(w.substr(0, w.size() - 2)))
        stem = w.substr(0, w.size() - 2);
    else if (w.size() > 4 && endsWith(w, U"ies") && isKnown(w.substr(0, w.size() - 3) + U"y"))
        stem = w.substr(0, w.size() - 3) + U"y";
    else return false;
    std::u32string ps;
    if (!lookup(stem, ps) || ps.empty()) return false;
    const char32_t last = ps.back();
    if (in(U"ptkfθ", last)) out = ps + U"s";
    else if (in(U"szʃʒʧʤ", last)) out = ps + (british_ ? U"ɪz" : U"ᵻz");
    else out = ps + U"z";
    return true;
}

bool KokoroG2P::stemEd(const std::u32string& w, std::u32string& out) const {
    if (w.size() < 4 || w.back() != U'd') return false;
    std::u32string stem;
    if (!endsWith(w, U"dd") && isKnown(w.substr(0, w.size() - 1))) stem = w.substr(0, w.size() - 1);
    else if (w.size() > 4 && endsWith(w, U"ed") && !endsWith(w, U"eed") && isKnown(w.substr(0, w.size() - 2)))
        stem = w.substr(0, w.size() - 2);
    else return false;
    std::u32string ps;
    if (!lookup(stem, ps) || ps.empty()) return false;
    const char32_t last = ps.back();
    if (in(U"pkfθʃsʧ", last)) out = ps + U"t";
    else if (last == U'd') out = ps + (british_ ? U"ɪd" : U"ᵻd");
    else if (last != U't') out = ps + U"d";
    else if (british_ || ps.size() < 2) out = ps + U"ɪd";
    else if (in(kUsTaus, ps[ps.size() - 2])) out = ps.substr(0, ps.size() - 1) + U"ɾᵻd";
    else out = ps + U"ᵻd";
    return true;
}

bool KokoroG2P::stemIng(const std::u32string& w, std::u32string& out) const {
    if (w.size() < 5 || !endsWith(w, U"ing")) return false;
    std::u32string stem, base = w.substr(0, w.size() - 3);
    if (w.size() > 5 && isKnown(base)) stem = base;
    else if (isKnown(base + U"e")) stem = base + U"e";
    else if (w.size() > 5 && base.size() >= 2 && base[base.size() - 1] == base[base.size() - 2] &&
             in(U"bcdgklmnprstvxz", base.back()) && isKnown(base.substr(0, base.size() - 1)))
        stem = base.substr(0, base.size() - 1);
    else return false;
    std::u32string ps;
    if (!lookup(stem, ps) || ps.empty()) return false;
    if (british_ && in(U"əː", ps.back())) return false;
    if (!british_ && ps.size() > 1 && ps.back() == U't' && in(kUsTaus, ps[ps.size() - 2]))
        out = ps.substr(0, ps.size() - 1) + U"ɾɪŋ";
    else out = ps + U"ɪŋ";
    return true;
}

bool KokoroG2P::getWord(std::u32string word, double stress, bool hasStress, std::u32string& out) const {
    const std::u32string wl = lowerS(word);
    // A capitalised or shouted word that the lexicon only knows in lowercase.
    if (word.size() > 1 && word != wl && !gold_.count(word) && !silver_.count(word) &&
        (word == upperS(word) || word.substr(1) == lowerS(word.substr(1)))) {
        std::u32string tmp;
        if (gold_.count(wl) || silver_.count(wl) || stemS(wl, tmp) || stemEd(wl, tmp) || stemIng(wl, tmp)) word = wl;
    }
    std::u32string ps;
    if (isKnown(word) && lookup(word, ps)) { out = applyStress(ps, hasStress, stress); return true; }
    if (endsWith(word, U"s'") && isKnown(word.substr(0, word.size() - 2) + U"'s") &&
        lookup(word.substr(0, word.size() - 2) + U"'s", ps)) { out = applyStress(ps, hasStress, stress); return true; }
    if (endsWith(word, U"'") && isKnown(word.substr(0, word.size() - 1)) && lookup(word.substr(0, word.size() - 1), ps)) {
        out = applyStress(ps, hasStress, stress);
        return true;
    }
    if (stemS(word, ps) || stemEd(word, ps)) { out = applyStress(ps, hasStress, stress); return true; }
    if (stemIng(word, ps)) { out = applyStress(ps, true, hasStress ? stress : 0.5); return true; }
    return false;
}

std::u32string KokoroG2P::words(const std::string& english) const {
    std::u32string out;
    size_t start = 0;
    const std::u32string s = utf8To32(english);
    while (start < s.size()) {
        size_t end = s.find_first_of(U" -", start);
        if (end == s.npos) end = s.size();
        std::u32string w = s.substr(start, end - start), ps;
        if (!w.empty()) {
            if (!out.empty()) out += U' ';
            if (lookup(w, ps)) out += ps;
            else out += spell(w);
        }
        start = end + 1;
    }
    return out;
}

std::u32string KokoroG2P::numberPhonemes(const std::u32string& raw, const std::u32string& currency) const {
    std::string n;
    std::string suffix;
    for (char32_t c : raw) {
        if (isDigit(c) || c == U'.' || c == U'-') n += char(c);
        else if (c == U',') continue;
        else suffix += char(lower(c));
    }
    bool negative = !n.empty() && n[0] == '-';
    if (negative) n = n.substr(1);
    std::string text;
    const size_t dot = n.find('.');
    auto intOf = [](const std::string& s) { return s.empty() ? 0LL : std::stoll(s); };
    if (!currency.empty() && (currency == U"$" || currency == U"£" || currency == U"€")) {
        const char* unit = currency == U"$" ? "dollar" : currency == U"£" ? "pound" : "euro";
        const char* sub = currency == U"£" ? "pence" : "cent";
        long long whole = intOf(n.substr(0, dot)), cents = dot == std::string::npos ? 0 : intOf((n.substr(dot + 1) + "00").substr(0, 2));
        if (whole || !cents) text = cardinal(whole) + " " + unit + (whole == 1 ? "" : "s");
        if (cents) text += std::string(text.empty() ? "" : " and ") + cardinal(cents) + " " + sub + (cents == 1 || currency == U"£" ? "" : "s");
    } else if (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") {
        text = ordinal(intOf(n));
    } else if (dot != std::string::npos) {
        text = cardinal(intOf(n.substr(0, dot))) + " point";
        for (char d : n.substr(dot + 1)) text += std::string(" ") + kOnes[d - '0'];
    } else if (n.size() == 4 && raw.find(U',') == raw.npos && n[0] != '0' && suffix.empty()) {
        text = year(int(intOf(n)));
    } else if (n.size() > 1 && n[0] == '0') {
        for (char d : n) text += std::string(text.empty() ? "" : " ") + (d == '0' ? "oh" : kOnes[d - '0']);
    } else {
        text = n.size() > 15 ? "" : cardinal(intOf(n));
        if (text.empty()) for (char d : n) text += std::string(text.empty() ? "" : " ") + kOnes[d - '0'];
    }
    if (negative) text = "minus " + text;
    std::u32string ps = words(text);
    if (suffix == "s") ps += U"z";
    return ps;
}

std::u32string KokoroG2P::wordPhonemes(const std::u32string& word, bool futureKnown, bool futureVowel,
                                      bool nextIsWord) const {
    // Function words whose reading depends on what follows (misaki's get_special_case, minus the tagger).
    if (word == U"a") return U"ɐ";
    if (word == U"A") return nextIsWord ? U"ɐ" : U"ˈA";
    if (word == U"an" || word == U"An") return U"ɐn";
    if (word == U"I") return U"ˌI";
    if (word == U"the" || word == U"The") return futureKnown && futureVowel ? U"ði" : U"ðə";
    if (word == U"to" || word == U"To") {
        std::u32string g;
        if (!futureKnown) return lookup(U"to", g) ? g : U"tə";
        return futureVowel ? U"tʊ" : U"tə";
    }
    if (word == U"in" || word == U"In") return futureKnown ? U"ɪn" : U"ˈɪn";
    if (word == U"vs" || word == U"Vs") return words("versus");
    static const std::pair<const char32_t*, const char*> symbols[] = {
        {U"%", "percent"}, {U"&", "and"}, {U"+", "plus"}, {U"@", "at"}};
    for (auto& s : symbols)
        if (word == s.first) return words(s.second);

    // Capitalisation stresses: "Word" 0.5, "WORD" 2 (misaki's cap_stresses).
    const bool hasStress = word != lowerS(word);
    const double stress = word == upperS(word) ? 2.0 : 0.5;
    std::u32string ps;
    if (getWord(word, stress, hasStress, ps)) return ps;
    // A compound of two known words ("keyboardless" isn't, "catnap" is): longest known head first.
    const std::u32string wl = lowerS(word);
    for (size_t cut = wl.size() >= 3 ? wl.size() - 3 : 0; cut >= 3; cut--) {
        std::u32string head, tail;
        if (getWord(wl.substr(0, cut), 0, false, head) && getWord(wl.substr(cut), 0, false, tail) &&
            hasVowel(head) && hasVowel(tail))
            return applyStress(head + tail, hasStress, stress);
    }
    return spell(word);
}

std::vector<KokoroG2P::Token> KokoroG2P::tokenize(const std::u32string& input) const {
    std::vector<Token> out;
    std::u32string text;
    for (char32_t c : input) {
        if (c == U'‘' || c == U'’' || c == U'`') c = U'\'';
        text += fold(c);
    }
    std::u32string currency;
    for (size_t i = 0; i < text.size();) {
        const char32_t c = text[i];
        if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r') {
            if (!out.empty()) out.back().spaceAfter = true;
            i++;
            continue;
        }
        Token t;
        if (isAsciiAlpha(c)) {
            size_t j = i;
            while (j < text.size() && (isAsciiAlpha(text[j]) || (text[j] == U'\'' && j + 1 < text.size() && isAsciiAlpha(text[j + 1]))))
                j++;
            t.text = text.substr(i, j - i);
            t.word = true;
            i = j;
        } else if (isDigit(c) && i + 3 < text.size() && text[i + 1] == U':' && isDigit(text[i + 2]) && isDigit(text[i + 3]) ||
                   isDigit(c) && i + 4 < text.size() && isDigit(text[i + 1]) && text[i + 2] == U':' && isDigit(text[i + 3]) &&
                       isDigit(text[i + 4])) {
            // A time: "3:15" -> three fifteen, "7:05" -> seven oh five, "9:00" -> nine o'clock.
            const size_t colon = text.find(U':', i);
            const int hour = std::stoi(utf32To8(text.substr(i, colon - i)));
            const int minute = std::stoi(utf32To8(text.substr(colon + 1, 2)));
            std::string spoken = cardinal(hour);
            if (minute == 0) spoken += " o'clock";
            else if (minute < 10) spoken += std::string(" oh ") + kOnes[minute];
            else spoken += " " + below1000(minute);
            t.text = text.substr(i, colon + 3 - i);
            t.phonemes = words(spoken);
            i = colon + 3;
        } else if (isDigit(c) || ((c == U'-' || c == U'.') && i + 1 < text.size() && isDigit(text[i + 1]) &&
                                  (out.empty() || out.back().spaceAfter))) {
            size_t j = i + 1;
            while (j < text.size() && (isDigit(text[j]) || ((text[j] == U',' || text[j] == U'.') && j + 1 < text.size() && isDigit(text[j + 1]))))
                j++;
            size_t k = j;  // ordinal / plural suffix: 1st, 2nd, 1990s
            while (k < text.size() && isAsciiAlpha(text[k]) && k - j < 2) k++;
            std::u32string suffix = lowerS(text.substr(j, k - j));
            if (suffix == U"st" || suffix == U"nd" || suffix == U"rd" || suffix == U"th" || suffix == U"s") j = k;
            t.text = text.substr(i, j - i);
            t.phonemes = numberPhonemes(t.text, currency);
            currency.clear();
            i = j;
            if (i < text.size() && text[i] == U'%') {
                t.phonemes += U' ' + words("percent");
                i++;
            }
        } else if (c == U'$' || c == U'£' || c == U'€') {
            currency = std::u32string(1, c);
            i++;
            continue;
        } else if (c == U'-' || c == U'–' || c == U'—') {
            // Inside a word ("well-known") it is a space; standing alone it is a pause.
            const bool joined = !out.empty() && !out.back().spaceAfter && i + 1 < text.size() && isAsciiAlpha(text[i + 1]);
            if (joined) {
                out.back().spaceAfter = true;
                i++;
                continue;
            }
            t.text = U"—";
            t.phonemes = U"—";
            i++;
            while (i < text.size() && (text[i] == U'-' || text[i] == U'–')) i++;
        } else if (c == U'%' || c == U'&' || c == U'+' || c == U'@') {
            t.text = std::u32string(1, c);
            t.word = true;
            i++;
        } else {
            std::u32string p;
            if (in(kPuncts, c)) p = std::u32string(1, c);
            else if (c == U'(' || c == U')') p = std::u32string(1, c);
            else if (c == U'…') p = U"…";
            i++;
            if (p.empty()) continue;  // symbols the model can't say
            t.text = p;
            t.phonemes = p;
        }
        out.push_back(std::move(t));
    }
    return out;
}

std::string KokoroG2P::phonemize(const std::string& utf8) const {
    std::vector<Token> tokens = tokenize(utf8To32(utf8));
    // Right to left, so "the" and "to" can see whether the next sound is a vowel.
    bool futureKnown = false, futureVowel = false;
    for (size_t i = tokens.size(); i-- > 0;) {
        Token& t = tokens[i];
        if (t.word) {
            const bool nextIsWord = i + 1 < tokens.size() && tokens[i + 1].word && tokens[i + 1].text != upperS(tokens[i + 1].text);
            t.phonemes = wordPhonemes(t.text, futureKnown, futureVowel, nextIsWord);
        }
        for (char32_t c : t.phonemes) {
            if (in(kNonQuotePuncts, c)) { futureKnown = false; break; }
            if (in(kVowels, c)) { futureKnown = true; futureVowel = true; break; }
            if (in(kConsonants, c)) { futureKnown = true; futureVowel = false; break; }
        }
    }
    std::u32string out;
    for (size_t i = 0; i < tokens.size(); i++) {
        out += tokens[i].phonemes;
        if (tokens[i].spaceAfter && i + 1 < tokens.size() && !tokens[i].phonemes.empty()) out += U' ';
    }
    // Kokoro v1.0 was trained with flap and glottal stop written as T and t (misaki version != 2.0).
    for (char32_t& c : out) {
        if (c == U'ɾ') c = U'T';
        else if (c == U'ʔ') c = U't';
    }
    while (!out.empty() && out.back() == U' ') out.pop_back();
    return utf32To8(out);
}

}  // namespace neko
