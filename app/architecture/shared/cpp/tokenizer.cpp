#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <sstream>
#include <sys/stat.h>

#include "common.h"
#include "json.h"

namespace neko {

namespace {

bool fileExists(const std::string& p) {
    struct stat st{};
    return ::lstat(p.c_str(), &st) == 0;  // lstat: SAF model files are fd symlinks
}

std::string utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += char(cp);
    } else if (cp < 0x800) {
        out += char(0xC0 | (cp >> 6));
        out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += char(0xE0 | (cp >> 12));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | (cp >> 18));
        out += char(0x80 | ((cp >> 12) & 0x3F));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    }
    return out;
}

// Decodes one code point; invalid bytes decode as themselves (U+FFFD semantics are not needed).
uint32_t nextCp(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + size_t(len) > s.size()) len = 1;
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)s[i + size_t(k)] & 0x3F);
    i += size_t(len);
    return cp;
}

bool isSpace(uint32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool isNumber(uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F) ||
           (c >= 0x2070 && c <= 0x2089) || (c >= 0x2150 && c <= 0x2189) || (c >= 0x2460 && c <= 0x249B) ||
           (c >= 0xFF10 && c <= 0xFF19);
}

// Approximation of \p{L}: everything that is not a known punctuation/symbol/mark/number range.
bool isLetter(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (isNumber(c) || isSpace(c)) return false;
    if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA;
    if (c == 0xD7 || c == 0xF7) return false;
    if (c >= 0x300 && c <= 0x36F) return false;    // combining marks
    if (c >= 0x2000 && c <= 0x2BFF) return false;  // punctuation, symbols, arrows, math, dingbats
    if (c >= 0x2E00 && c <= 0x2E7F) return false;
    if (c >= 0x3000 && c <= 0x3004) return false;
    if (c >= 0x3008 && c <= 0x303F) return false;
    if (c >= 0xE000 && c <= 0xF8FF) return false;  // private use
    if (c >= 0xFE00 && c <= 0xFE0F) return false;  // variation selectors
    if (c >= 0xFE30 && c <= 0xFE6F) return false;
    if ((c >= 0xFF00 && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65)) return false;
    if (c >= 0x1F000 && c <= 0x1FAFF) return false;  // emoji & pictographs
    if (c >= 0xE0000) return false;
    return true;
}

bool isOther(uint32_t c) { return !isSpace(c) && !isLetter(c) && !isNumber(c); }
bool isNewline(uint32_t c) { return c == '\r' || c == '\n'; }

// End of the whitespace alternatives shared by both regexes: \s+(?!\S)|\s+
// (a run of spaces leaves its last one to prefix the following word).
size_t matchSpaces(const std::vector<uint32_t>& c, size_t i) {
    size_t j = i;
    while (j < c.size() && isSpace(c[j])) j++;
    if (j < c.size() && j - i > 1) j--;
    return j;
}

// GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
size_t matchGpt2(const std::vector<uint32_t>& c, size_t i) {
    const size_t n = c.size();
    if (c[i] == '\'' && i + 1 < n) {
        uint32_t a = c[i + 1], b = i + 2 < n ? c[i + 2] : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    size_t k = i + (c[i] == ' ' && i + 1 < n && !isSpace(c[i + 1]) ? 1 : 0);
    bool (*cls)(uint32_t) = isLetter(c[k]) ? isLetter : isNumber(c[k]) ? isNumber : isOther(c[k]) ? isOther : nullptr;
    if (!cls) return matchSpaces(c, i);
    while (k < n && cls(c[k])) k++;
    return k;
}

// Qwen2/Qwen3:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
size_t matchQwen2(const std::vector<uint32_t>& c, size_t i) {
    const size_t n = c.size();
    auto lower = [](uint32_t x) { return x >= 'A' && x <= 'Z' ? x + 32 : x; };
    if (c[i] == '\'' && i + 1 < n) {
        uint32_t a = lower(c[i + 1]), b = i + 2 < n ? lower(c[i + 2]) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    // Letters, optionally prefixed by one character that is not a newline, letter or digit.
    size_t k = i;
    if (!isLetter(c[k]) && !isNewline(c[k]) && !isNumber(c[k]) && k + 1 < n && isLetter(c[k + 1])) k++;
    if (isLetter(c[k])) {
        while (k < n && isLetter(c[k])) k++;
        return k;
    }
    if (isNumber(c[i])) return i + 1;  // digits are split one by one
    k = i + (c[i] == ' ' && i + 1 < n && isOther(c[i + 1]) ? 1 : 0);
    if (isOther(c[k])) {
        while (k < n && isOther(c[k])) k++;
        while (k < n && isNewline(c[k])) k++;
        return k;
    }
    // Whitespace containing newlines ends after its last newline.
    size_t lastNl = SIZE_MAX;
    for (k = i; k < n && isSpace(c[k]); k++)
        if (isNewline(c[k])) lastNl = k;
    if (lastNl != SIZE_MAX) return lastNl + 1;
    return matchSpaces(c, i);
}

}  // namespace

void BpeTokenizer::load(const std::string& dir, PreTokenizer style) {
    style_ = style;
    // bytes_to_unicode() from the original GPT-2 encoder.
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<bool> present(256, false);
    for (int b : bs) { present[size_t(b)] = true; byteToUnicode_[b] = utf8(uint32_t(b)); }
    int n = 0;
    for (int b = 0; b < 256; b++)
        if (!present[size_t(b)]) byteToUnicode_[b] = utf8(uint32_t(256 + n++));

    std::string tj = dir + "/tokenizer.json";
    if (fileExists(dir + "/vocab.json") && fileExists(dir + "/merges.txt")) {
        Json v = Json::parseFile(dir + "/vocab.json");
        for (auto& kv : v.members()) vocab_[kv.first] = int(kv.second.i64());
        std::istringstream mf(readFile(dir + "/merges.txt"));
        std::string line;
        int rank = 0;
        while (std::getline(mf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || starts_with(line, "#version")) continue;
            mergeRank_.emplace(line, rank++);
        }
        if (fileExists(tj)) {
            Json t = Json::parseFile(tj);
            for (auto& a : t["added_tokens"].items()) special_.emplace_back(a["content"].str(), int(a["id"].i64()));
        }
    } else if (fileExists(tj)) {
        Json t = Json::parseFile(tj);
        const Json& model = t["model"];
        if (model.getStr("type", "BPE") != "BPE") fail("tokenizer.json is not a BPE tokenizer");
        for (auto& kv : model["vocab"].members()) vocab_[kv.first] = int(kv.second.i64());
        int rank = 0;
        for (auto& m : model["merges"].items()) {
            if (m.isString()) mergeRank_.emplace(m.str(), rank++);
            else if (m.isArray() && m.size() == 2) mergeRank_.emplace(m[0].str() + " " + m[1].str(), rank++);
        }
        for (auto& a : t["added_tokens"].items()) special_.emplace_back(a["content"].str(), int(a["id"].i64()));
    } else {
        fail("no tokenizer found (need vocab.json + merges.txt or tokenizer.json)");
    }
    finalize();
}

void BpeTokenizer::finalize() {
    std::unordered_map<std::string, int> unicodeToByte;
    for (int b = 0; b < 256; b++) unicodeToByte[byteToUnicode_[b]] = b;
    int maxId = 0;
    for (auto& kv : vocab_) maxId = std::max(maxId, kv.second);
    for (auto& s : special_) maxId = std::max(maxId, s.second);
    idToBytes_.assign(size_t(maxId) + 1, std::string());
    for (auto& kv : vocab_) {
        std::string raw;
        size_t i = 0;
        const std::string& s = kv.first;
        while (i < s.size()) {
            size_t st = i;
            nextCp(s, i);
            auto it = unicodeToByte.find(s.substr(st, i - st));
            if (it != unicodeToByte.end()) raw += char(it->second);
            else raw += s.substr(st, i - st);
        }
        idToBytes_[size_t(kv.second)] = raw;
    }
    for (auto& s : special_) idToBytes_[size_t(s.second)] = s.first;
    eosId_ = tokenId("<|endoftext|>");  // GPT-2 keeps it in the vocabulary, Qwen as an added token
    if (std::none_of(special_.begin(), special_.end(), [](auto& s) { return s.first == "<|endoftext|>"; }) && eosId_ >= 0)
        special_.emplace_back("<|endoftext|>", eosId_);
    // Longest match first when splitting on added tokens.
    std::sort(special_.begin(), special_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
}

int BpeTokenizer::tokenId(const std::string& text) const {
    for (auto& s : special_)
        if (s.first == text) return s.second;
    auto it = vocab_.find(text);
    return it != vocab_.end() ? it->second : -1;
}

const std::string& BpeTokenizer::tokenBytes(int id) const {
    static const std::string empty;
    if (id < 0 || size_t(id) >= idToBytes_.size()) return empty;
    return idToBytes_[size_t(id)];
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += tokenBytes(id);
    return out;
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    size_t start = 0;
    while (start < text.size()) {
        // Find the earliest special token occurrence.
        size_t best = std::string::npos;
        const std::pair<std::string, int>* hit = nullptr;
        for (auto& s : special_) {
            size_t p = text.find(s.first, start);
            if (p != std::string::npos && (p < best)) { best = p; hit = &s; }
        }
        if (!hit) { encodeOrdinary(text.substr(start), out); break; }
        if (best > start) encodeOrdinary(text.substr(start, best - start), out);
        out.push_back(hit->second);
        start = best + hit->first.size();
    }
    return out;
}

// Splits text with the hand-written equivalent of the pre-tokenizer regex, then BPE-encodes each piece.
void BpeTokenizer::encodeOrdinary(const std::string& text, std::vector<int>& out) const {
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;  // byte offset of each code point, plus end
    for (size_t i = 0; i < text.size();) {
        offs.push_back(i);
        cps.push_back(nextCp(text, i));
    }
    offs.push_back(text.size());
    for (size_t i = 0; i < cps.size();) {
        size_t j = style_ == PreTokenizer::Qwen2 ? matchQwen2(cps, i) : matchGpt2(cps, i);
        encodeWord(text.substr(offs[i], offs[j] - offs[i]), out);
        i = j;
    }
}

void BpeTokenizer::encodeWord(const std::string& word, std::vector<int>& out) const {
    auto cached = cache_.find(word);
    if (cached != cache_.end()) {
        out.insert(out.end(), cached->second.begin(), cached->second.end());
        return;
    }
    std::vector<std::string> sym;
    for (unsigned char c : word) sym.push_back(byteToUnicode_[c]);
    while (sym.size() > 1) {
        int bestRank = INT_MAX;
        size_t bestAt = 0;
        for (size_t k = 0; k + 1 < sym.size(); k++) {
            auto it = mergeRank_.find(sym[k] + " " + sym[k + 1]);
            if (it != mergeRank_.end() && it->second < bestRank) { bestRank = it->second; bestAt = k; }
        }
        if (bestRank == INT_MAX) break;
        std::string a = sym[bestAt], b = sym[bestAt + 1];
        std::vector<std::string> next;
        for (size_t k = 0; k < sym.size();) {
            if (k + 1 < sym.size() && sym[k] == a && sym[k + 1] == b) {
                next.push_back(a + b);
                k += 2;
            } else {
                next.push_back(sym[k++]);
            }
        }
        sym.swap(next);
    }
    std::vector<int> ids;
    for (auto& s : sym) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            // Unknown merge result: fall back to single bytes (always in a byte-level vocab).
            size_t i = 0;
            while (i < s.size()) {
                size_t st = i;
                nextCp(s, i);
                auto bt = vocab_.find(s.substr(st, i - st));
                if (bt != vocab_.end()) ids.push_back(bt->second);
            }
        }
    }
    if (cache_.size() < 50000) cache_.emplace(word, ids);
    out.insert(out.end(), ids.begin(), ids.end());
}

}  // namespace neko
