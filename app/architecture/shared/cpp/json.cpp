#include "json.h"

#include <fcntl.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>

#include "common.h"

namespace neko {

namespace {
const Json kNull;

void appendUtf8(std::string& out, uint32_t cp) {
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
}
}  // namespace

class JsonParser {
public:
    JsonParser(const char* p, size_t n) : p_(p), end_(p + n) {}

    Json parseValue() {
        skipWs();
        if (p_ >= end_) fail("json: unexpected end");
        Json j;
        char c = *p_;
        if (c == '{') {
            j.type_ = Json::Type::Object;
            ++p_;
            skipWs();
            if (peek() == '}') { ++p_; return j; }
            for (;;) {
                skipWs();
                std::string key = parseString();
                skipWs();
                expect(':');
                j.object_.emplace_back(std::move(key), parseValue());
                skipWs();
                if (peek() == ',') { ++p_; continue; }
                expect('}');
                break;
            }
        } else if (c == '[') {
            j.type_ = Json::Type::Array;
            ++p_;
            skipWs();
            if (peek() == ']') { ++p_; return j; }
            for (;;) {
                j.array_.push_back(parseValue());
                skipWs();
                if (peek() == ',') { ++p_; continue; }
                expect(']');
                break;
            }
        } else if (c == '"') {
            j.type_ = Json::Type::String;
            j.string_ = parseString();
        } else if (c == 't' || c == 'f') {
            j.type_ = Json::Type::Bool;
            j.bool_ = c == 't';
            p_ += j.bool_ ? 4 : 5;
        } else if (c == 'n') {
            p_ += 4;
        } else {
            // NaN/Infinity appear in some HF configs; treat them as numbers.
            j.type_ = Json::Type::Number;
            char* e = nullptr;
            std::string tmp;
            const char* s = p_;
            while (p_ < end_ && (std::strchr("+-.eE0123456789", *p_) || std::isalpha((unsigned char)*p_))) ++p_;
            tmp.assign(s, p_);
            j.number_ = std::strtod(tmp.c_str(), &e);
        }
        return j;
    }

    void skipWs() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
    }

private:
    char peek() const { return p_ < end_ ? *p_ : '\0'; }
    void expect(char c) {
        if (peek() != c) fail(std::string("json: expected '") + c + "'");
        ++p_;
    }
    uint32_t hex4() {
        if (end_ - p_ < 4) fail("json: bad escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("json: bad hex");
        }
        return v;
    }
    std::string parseString() {
        expect('"');
        std::string out;
        while (p_ < end_ && *p_ != '"') {
            char c = *p_++;
            if (c != '\\') { out += c; continue; }
            if (p_ >= end_) break;
            char e = *p_++;
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                        p_ += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: out += e; break;
            }
        }
        expect('"');
        return out;
    }

    const char* p_;
    const char* end_;
};

Json Json::parse(const char* data, size_t len) {
    JsonParser p(data, len);
    return p.parseValue();
}

int openReadOnly(const std::string& path) {
    char target[64];
    ssize_t n = ::readlink(path.c_str(), target, sizeof target - 1);
    if (n > 0) {
        target[n] = '\0';
        static const char kFdPrefix[] = "/proc/self/fd/";
        if (std::strncmp(target, kFdPrefix, sizeof kFdPrefix - 1) == 0) {
            int fd = std::atoi(target + sizeof kFdPrefix - 1);
            int d = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
            if (d >= 0) {
                ::lseek(d, 0, SEEK_SET);
                return d;
            }
        }
    }
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
}

std::string readFile(const std::string& path) {
    int fd = openReadOnly(path);
    if (fd < 0) fail("cannot open " + path);
    std::string s;
    char buf[1 << 16];
    off_t off = 0;
    for (;;) {
        ssize_t got = ::pread(fd, buf, sizeof buf, off);  // pread: dup'ed fds share the file offset
        if (got <= 0) break;
        s.append(buf, size_t(got));
        off += got;
    }
    ::close(fd);
    return s;
}

Json Json::parseFile(const std::string& path) {
    std::string s = readFile(path);
    return parse(s);
}

const Json& Json::operator[](const std::string& key) const {
    for (auto& kv : object_)
        if (kv.first == key) return kv.second;
    return kNull;
}

bool Json::has(const std::string& key) const {
    for (auto& kv : object_)
        if (kv.first == key) return true;
    return false;
}

int64_t Json::getInt(const std::string& key, int64_t fallback) const {
    const Json& j = (*this)[key];
    return j.isNumber() ? j.i64() : fallback;
}

double Json::getNum(const std::string& key, double fallback) const {
    const Json& j = (*this)[key];
    return j.isNumber() ? j.num() : fallback;
}

std::string Json::getStr(const std::string& key, const std::string& fallback) const {
    const Json& j = (*this)[key];
    return j.isString() ? j.str() : fallback;
}

}  // namespace neko
