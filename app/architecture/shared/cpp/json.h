// Minimal JSON reader for config.json, safetensors headers and tokenizer files.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neko {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    static Json parse(const char* data, size_t len);
    static Json parse(const std::string& s) { return parse(s.data(), s.size()); }
    static Json parseFile(const std::string& path);

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isObject() const { return type_ == Type::Object; }
    bool isArray() const { return type_ == Type::Array; }
    bool isString() const { return type_ == Type::String; }
    bool isNumber() const { return type_ == Type::Number; }

    double num(double fallback = 0) const { return type_ == Type::Number ? number_ : fallback; }
    int64_t i64(int64_t fallback = 0) const { return type_ == Type::Number ? int64_t(number_) : fallback; }
    bool boolean(bool fallback = false) const { return type_ == Type::Bool ? bool_ : fallback; }
    const std::string& str() const { return string_; }

    const std::vector<Json>& items() const { return array_; }
    // Objects keep insertion order: tokenizer vocabularies rely on deterministic iteration.
    const std::vector<std::pair<std::string, Json>>& members() const { return object_; }

    const Json& operator[](const std::string& key) const;  // Null if missing
    const Json& operator[](size_t i) const { return array_[i]; }
    bool has(const std::string& key) const;
    size_t size() const { return type_ == Type::Array ? array_.size() : object_.size(); }

    int64_t getInt(const std::string& key, int64_t fallback) const;
    double getNum(const std::string& key, double fallback) const;
    std::string getStr(const std::string& key, const std::string& fallback) const;

private:
    friend class JsonParser;
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0;
    std::string string_;
    std::vector<Json> array_;
    std::vector<std::pair<std::string, Json>> object_;
};

std::string readFile(const std::string& path);

}  // namespace neko
