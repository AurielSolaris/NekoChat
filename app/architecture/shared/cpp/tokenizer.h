// Byte-level BPE tokenizer (vocab.json + merges.txt, or HF tokenizer.json) for GPT-2 and Qwen2/Qwen3 vocabularies.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace neko {

// The regex that splits text into words before BPE; it is the only difference between the two families.
enum class PreTokenizer { Gpt2, Qwen2 };

class BpeTokenizer {
public:
    void load(const std::string& modelDir, PreTokenizer style = PreTokenizer::Gpt2);

    std::vector<int> encode(const std::string& text) const;
    // Raw bytes of one token (may be a partial UTF-8 sequence).
    const std::string& tokenBytes(int id) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocabSize() const { return int(idToBytes_.size()); }
    int eosId() const { return eosId_; }
    // Id of a special/added token or a vocabulary entry, -1 if absent.
    int tokenId(const std::string& text) const;

private:
    void encodeWord(const std::string& word, std::vector<int>& out) const;
    void encodeOrdinary(const std::string& text, std::vector<int>& out) const;
    void finalize();

    PreTokenizer style_ = PreTokenizer::Gpt2;
    std::unordered_map<std::string, int> vocab_;       // byte-mapped token string -> id
    std::unordered_map<std::string, int> mergeRank_;   // "a b" -> rank
    std::vector<std::string> idToBytes_;               // id -> raw bytes
    std::vector<std::pair<std::string, int>> special_; // added tokens matched literally
    std::string byteToUnicode_[256];
    int eosId_ = -1;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace neko
