#include "tensor_store.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common.h"
#include "json.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace neko {

size_t dtypeSize(DType t) {
    switch (t) {
        case DType::F32: case DType::I32: return 4;
        case DType::F16: case DType::BF16: return 2;
        case DType::F64: case DType::I64: return 8;
        default: return 1;
    }
}

const char* dtypeName(DType t) {
    switch (t) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::F64: return "F64";
        case DType::I64: return "I64";
        case DType::I32: return "I32";
        case DType::I8: return "I8";
        case DType::U8: return "U8";
        case DType::Bool: return "BOOL";
    }
    return "?";
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
    int fd = openReadOnly(path);
    if (fd < 0) fail("cannot open " + path);
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); fail("cannot stat " + path); }
    size_ = size_t(st.st_size);
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) fail("mmap failed for " + path);
    data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
    if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
}

const TensorView& TensorStore::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("missing tensor: " + name);
    return it->second;
}

void TensorStore::addFile(const std::string& path) {
    auto f = std::make_shared<MappedFile>(path);
    files_.push_back(f);
    if (ends_with(path, ".safetensors")) {
        loadSafetensors(f);
    } else {
        // PyTorch >= 1.6 checkpoints are zip archives ("PK\x03\x04").
        if (f->size() < 4 || std::memcmp(f->data(), "PK\x03\x04", 4) != 0)
            fail("unsupported checkpoint (legacy non-zip torch format): " + path);
        loadTorchZip(f);
    }
}

// ---------------------------------------------------------------- safetensors

static DType parseSafetensorsDtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "F64") return DType::F64;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "BOOL") return DType::Bool;
    fail("unsupported safetensors dtype " + s);
}

void TensorStore::loadSafetensors(std::shared_ptr<MappedFile> f) {
    if (f->size() < 8) fail("truncated safetensors file");
    uint64_t hlen;
    std::memcpy(&hlen, f->data(), 8);
    if (hlen > f->size() - 8) fail("bad safetensors header");
    Json header = Json::parse(reinterpret_cast<const char*>(f->data() + 8), size_t(hlen));
    const uint8_t* base = f->data() + 8 + hlen;
    size_t dataSize = f->size() - 8 - hlen;
    for (auto& kv : header.members()) {
        if (kv.first == "__metadata__") continue;
        const Json& t = kv.second;
        TensorView v;
        v.dtype = parseSafetensorsDtype(t["dtype"].str());
        for (auto& d : t["shape"].items()) v.shape.push_back(d.i64());
        uint64_t b = uint64_t(t["data_offsets"][0].i64());
        uint64_t e = uint64_t(t["data_offsets"][1].i64());
        if (e > dataSize || b > e || e - b != uint64_t(v.numel()) * dtypeSize(v.dtype))
            fail("bad data offsets for " + kv.first);
        v.data = base + b;
        tensors_[kv.first] = std::move(v);
    }
}

// ---------------------------------------------------------------- torch zip + pickle

namespace {

struct ZipEntry {
    std::string name;
    uint64_t localHeader = 0;
    uint64_t size = 0;
    uint16_t method = 0;
};

template <typename T>
T rd(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

std::vector<ZipEntry> readZipDirectory(const MappedFile& f) {
    const uint8_t* d = f.data();
    size_t n = f.size();
    if (n < 22) fail("zip too small");
    // Find End Of Central Directory record (scan back over the optional comment).
    size_t eocd = std::string::npos;
    for (size_t i = n - 22 + 1; i-- > (n > 65557 ? n - 65557 : 0);) {
        if (rd<uint32_t>(d + i) == 0x06054b50) { eocd = i; break; }
    }
    if (eocd == std::string::npos) fail("zip: no end of central directory");
    uint64_t count = rd<uint16_t>(d + eocd + 10);
    uint64_t cdOffset = rd<uint32_t>(d + eocd + 16);
    // ZIP64 (torch uses it for large checkpoints).
    if (eocd >= 20 && rd<uint32_t>(d + eocd - 20) == 0x07064b50) {
        uint64_t z64 = rd<uint64_t>(d + eocd - 20 + 8);
        if (z64 + 56 <= n && rd<uint32_t>(d + z64) == 0x06064b50) {
            count = rd<uint64_t>(d + z64 + 32);
            cdOffset = rd<uint64_t>(d + z64 + 48);
        }
    }
    std::vector<ZipEntry> out;
    size_t p = size_t(cdOffset);
    for (uint64_t i = 0; i < count; i++) {
        if (p + 46 > n || rd<uint32_t>(d + p) != 0x02014b50) fail("zip: bad central directory");
        ZipEntry e;
        e.method = rd<uint16_t>(d + p + 10);
        uint64_t csize = rd<uint32_t>(d + p + 20);
        uint64_t usize = rd<uint32_t>(d + p + 24);
        uint16_t nameLen = rd<uint16_t>(d + p + 28);
        uint16_t extraLen = rd<uint16_t>(d + p + 30);
        uint16_t commentLen = rd<uint16_t>(d + p + 32);
        uint64_t local = rd<uint32_t>(d + p + 42);
        e.name.assign(reinterpret_cast<const char*>(d + p + 46), nameLen);
        // ZIP64 extended info: fields present only for values saturated at 0xFFFFFFFF.
        const uint8_t* x = d + p + 46 + nameLen;
        const uint8_t* xe = x + extraLen;
        while (x + 4 <= xe) {
            uint16_t id = rd<uint16_t>(x), len = rd<uint16_t>(x + 2);
            const uint8_t* v = x + 4;
            if (id == 0x0001) {
                if (usize == 0xFFFFFFFFu) { usize = rd<uint64_t>(v); v += 8; }
                if (csize == 0xFFFFFFFFu) { csize = rd<uint64_t>(v); v += 8; }
                if (local == 0xFFFFFFFFu) { local = rd<uint64_t>(v); v += 8; }
            }
            x += 4 + len;
        }
        e.size = usize;
        e.localHeader = local;
        out.push_back(std::move(e));
        p += 46 + nameLen + extraLen + commentLen;
    }
    return out;
}

const uint8_t* zipEntryData(const MappedFile& f, const ZipEntry& e) {
    if (e.method != 0) fail("zip entry is compressed (unsupported): " + e.name);
    const uint8_t* d = f.data();
    size_t p = size_t(e.localHeader);
    if (p + 30 > f.size() || rd<uint32_t>(d + p) != 0x04034b50) fail("zip: bad local header");
    size_t off = p + 30 + rd<uint16_t>(d + p + 26) + rd<uint16_t>(d + p + 28);
    if (off + e.size > f.size()) fail("zip: entry out of range");
    return d + off;
}

// A tiny pickle VM that understands exactly what torch.save() emits for state dicts.
struct PObj;
using PRef = std::shared_ptr<PObj>;

struct PObj {
    enum Kind { None, Bool, Int, Float, Str, Tuple, List, Dict, Global, Storage, Tensor, Mark, Opaque } kind = None;
    int64_t i = 0;
    double f = 0;
    std::string s, s2;  // Str / Global(module, name) / Storage key
    std::vector<PRef> items;                       // Tuple / List
    std::vector<std::pair<PRef, PRef>> dict;       // Dict
    DType dtype = DType::F32;                      // Storage / Tensor
    PRef storage;                                  // Tensor
    int64_t offset = 0;                            // Tensor (elements)
    std::vector<int64_t> shape, stride;            // Tensor
};

PRef mk(PObj::Kind k) {
    auto p = std::make_shared<PObj>();
    p->kind = k;
    return p;
}

DType storageDtype(const std::string& name) {
    if (name == "FloatStorage") return DType::F32;
    if (name == "HalfStorage") return DType::F16;
    if (name == "BFloat16Storage") return DType::BF16;
    if (name == "DoubleStorage") return DType::F64;
    if (name == "LongStorage") return DType::I64;
    if (name == "IntStorage") return DType::I32;
    if (name == "CharStorage") return DType::I8;
    if (name == "ByteStorage") return DType::U8;
    if (name == "BoolStorage") return DType::Bool;
    fail("unsupported torch storage type " + name);
}

std::vector<int64_t> intTuple(const PRef& t) {
    std::vector<int64_t> v;
    for (auto& x : t->items) v.push_back(x->i);
    return v;
}

PRef reduceCall(const PRef& fn, const PRef& args) {
    if (fn->kind == PObj::Global) {
        const std::string& name = fn->s2;
        if (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor") {
            auto t = mk(PObj::Tensor);
            t->storage = args->items.at(0);
            t->dtype = t->storage->dtype;
            t->offset = args->items.at(1)->i;
            t->shape = intTuple(args->items.at(2));
            t->stride = intTuple(args->items.at(3));
            return t;
        }
        if (name == "_rebuild_parameter" || name == "_rebuild_parameter_with_state") return args->items.at(0);
        if (name == "OrderedDict" || name == "dict") return mk(PObj::Dict);
    }
    return mk(PObj::Opaque);
}

PRef runPickle(const uint8_t* p, size_t n) {
    const uint8_t* end = p + n;
    std::vector<PRef> stack;
    std::vector<size_t> marks;
    std::unordered_map<uint32_t, PRef> memo;
    auto need = [&](size_t k) { if (size_t(end - p) < k) fail("pickle: truncated"); };
    auto pop = [&]() {
        if (stack.empty()) fail("pickle: stack underflow");
        PRef r = stack.back();
        stack.pop_back();
        return r;
    };
    auto popMark = [&]() {
        if (marks.empty()) fail("pickle: no mark");
        size_t m = marks.back();
        marks.pop_back();
        std::vector<PRef> items(stack.begin() + long(m), stack.end());
        stack.resize(m);
        return items;
    };
    auto str = [&](size_t len) {
        need(len);
        auto o = mk(PObj::Str);
        o->s.assign(reinterpret_cast<const char*>(p), len);
        p += len;
        return o;
    };
    auto readLine = [&]() {
        std::string s;
        while (p < end && *p != '\n') s += char(*p++);
        if (p < end) ++p;
        return s;
    };

    while (p < end) {
        uint8_t op = *p++;
        switch (op) {
            case 0x80: need(1); p += 1; break;          // PROTO
            case 0x95: need(8); p += 8; break;          // FRAME
            case '(': marks.push_back(stack.size()); break;
            case '.': return pop();                      // STOP
            case 'N': stack.push_back(mk(PObj::None)); break;
            case 0x88: case 0x89: { auto b = mk(PObj::Bool); b->i = op == 0x88; stack.push_back(b); break; }
            case 'J': { need(4); auto o = mk(PObj::Int); o->i = rd<int32_t>(p); p += 4; stack.push_back(o); break; }
            case 'K': { need(1); auto o = mk(PObj::Int); o->i = *p++; stack.push_back(o); break; }
            case 'M': { need(2); auto o = mk(PObj::Int); o->i = rd<uint16_t>(p); p += 2; stack.push_back(o); break; }
            case 0x8a: {  // LONG1
                need(1);
                uint8_t len = *p++;
                need(len);
                int64_t v = 0;
                for (int k = 0; k < len && k < 8; k++) v |= int64_t(p[k]) << (8 * k);
                if (len > 0 && len < 8 && (p[len - 1] & 0x80)) v -= int64_t(1) << (8 * len);
                p += len;
                auto o = mk(PObj::Int);
                o->i = v;
                stack.push_back(o);
                break;
            }
            case 'G': {  // BINFLOAT (big endian)
                need(8);
                uint64_t b = 0;
                for (int k = 0; k < 8; k++) b = (b << 8) | p[k];
                p += 8;
                auto o = mk(PObj::Float);
                std::memcpy(&o->f, &b, 8);
                stack.push_back(o);
                break;
            }
            case 'X': { need(4); uint32_t l = rd<uint32_t>(p); p += 4; stack.push_back(str(l)); break; }
            case 0x8c: case 'C': { need(1); uint8_t l = *p++; stack.push_back(str(l)); break; }
            case 'B': { need(4); uint32_t l = rd<uint32_t>(p); p += 4; stack.push_back(str(l)); break; }
            case 0x8d: case 0x8e: { need(8); uint64_t l = rd<uint64_t>(p); p += 8; stack.push_back(str(size_t(l))); break; }
            case '}': stack.push_back(mk(PObj::Dict)); break;
            case ']': stack.push_back(mk(PObj::List)); break;
            case ')': stack.push_back(mk(PObj::Tuple)); break;
            case 't': { auto t = mk(PObj::Tuple); t->items = popMark(); stack.push_back(t); break; }
            case 0x85: case 0x86: case 0x87: {
                size_t k = op - 0x84;
                auto t = mk(PObj::Tuple);
                if (stack.size() < k) fail("pickle: tuple underflow");
                t->items.assign(stack.end() - long(k), stack.end());
                stack.resize(stack.size() - k);
                stack.push_back(t);
                break;
            }
            case 'q': { need(1); memo[*p++] = stack.back(); break; }
            case 'r': { need(4); memo[rd<uint32_t>(p)] = stack.back(); p += 4; break; }
            case 0x94: { memo[uint32_t(memo.size())] = stack.back(); break; }
            case 'h': { need(1); stack.push_back(memo.at(*p++)); break; }
            case 'j': { need(4); stack.push_back(memo.at(rd<uint32_t>(p))); p += 4; break; }
            case 'c': {  // GLOBAL "module\nname\n"
                auto g = mk(PObj::Global);
                g->s = readLine();
                g->s2 = readLine();
                stack.push_back(g);
                break;
            }
            case 0x93: {  // STACK_GLOBAL
                auto name = pop();
                auto mod = pop();
                auto g = mk(PObj::Global);
                g->s = mod->s;
                g->s2 = name->s;
                stack.push_back(g);
                break;
            }
            case 'Q': {  // BINPERSID: ('storage', type, key, location, numel)
                auto pid = pop();
                if (pid->kind != PObj::Tuple || pid->items.size() < 3) fail("pickle: bad persistent id");
                auto st = mk(PObj::Storage);
                st->dtype = storageDtype(pid->items[1]->s2);
                st->s = pid->items[2]->s;
                stack.push_back(st);
                break;
            }
            case 'R': {
                auto args = pop();
                auto fn = pop();
                stack.push_back(reduceCall(fn, args));
                break;
            }
            case 0x81: {  // NEWOBJ
                auto args = pop();
                auto cls = pop();
                stack.push_back(reduceCall(cls, args));
                break;
            }
            case 'b': pop(); break;  // BUILD: state is irrelevant for weights
            case 's': {
                auto v = pop();
                auto k = pop();
                stack.back()->dict.emplace_back(k, v);
                break;
            }
            case 'u': {
                auto items = popMark();
                auto& d = stack.back();
                for (size_t k = 0; k + 1 < items.size(); k += 2) d->dict.emplace_back(items[k], items[k + 1]);
                break;
            }
            case 'a': { auto v = pop(); stack.back()->items.push_back(v); break; }
            case 'e': {
                auto items = popMark();
                auto& l = stack.back();
                l->items.insert(l->items.end(), items.begin(), items.end());
                break;
            }
            default: {
                char buf[64];
                std::snprintf(buf, sizeof buf, "pickle: unsupported opcode 0x%02x", op);
                fail(buf);
            }
        }
    }
    fail("pickle: missing STOP");
}

bool dictHasTensors(const PRef& d) {
    for (auto& kv : d->dict)
        if (kv.second->kind == PObj::Tensor) return true;
    return false;
}

}  // namespace

void TensorStore::loadTorchZip(std::shared_ptr<MappedFile> f) {
    auto entries = readZipDirectory(*f);
    const ZipEntry* pkl = nullptr;
    for (auto& e : entries)
        if (ends_with(e.name, "/data.pkl") || e.name == "data.pkl") { pkl = &e; break; }
    if (!pkl) fail("torch checkpoint has no data.pkl");
    std::string prefix = pkl->name.substr(0, pkl->name.size() - std::strlen("data.pkl"));

    std::unordered_map<std::string, const ZipEntry*> byName;
    for (auto& e : entries) byName[e.name] = &e;

    PRef root = runPickle(zipEntryData(*f, *pkl), size_t(pkl->size));
    // Accept a raw state_dict or a training checkpoint wrapping one.
    if (root->kind == PObj::Dict && !dictHasTensors(root)) {
        for (auto& kv : root->dict) {
            if (kv.second->kind == PObj::Dict && dictHasTensors(kv.second)) { root = kv.second; break; }
        }
    }
    if (root->kind != PObj::Dict || !dictHasTensors(root)) fail("torch checkpoint does not contain a state_dict");

    for (auto& kv : root->dict) {
        const PRef& t = kv.second;
        if (kv.first->kind != PObj::Str || t->kind != PObj::Tensor) continue;
        auto it = byName.find(prefix + "data/" + t->storage->s);
        if (it == byName.end()) fail("missing storage for " + kv.first->s);
        const uint8_t* base = zipEntryData(*f, *it->second);
        // Only contiguous tensors are supported (what state_dict() produces for linear weights).
        int64_t expect = 1;
        for (size_t d = t->shape.size(); d-- > 0;) {
            if (t->shape[d] != 1 && t->stride[d] != expect) fail("non-contiguous tensor " + kv.first->s);
            expect *= t->shape[d];
        }
        TensorView v;
        v.dtype = t->dtype;
        v.shape = t->shape;
        v.data = base + size_t(t->offset) * dtypeSize(t->dtype);
        if (v.data + size_t(v.numel()) * dtypeSize(v.dtype) > base + it->second->size)
            fail("tensor out of storage range: " + kv.first->s);
        tensors_[kv.first->s] = std::move(v);
    }
}

// ---------------------------------------------------------------- conversions

static float loadAsF32(const TensorView& v, int64_t i) {
    switch (v.dtype) {
        case DType::F32: return reinterpret_cast<const float*>(v.data)[i];
        case DType::F16: return f16_to_f32(reinterpret_cast<const uint16_t*>(v.data)[i]);
        case DType::BF16: return bf16_to_f32(reinterpret_cast<const uint16_t*>(v.data)[i]);
        case DType::F64: return float(reinterpret_cast<const double*>(v.data)[i]);
        default: fail(std::string("tensor is not floating point: ") + dtypeName(v.dtype));
    }
}

std::vector<float> TensorStore::toF32(const std::string& name) const {
    const TensorView& v = get(name);
    std::vector<float> out(size_t(v.numel()));
    if (v.dtype == DType::F32) {
        std::memcpy(out.data(), v.data, out.size() * 4);
    } else {
        for (int64_t i = 0; i < v.numel(); i++) out[size_t(i)] = loadAsF32(v, i);
    }
    return out;
}

std::vector<uint16_t> TensorStore::toF16(const std::string& name, bool transpose2d) const {
    const TensorView& v = get(name);
    std::vector<uint16_t> out(size_t(v.numel()));
    if (!transpose2d) {
        if (v.dtype == DType::F16) {
            std::memcpy(out.data(), v.data, out.size() * 2);
        } else if (v.dtype == DType::BF16) {
            // bf16 has fewer mantissa bits than f16, so this is exact for weights within f16 range.
            const uint16_t* s = reinterpret_cast<const uint16_t*>(v.data);
            size_t i = 0;
#if defined(__aarch64__)
            for (; i + 4 <= out.size(); i += 4) {
                float32x4_t f = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s + i), 16));
                vst1_u16(out.data() + i, vreinterpret_u16_f16(vcvt_f16_f32(f)));
            }
#endif
            for (; i < out.size(); i++) out[i] = f32_to_f16(bf16_to_f32(s[i]));
        } else {
            for (int64_t i = 0; i < v.numel(); i++) out[size_t(i)] = f32_to_f16(loadAsF32(v, i));
        }
        return out;
    }
    if (v.shape.size() != 2) fail("transpose of non-2D tensor " + name);
    int64_t R = v.shape[0], C = v.shape[1];
    // Blocked transpose keeps the source reads cache friendly.
    const int64_t B = 64;
    for (int64_t r0 = 0; r0 < R; r0 += B)
        for (int64_t c0 = 0; c0 < C; c0 += B)
            for (int64_t r = r0; r < std::min(R, r0 + B); r++)
                for (int64_t c = c0; c < std::min(C, c0 + B); c++)
                    out[size_t(c * R + r)] = f32_to_f16(loadAsF32(v, r * C + c));
    return out;
}

}  // namespace neko
