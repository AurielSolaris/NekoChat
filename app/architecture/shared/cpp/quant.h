// Weight storage formats shared by the CPU and GPU paths.
//
// F16 keeps IEEE halves. FP8 (OCP E4M3) and FP4 (OCP E2M1) group each row into blocks of 32 weights that share
// one f16 scale (weight = code * scale), so an outlier only costs precision inside its own block.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace neko {

class ThreadPool;

enum class WeightFormat : int { F16 = 0, FP8 = 1, FP4 = 2 };

const char* weightFormatName(WeightFormat f);

constexpr int kQuantBlock = 32;

// Codes are decoded by reinterpreting them as f16 bit patterns (see quant.cpp), which yields value / factor;
// kernels fold the factor into the block scale.
constexpr float kFp8Factor = 256.0f;
constexpr float kFp4Factor = 16384.0f;

// Bytes per row: F16 2 * cols; FP8/FP4 the codes of cols / 32 blocks (32 or 16 bytes each) followed by the
// cols / 32 f16 scales, zero-padded to a multiple of 16 bytes.
size_t quantRowBytes(WeightFormat f, int cols);

// Row-major [rows][cols] weights. Rows are self-contained, so row ranges can be sliced (GPU buffer chunks) and
// matrices concatenated by rows (q|k|v, gate|up) as plain bytes. Inside an FP4 block, weight j sits in the low
// nibble of byte j and weight 16 + j in its high nibble.
struct QMatrix {
    WeightFormat format = WeightFormat::F16;
    int rows = 0;
    int cols = 0;
    size_t rowBytes = 0;
    std::vector<uint8_t> data;

    bool empty() const { return data.empty(); }
    size_t bytes() const { return data.size(); }
    const uint8_t* row(int r) const { return data.data() + size_t(r) * rowBytes; }
    const uint16_t* f16() const { return reinterpret_cast<const uint16_t*>(data.data()); }

    void appendRows(const QMatrix& m);  // same format and width
    QMatrix sliceRows(int r0, int n) const;
    void dequantizeRow(int r, float* out) const;  // cols floats
};

// Builds a matrix in format f from rows produced by read(row, out) (cols floats each; must be safe to call from
// several threads). FP8/FP4 need cols % 32 == 0; other widths are stored as F16. Rows are converted in parallel
// when a pool is given.
QMatrix quantizeMatrix(WeightFormat f, int rows, int cols, const std::function<void(int, float*)>& read,
                       ThreadPool* pool);

}  // namespace neko
