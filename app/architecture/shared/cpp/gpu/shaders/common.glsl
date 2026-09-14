// Shared preamble. The build prepends "#version 450 + NEKO_VULKAN" for SPIR-V, GLES prepends "#version 310 es".
// Every kernel takes two ivec4 parameter blocks (floats are passed as raw bits) and up to 4 SSBOs.
#ifdef NEKO_VULKAN
#define BUF(n) layout(std430, set = 0, binding = n)
layout(push_constant) uniform Params { ivec4 pa; ivec4 pb; } params;
#define PA params.pa
#define PB params.pb
#else
precision highp float;
precision highp int;
#define BUF(n) layout(std430, binding = n)
uniform ivec4 pa;
uniform ivec4 pb;
#define PA pa
#define PB pb
#endif

// Weights are f16 packed two per uint.
float halfAt(uint word, uint element) {
    vec2 v = unpackHalf2x16(word);
    return (element & 1u) == 0u ? v.x : v.y;
}
