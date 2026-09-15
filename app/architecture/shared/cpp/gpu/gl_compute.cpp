// OpenGL ES 3.1 compute backend with a headless EGL context.
// The context is current on the creating thread: all calls must come from the inference thread.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <unistd.h>

#include <cstdlib>
#include <vector>

#include "../common.h"
#include "compute.h"
#include "shaders_gen.h"

namespace neko::gpu {

namespace {

struct GlBuf : Buffer {
    GLuint id = 0;
};

class GlesBackend final : public ComputeBackend {
public:
    GlesBackend() { init(); }
    ~GlesBackend() override { destroy(); }

    const char* name() const override { return "OpenGL ES"; }
    std::string deviceName() const override { return deviceName_; }
    size_t maxBufferBytes() const override { return maxBuffer_; }

    Buffer* create(size_t bytes, const void* init, bool) override {
        auto b = std::make_unique<GlBuf>();
        b->size = bytes;
        glGenBuffers(1, &b->id);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->id);
        glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(std::max<size_t>(bytes, 16)), init,
                     init ? GL_STATIC_DRAW : GL_DYNAMIC_COPY);
        if (glGetError() != GL_NO_ERROR) fail("GLES: buffer allocation failed (" + std::to_string(bytes) + " bytes)");
        buffers_.push_back(std::move(b));
        return buffers_.back().get();
    }

    void upload(Buffer* buf, size_t offset, const void* src, size_t bytes) override {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, static_cast<GlBuf*>(buf)->id);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, GLintptr(offset), GLsizeiptr(bytes), src);
    }

    void download(Buffer* buf, size_t offset, void* dst, size_t bytes) override {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, static_cast<GlBuf*>(buf)->id);
        void* p = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, GLintptr(offset), GLsizeiptr(bytes), GL_MAP_READ_BIT);
        if (!p) fail("GLES: glMapBufferRange failed");
        std::memcpy(dst, p, bytes);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    void begin() override {}

    void dispatch(Kernel k, Buffer* const bindings[4], const int32_t params[8], uint32_t gx, uint32_t gy) override {
        const Program& p = programs_[int(k)];
        glUseProgram(p.id);
        for (int i = 0; i < 4; i++) {
            auto* b = static_cast<GlBuf*>(bindings[i] ? bindings[i] : dummy_);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLuint(i), b->id);
        }
        glUniform4iv(p.pa, 1, params);
        glUniform4iv(p.pb, 1, params + 4);
        glDispatchCompute(gx, gy, 1);
        // GL_SHADER_STORAGE_BARRIER_BIT alone is not honoured by some Mali drivers for back-to-back
        // dispatches (verified on Mali-G72 r38p1); the full barrier fixes it at no measurable cost.
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        if (syncEach_) glFinish();  // debugging aid: NEKO_GL_SYNC=1
    }

    // Runs the work so far and waits for it (see VulkanBackend::flush): with one layer in flight, frames from
    // the UI and the keyboard get the GPU between layers instead of after the whole token.
    void flush() override {
        pending_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();
        dropPending(true);
    }

    void submitAndWait() override {
        pending_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();
        dropPending(true);
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) fail("GLES error " + std::to_string(err));
    }

private:
    struct Program {
        GLuint id = 0;
        GLint pa = -1, pb = -1;
    };

    // Polls instead of blocking in glClientWaitSync/glFinish: the UI renders through the same driver in this
    // process, and a thread parked inside the driver's wait can hold up the RenderThread's frames.
    void dropPending(bool wait) {
        if (!pending_) return;
        if (wait) {
            for (;;) {
                GLenum r = glClientWaitSync(pending_, 0, 0);
                if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED || r == GL_WAIT_FAILED) break;
                usleep(200);
            }
        }
        glDeleteSync(pending_);
        pending_ = nullptr;
    }

    GLsync pending_ = nullptr;

    void init() {
        display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display_ == EGL_NO_DISPLAY || !eglInitialize(display_, nullptr, nullptr)) fail("EGL init failed");
        eglBindAPI(EGL_OPENGL_ES_API);
        const EGLint cfgAttr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                  EGL_NONE};
        EGLConfig cfg;
        EGLint n = 0;
        if (!eglChooseConfig(display_, cfgAttr, &cfg, 1, &n) || n == 0) fail("EGL: no ES3 config");
        // Low context priority (where the driver offers it) lets the UI's and the keyboard's GPU work go first.
        const char* ext = eglQueryString(display_, EGL_EXTENSIONS);
        const bool lowPriority = ext && std::strstr(ext, "EGL_IMG_context_priority");
        const EGLint ctxAttr[] = {EGL_CONTEXT_MAJOR_VERSION_KHR, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 1,
                                  lowPriority ? EGL_CONTEXT_PRIORITY_LEVEL_IMG : EGL_NONE, EGL_CONTEXT_PRIORITY_LOW_IMG,
                                  EGL_NONE};
        context_ = eglCreateContext(display_, cfg, EGL_NO_CONTEXT, ctxAttr);
        if (context_ == EGL_NO_CONTEXT) fail("EGL: cannot create ES 3.1 context");
        const EGLint pbAttr[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        surface_ = eglCreatePbufferSurface(display_, cfg, pbAttr);
        if (!eglMakeCurrent(display_, surface_, surface_, context_)) fail("EGL: makeCurrent failed");

        GLint major = 0, minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);
        if (major * 10 + minor < 31) fail("OpenGL ES 3.1 compute not supported");
        const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        deviceName_ = renderer ? renderer : "GLES";
        GLint64 maxBlock = 0;
        glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlock);
        maxBuffer_ = size_t(maxBlock);
        GLint maxSsbo = 0;
        glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxSsbo);
        if (maxSsbo < 4) fail("GLES: fewer than 4 compute SSBOs");

        for (int k = 0; k < int(Kernel::Count); k++) programs_[k] = compile(k);
        dummy_ = create(16, nullptr, false);
        NEKO_LOGI("GLES backend on %s (max SSBO %zu MB)", deviceName_.c_str(), maxBuffer_ >> 20);
    }

    Program compile(int k) {
        std::string src = std::string("#version 310 es\n") + kShaderGlsl[k];
        const char* s = src.c_str();
        GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(sh, 1, &s, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetShaderInfoLog(sh, sizeof log, nullptr, log);
            glDeleteShader(sh);
            fail(std::string("GLES compile ") + kShaderNames[k] + ": " + log);
        }
        Program p;
        p.id = glCreateProgram();
        glAttachShader(p.id, sh);
        glLinkProgram(p.id);
        glDeleteShader(sh);
        glGetProgramiv(p.id, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(p.id, sizeof log, nullptr, log);
            fail(std::string("GLES link ") + kShaderNames[k] + ": " + log);
        }
        p.pa = glGetUniformLocation(p.id, "pa");
        p.pb = glGetUniformLocation(p.id, "pb");
        return p;
    }

    void destroy() {
        if (display_ == EGL_NO_DISPLAY) return;
        if (context_ != EGL_NO_CONTEXT) {
            eglMakeCurrent(display_, surface_, surface_, context_);
            for (auto& b : buffers_) glDeleteBuffers(1, &b->id);
            for (auto& p : programs_)
                if (p.id) glDeleteProgram(p.id);
            eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(display_, context_);
        }
        if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
        buffers_.clear();
    }

    const bool syncEach_ = std::getenv("NEKO_GL_SYNC") != nullptr;    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;
    Program programs_[int(Kernel::Count)];
    std::vector<std::unique_ptr<GlBuf>> buffers_;
    Buffer* dummy_ = nullptr;
    std::string deviceName_;
    size_t maxBuffer_ = 0;
};

}  // namespace

std::unique_ptr<ComputeBackend> createGles(std::string& error) {
    try {
        return std::make_unique<GlesBackend>();
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

}  // namespace neko::gpu
