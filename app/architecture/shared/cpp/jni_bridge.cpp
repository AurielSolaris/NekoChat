// JNI surface for com.nekochat.engine.NativeBridge. Every call except nativeCancel must come from
// the Kotlin inference thread (GPU contexts are bound to it).
#include <jni.h>

#include <string>
#include <vector>

#include "common.h"
#include "engine.h"

using neko::Engine;

namespace {

void throwJava(JNIEnv* env, const char* msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(cls, msg);
}

std::string bytesToString(JNIEnv* env, jbyteArray arr) {
    jsize n = env->GetArrayLength(arr);
    std::string s(size_t(n), '\0');
    if (n > 0) env->GetByteArrayRegion(arr, 0, n, reinterpret_cast<jbyte*>(&s[0]));
    return s;
}

std::string jstr(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    env->ReleaseStringUTFChars(s, c);
    return out;
}

Engine* engine(jlong h) { return reinterpret_cast<Engine*>(h); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nekochat_engine_NativeBridge_nativeLoad(JNIEnv* env, jclass, jstring dir, jint backend,
                                                                         jint threads, jobject progress) {
    jmethodID onProgress = nullptr;
    if (progress) {
        jclass cls = env->GetObjectClass(progress);
        onProgress = env->GetMethodID(cls, "onProgress", "(FLjava/lang/String;)V");
    }
    auto report = [&](float f, const std::string& stage) {
        if (!onProgress || env->ExceptionCheck()) return;
        jstring s = env->NewStringUTF(stage.c_str());
        env->CallVoidMethod(progress, onProgress, jfloat(f), s);
        env->DeleteLocalRef(s);
    };
    try {
        auto* e = new Engine(jstr(env, dir), static_cast<neko::BackendPref>(backend), threads, report);
        return reinterpret_cast<jlong>(e);
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_nekochat_engine_NativeBridge_nativeRelease(JNIEnv*, jclass, jlong h) {
    delete engine(h);
}

JNIEXPORT jstring JNICALL Java_com_nekochat_engine_NativeBridge_nativeInfo(JNIEnv* env, jclass, jlong h) {
    return env->NewStringUTF(engine(h)->infoJson().c_str());
}

JNIEXPORT jint JNICALL Java_com_nekochat_engine_NativeBridge_nativeContextLength(JNIEnv*, jclass, jlong h) {
    return engine(h)->contextLength();
}

JNIEXPORT jintArray JNICALL Java_com_nekochat_engine_NativeBridge_nativeTokenize(JNIEnv* env, jclass, jlong h,
                                                                                jbyteArray utf8) {
    std::vector<int> ids = engine(h)->tokenize(bytesToString(env, utf8));
    jintArray out = env->NewIntArray(jsize(ids.size()));
    env->SetIntArrayRegion(out, 0, jsize(ids.size()), ids.data());
    return out;
}

JNIEXPORT void JNICALL Java_com_nekochat_engine_NativeBridge_nativeCancel(JNIEnv*, jclass, jlong h) {
    engine(h)->cancel();
}

JNIEXPORT void JNICALL Java_com_nekochat_engine_NativeBridge_nativeResetCache(JNIEnv*, jclass, jlong h) {
    engine(h)->resetCache();
}

// Returns [promptTokens, reusedTokens, generatedTokens, prefillMs, decodeMs, stopReason].
JNIEXPORT jdoubleArray JNICALL Java_com_nekochat_engine_NativeBridge_nativeGenerate(
    JNIEnv* env, jclass, jlong h, jintArray prompt, jint maxNew, jfloat temperature, jint topK, jfloat topP,
    jfloat repeatPenalty, jint repeatLastN, jlong seed, jobject sink) {
    jclass cls = env->GetObjectClass(sink);
    jmethodID onText = env->GetMethodID(cls, "onText", "([B)Z");
    std::vector<int> ids(size_t(env->GetArrayLength(prompt)));
    if (!ids.empty()) env->GetIntArrayRegion(prompt, 0, jsize(ids.size()), ids.data());

    neko::SamplingParams sp;
    sp.temperature = temperature;
    sp.topK = topK;
    sp.topP = topP;
    sp.repeatPenalty = repeatPenalty;
    sp.repeatLastN = repeatLastN;
    sp.seed = uint64_t(seed);
    try {
        neko::GenerateStats st = engine(h)->generate(ids, maxNew, sp, [&](const std::string& piece) {
            jbyteArray b = env->NewByteArray(jsize(piece.size()));
            env->SetByteArrayRegion(b, 0, jsize(piece.size()), reinterpret_cast<const jbyte*>(piece.data()));
            jboolean keep = env->CallBooleanMethod(sink, onText, b);
            env->DeleteLocalRef(b);
            return !env->ExceptionCheck() && keep;
        });
        double v[6] = {double(st.promptTokens), double(st.reusedTokens), double(st.generatedTokens),
                       st.prefillMs,  st.decodeMs, double(st.stopReason)};
        jdoubleArray out = env->NewDoubleArray(6);
        env->SetDoubleArrayRegion(out, 0, 6, v);
        return out;
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return nullptr;
    }
}

}  // extern "C"
