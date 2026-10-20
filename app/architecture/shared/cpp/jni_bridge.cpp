// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

// JNI surface for com.nekochat.engine.NativeBridge. Every call except nativeCancel must come from
// the Kotlin inference thread (GPU contexts are bound to it).
#include <jni.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "engine.h"
#include "granite_speech_model.h"
#include "kokoro_g2p.h"
#include "kokoro_model.h"

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
                                                                         jint threads, jint weights, jobject progress) {
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
        neko::LoadOptions opt;
        opt.backend = static_cast<neko::BackendPref>(backend);
        opt.threads = threads;
        opt.weights = static_cast<neko::WeightFormat>(weights);
        auto* e = new Engine(jstr(env, dir), opt, report);
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

// Returns [weights, kvUsed, kvCapacity, kvResident, workspace] in bytes. Callable from any thread.
JNIEXPORT jlongArray JNICALL Java_com_nekochat_engine_NativeBridge_nativeMemory(JNIEnv* env, jclass, jlong h) {
    neko::MemoryStats m = engine(h)->memory();
    jlong v[5] = {jlong(m.weights), jlong(m.kvUsed), jlong(m.kvCapacity), jlong(m.kvResident), jlong(m.workspace)};
    jlongArray out = env->NewLongArray(5);
    env->SetLongArrayRegion(out, 0, 5, v);
    return out;
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

// ------------------------------------------------------------------ text-to-speech (Kokoro)

struct Tts {
    std::unique_ptr<neko::KokoroModel> model;
    std::unique_ptr<neko::KokoroG2P> us, gb;
    std::string lexiconDir;
    std::map<std::string, std::unique_ptr<neko::KokoroVoice>> voices;
};

static Tts* tts(jlong h) { return reinterpret_cast<Tts*>(h); }

JNIEXPORT jlong JNICALL Java_com_nekochat_engine_NativeBridge_nativeTtsLoad(JNIEnv* env, jclass, jstring modelDir,
                                                                            jstring lexiconDir, jint weights,
                                                                            jint threads) {
    try {
        auto t = std::make_unique<Tts>();
        neko::LoadOptions opt;
        opt.backend = neko::BackendPref::CPU;
        opt.threads = threads;
        opt.weights = static_cast<neko::WeightFormat>(weights);
        t->model = std::make_unique<neko::KokoroModel>(jstr(env, modelDir), opt, [](float, const std::string&) {});
        t->lexiconDir = jstr(env, lexiconDir);
        return reinterpret_cast<jlong>(t.release());
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_nekochat_engine_NativeBridge_nativeTtsRelease(JNIEnv*, jclass, jlong h) {
    delete tts(h);
}

JNIEXPORT jlong JNICALL Java_com_nekochat_engine_NativeBridge_nativeTtsWeightBytes(JNIEnv*, jclass, jlong h) {
    return jlong(tts(h)->model->weightBytes());
}

// English text -> phonemes; the US or UK lexicon is loaded on first use.
JNIEXPORT jstring JNICALL Java_com_nekochat_engine_NativeBridge_nativeTtsPhonemize(JNIEnv* env, jclass, jlong h,
                                                                                  jbyteArray utf8, jboolean british) {
    try {
        Tts* t = tts(h);
        auto& g2p = british ? t->gb : t->us;
        if (!g2p) {
            const std::string p = t->lexiconDir + (british ? "/gb_" : "/us_");
            g2p = std::make_unique<neko::KokoroG2P>(p + "gold.json", p + "silver.json", british);
        }
        return env->NewStringUTF(g2p->phonemize(bytesToString(env, utf8)).c_str());
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return nullptr;
    }
}

// Phonemes (at most 510 of them) -> 24 kHz samples with the voice pack at voicePath.
JNIEXPORT jfloatArray JNICALL Java_com_nekochat_engine_NativeBridge_nativeTtsSynthesize(
    JNIEnv* env, jclass, jlong h, jbyteArray phonemes, jstring voicePath, jfloat speed, jlong seed) {
    try {
        Tts* t = tts(h);
        const std::string path = jstr(env, voicePath);
        auto& voice = t->voices[path];
        if (!voice) voice = std::make_unique<neko::KokoroVoice>(path);
        std::vector<int> ids;
        for (char32_t c : neko::utf8To32(bytesToString(env, phonemes))) {
            int id = t->model->phonemeId(neko::utf32To8(std::u32string(1, c)));
            if (id >= 0) ids.push_back(id);
            if (int(ids.size()) == neko::KokoroModel::kMaxPhonemes) break;
        }
        std::vector<float> pcm;
        if (!ids.empty()) pcm = t->model->synthesize(ids, voice->style(int(ids.size())), speed, uint64_t(seed));
        jfloatArray out = env->NewFloatArray(jsize(pcm.size()));
        if (!pcm.empty()) env->SetFloatArrayRegion(out, 0, jsize(pcm.size()), pcm.data());
        return out;
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return nullptr;
    }
}

// ------------------------------------------------------------------ speech to text (Granite Speech)

static neko::GraniteSpeechModel* asr(jlong h) { return reinterpret_cast<neko::GraniteSpeechModel*>(h); }

JNIEXPORT jlong JNICALL Java_com_nekochat_engine_NativeBridge_nativeAsrLoad(JNIEnv* env, jclass, jstring modelDir,
                                                                            jint weights, jint threads) {
    try {
        neko::LoadOptions opt;
        opt.backend = neko::BackendPref::CPU;
        opt.threads = threads;
        opt.weights = static_cast<neko::WeightFormat>(weights);
        auto m = std::make_unique<neko::GraniteSpeechModel>(jstr(env, modelDir), opt, [](float, const std::string&) {});
        return reinterpret_cast<jlong>(m.release());
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_nekochat_engine_NativeBridge_nativeAsrRelease(JNIEnv*, jclass, jlong h) {
    delete asr(h);
}

JNIEXPORT jlong JNICALL Java_com_nekochat_engine_NativeBridge_nativeAsrWeightBytes(JNIEnv*, jclass, jlong h) {
    return jlong(asr(h)->weightBytes());
}

// UTF-8 bytes rather than a jstring: NewStringUTF expects modified UTF-8, which mangles 4-byte characters.
JNIEXPORT jbyteArray JNICALL Java_com_nekochat_engine_NativeBridge_nativeAsrTranscribe(JNIEnv* env, jclass, jlong h,
                                                                                      jfloatArray pcm) {
    try {
        const jsize n = env->GetArrayLength(pcm);
        std::vector<float> samples(static_cast<size_t>(n));
        if (n > 0) env->GetFloatArrayRegion(pcm, 0, n, samples.data());
        const std::string text = asr(h)->transcribe(samples.data(), samples.size());
        jbyteArray out = env->NewByteArray(jsize(text.size()));
        if (!text.empty()) env->SetByteArrayRegion(out, 0, jsize(text.size()), reinterpret_cast<const jbyte*>(text.data()));
        return out;
    } catch (const std::exception& ex) {
        if (!env->ExceptionCheck()) throwJava(env, ex.what());
        return nullptr;
    }
}

}  // extern "C"
