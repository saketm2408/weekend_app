/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <android/log.h>
#include <jni.h>

#include <cinttypes>
#include <cstring>
#include <string>

#include "LowLightEnhancement.h"

namespace tflite {
namespace examples {
namespace superresolution {

extern "C" JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_LowLightEnhancementFromJNI(
    JNIEnv *env, jobject thiz, jlong native_handle, jintArray low_light_rgb) {

  __android_log_print(ANDROID_LOG_INFO, "tflite ", " inside superResolutionFromJNI 1");
  jint *low_light_img_rgb = env->GetIntArrayElements(low_light_rgb, NULL);

  __android_log_print(ANDROID_LOG_INFO, "tflite ", " inside superResolutionFromJNI 2");
  if (!reinterpret_cast<LowLightEnhancement *>(native_handle)
           ->IsInterpreterCreated()) {
    return nullptr;
  }

  // Generate low light enhanced image
  auto sr_rgb_colors = reinterpret_cast<LowLightEnhancement *>(native_handle)
          ->DoLowLightEnhancement(static_cast<int *>(low_light_img_rgb));
  __android_log_print(ANDROID_LOG_INFO, "tflite ", " inside superResolutionFromJNI 3");
  if (!sr_rgb_colors) {
    return nullptr;  // super resolution failed
  }
  // Create jintArray to pass through jni
  jintArray enhanced_img_rgb = env->NewIntArray(kNumberOfOutputPixels);
  env->SetIntArrayRegion(enhanced_img_rgb, 0, kNumberOfOutputPixels,
                         sr_rgb_colors.get());

    __android_log_print(ANDROID_LOG_INFO, "tflite ", " inside superResolutionFromJNI 4");

  // Clean up before we return
  env->ReleaseIntArrayElements(low_light_rgb, low_light_img_rgb, JNI_COMMIT);
  
  return enhanced_img_rgb;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_initWithByteBufferFromJNI(
    JNIEnv *env, jobject thiz, jobject model_buffer, jboolean use_gpu) {
  
  const void *model_data =
      static_cast<void *>(env->GetDirectBufferAddress(model_buffer));
  
  jlong model_size_bytes = env->GetDirectBufferCapacity(model_buffer);
  
  LowLightEnhancement *low_light_enhancement_obj = new LowLightEnhancement(
      model_data, static_cast<size_t>(model_size_bytes), use_gpu);
  
  if (low_light_enhancement_obj->IsInterpreterCreated()) {
    LOGI("Interpreter is created successfully");
    return reinterpret_cast<jlong>(low_light_enhancement_obj);
  } else {
    delete low_light_enhancement_obj;
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_deinitFromJNI(
    JNIEnv *env, jobject thiz, jlong native_handle) {
  delete reinterpret_cast<LowLightEnhancement*>(native_handle);
}

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
