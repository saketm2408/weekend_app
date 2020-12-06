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

#ifndef NATIVE_LIBS_SUPERRESOLUTION_H
#define NATIVE_LIBS_SUPERRESOLUTION_H

#include <string>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#define LOG_TAG "super_resolution::"
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

namespace tflite {
namespace examples {
namespace superresolution {

const int kInputImageHeight = 400;
const int kInputImageWidth = 600;
const int kImageChannels = 3;
const int kNumberOfInputPixels = kInputImageHeight * kInputImageWidth;
const int kOutputImageHeight = kInputImageHeight;
const int kOutputImageWidth = kInputImageWidth;
const int kNumberOfOutputPixels = kOutputImageHeight * kOutputImageWidth;

class LowLightEnhancementAndFaceId {
 public:
  LowLightEnhancementAndFaceId(const void* model_data, size_t model_size, bool use_gpu);
  ~LowLightEnhancementAndFaceId();
  bool IsInterpreterCreated();
  // DoLowLightEnhancement() performs super resolution on a low resolution image. It
  // returns a valid pointer if successful and nullptr if unsuccessful.
  // lr_img_rgb: the pointer to the RGB array extracted from low resolution
  // image
  std::unique_ptr<int[]> DoLowLightEnhancement(int* lr_img_rgb);

    float* ComputeFaceEMb(int *img_rgb);

private:
  // TODO: use unique_ptr
  TfLiteInterpreter* interpreter_;
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreterOptions* options_ = nullptr;
  TfLiteDelegate* delegate_ = nullptr;

};

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
#endif
