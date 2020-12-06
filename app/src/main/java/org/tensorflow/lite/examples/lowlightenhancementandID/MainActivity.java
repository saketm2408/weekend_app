/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.lowlightenhancementandID;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.WorkerThread;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/** A super resolution class to generate super resolution images from low resolution images * */
public class MainActivity extends AppCompatActivity {
  static {
    System.loadLibrary("LowLightEnhancement");
  }

  private static final String TAG = "LowLightEnhancement";
  private static final String LOW_LIGHT_EN_MODEL_NAME = "zdce.tflite";
  private static final String FACE_ID_MODEL_NAME = "mobile_face_net.tflite";
  private static final int LL_IMAGE_HEIGHT = 400;
  private static final int LL_IMAGE_WIDTH = 600;
  private static final int OL_IMAGE_HEIGHT = LL_IMAGE_HEIGHT;
  private static final int OL_IMAGE_WIDTH = LL_IMAGE_WIDTH;
  private static final String LL_IMG_1 = "ll-1.jpg";
  private static final String LL_IMG_2 = "ll-2.jpg";
  private static final String LL_IMG_3 = "ll-3.jpg";

  private MappedByteBuffer lowLightEnModel;
  private MappedByteBuffer faceIDModel;
  private long lowLightEnhancementNativeHandle = 0;
  private long faceIDNativeHandle = 0;
  private Bitmap selectedLRBitmap = null;
  private boolean useGPU = false;

  private ImageView lowLightImageView1;
  private ImageView lowLightImageView2;
  private ImageView lowLightImageView3;
  private TextView selectedImageTextView;
  private Switch gpuSwitch;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    final Button lowLightEnhancementButton = findViewById(R.id.enhancement_button);
    lowLightImageView1 = findViewById(R.id.low_light_image_1);
    lowLightImageView2 = findViewById(R.id.low_light_image_2);
    lowLightImageView3 = findViewById(R.id.low_light_image_3);
    selectedImageTextView = findViewById(R.id.chosen_image_tv);
    gpuSwitch = findViewById(R.id.switch_use_gpu);

    ImageView[] lowLightImageViews = {lowLightImageView1, lowLightImageView2, lowLightImageView3};

    AssetManager assetManager = getAssets();
    try {
      InputStream inputStream1 = assetManager.open(LL_IMG_1);
      Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream1);
      lowLightImageView1.setImageBitmap(bitmap1);

      InputStream inputStream2 = assetManager.open(LL_IMG_2);
      Bitmap bitmap2 = BitmapFactory.decodeStream(inputStream2);
      lowLightImageView2.setImageBitmap(bitmap2);

      InputStream inputStream3 = assetManager.open(LL_IMG_3);
      Bitmap bitmap3 = BitmapFactory.decodeStream(inputStream3);
      lowLightImageView3.setImageBitmap(bitmap3);
    } catch (IOException e) {
      Log.e(TAG, "Failed to open an low light image");
    }

    for (ImageView iv : lowLightImageViews) {
      setLLImageViewListener(iv);
    }

    lowLightEnhancementButton.setOnClickListener(
        new View.OnClickListener() {
          @Override
          public void onClick(View view) {
            if (selectedLRBitmap == null) {
              Toast.makeText(
                      getApplicationContext(),
                      "Please choose one low light image",
                      Toast.LENGTH_LONG)
                  .show();
              return;
            }

            if (lowLightEnhancementNativeHandle == 0) {
                lowLightEnhancementNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked(), LOW_LIGHT_EN_MODEL_NAME);
            } else if (useGPU != gpuSwitch.isChecked()) {
              // We need to reinitialize interpreter when execution hardware is changed
              deinit();
              lowLightEnhancementNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked(), LOW_LIGHT_EN_MODEL_NAME);
            }
            if (faceIDNativeHandle == 0) {
              faceIDNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked(), FACE_ID_MODEL_NAME);
            } else if (useGPU != gpuSwitch.isChecked()) {
              // We need to reinitialize interpreter when execution hardware is changed
              deinit();
              faceIDNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked(), FACE_ID_MODEL_NAME);
            }
            useGPU = gpuSwitch.isChecked();
            if (lowLightEnhancementNativeHandle == 0 || faceIDNativeHandle == 0) {
              showToast("TFLite interpreter failed to create!");
              return;
            }

            int[] lowLightRGB = new int[LL_IMAGE_HEIGHT * LL_IMAGE_WIDTH];
            selectedLRBitmap.getPixels(
                lowLightRGB, 0, LL_IMAGE_WIDTH, 0, 0, LL_IMAGE_WIDTH, LL_IMAGE_HEIGHT);

            final long startTime = SystemClock.uptimeMillis();
            int[] enhancedRGB = doLowLightEnhancement(lowLightRGB);
            System.out.println("#####################################################################################");
            System.out.println(lowLightRGB);
            final long processingTimeMs = SystemClock.uptimeMillis() - startTime;
            if (enhancedRGB == null) {
              showToast("Super resolution failed!");
              return;
            }

            final LinearLayout resultLayout = findViewById(R.id.result_layout);
            final ImageView enhancedImageView = findViewById(R.id.enhanced_image);
            final ImageView lowLightImageView = findViewById(R.id.low_light_image);
            final TextView enhancedTextView = findViewById(R.id.enhanced_tv);
            final TextView lowLightTextView =
                findViewById(R.id.low_light_image_tv);
            final TextView logTextView = findViewById(R.id.log_view);

            // Force refreshing the ImageView
            enhancedImageView.setImageDrawable(null);
            Bitmap srImgBitmap =
                Bitmap.createBitmap(
                    enhancedRGB, OL_IMAGE_WIDTH, OL_IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
            enhancedImageView.setImageBitmap(srImgBitmap);
            lowLightImageView.setImageBitmap(selectedLRBitmap);
            resultLayout.setVisibility(View.VISIBLE);
            logTextView.setText("Inference time: " + processingTimeMs + "ms");
          }
        });
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    deinit();
  }

  private void setLLImageViewListener(ImageView iv) {
    iv.setOnTouchListener(
        new View.OnTouchListener() {
          @Override
          public boolean onTouch(View v, MotionEvent event) {
            if (v.equals(lowLightImageView1)) {
              selectedLRBitmap = ((BitmapDrawable) lowLightImageView1.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 1 ");
            } else if (v.equals(lowLightImageView2)) {
              selectedLRBitmap = ((BitmapDrawable) lowLightImageView2.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 2 ");
            } else if (v.equals(lowLightImageView3)) {
              selectedLRBitmap = ((BitmapDrawable) lowLightImageView3.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 3 ");
            }
            return false;
          }
        });
  }

  @WorkerThread
  public synchronized int[] doLowLightEnhancement(int[] lowLightRGB) {
    return LowLightEnhancementFromJNI(lowLightEnhancementNativeHandle, lowLightRGB);
  }

  private MappedByteBuffer loadModelFile(String modelName) throws IOException {
    try (AssetFileDescriptor fileDescriptor =
            AssetsUtil.getAssetFileDescriptorOrCached(getApplicationContext(), modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  private void showToast(String str) {
    Toast.makeText(getApplicationContext(), str, Toast.LENGTH_LONG).show();
  }

  private long initTFLiteInterpreter(boolean useGPU, String modelName) {
    try {
      if (modelName.equals(LOW_LIGHT_EN_MODEL_NAME))
        lowLightEnModel = loadModelFile(modelName);
      else
        faceIDModel = loadModelFile(modelName);
    } catch (IOException e) {
      Log.e(TAG, "Fail to load model", e);
    }
    return initWithByteBufferFromJNI(lowLightEnModel, useGPU);
  }

  private void deinit() {
    deinitFromJNI(lowLightEnhancementNativeHandle);
  }

  private native int[] LowLightEnhancementFromJNI(long lowLightEnhancementNativeHandle, int[] lowLightRGB);

  private native long initWithByteBufferFromJNI(MappedByteBuffer modelBuffer, boolean useGPU);

  private native void deinitFromJNI(long lowLightEnhancementNativeHandle);
}
