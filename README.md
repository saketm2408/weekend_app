# Low Light Enhancement Android sample.

The task of recovering a photo taken in suboptimal lighting conditions.

The model used here is ZeroDCE

TFLite JAR files and uses TFLite C API
through Android NDK.

## Requirements

*   Android Studio 3.2 (installed on a Linux, Mac or Windows machine)
*   An Android device, or an Android Emulator

## Build and run

### Step 1. Clone this repository

### Step 2. Import the sample app to Android Studio

### Step 3. Download TFLite library

Open your terminal and go to the sample folder. Type './gradlew fetchTFLiteLibs'
to run the download tasks. Use 'gradlew.bat' on Windows.

### Step 4. Run the Android app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

To test the app, open the app called `TFL Low Light Enhancement` on your device.
Re-installing the app may require you to uninstall the previous installations.


## Resources used:

*   [TensorFlow Lite](https://www.tensorflow.org/lite)
