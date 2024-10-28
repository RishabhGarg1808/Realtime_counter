```markdown
# Real-Time Object Counter

This project is a real-time object counter application using TensorFlow Lite and YOLOv8. It leverages NNAPI and GPU delegates for optimized performance on supported devices.

## Features

- Real-time object detection and counting
- Supports NNAPI and GPU delegates for enhanced performance
- Fallback to CPU execution if delegates fail
- Configurable confidence and IOU thresholds
- FPS counter for performance monitoring

## Requirements

- Android device with Android 9.0 (Pie) or higher
- TensorFlow Lite model compatible with YOLOv8

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/realtime-object-counter.git
    cd realtime-object-counter
    ```

2. **Open the project in Android Studio:**

    - Open Android Studio.
    - Select `File > Open` and navigate to the project directory.

3. **Build the project:**

    - Click on `Build > Make Project` or press `Ctrl+F9`.

4. **Run the application:**

    - Connect your Android device.
    - Click on `Run > Run 'app'` or press `Shift+F10`.

## Configuration

You can configure the following parameters in the `Yolov8Classifier` class:

- **Number of Threads:**
    ```java
    classifier.setNUM_THREADS(8);
    ```

- **Use NNAPI:**
    ```java
    classifier.useNNAPI(true);
    ```

- **Use GPU:**
    ```java
    classifier.useGPU(true);
    ```

- **Confidence Threshold:**
    ```java
    classifier.setCONFIDENCE_THRESHOLD(0.75f);
    ```

- **IOU Threshold:**
    ```java
    classifier.setIOU_THRESHOLD(0.75f);
    ```

## Troubleshooting

- **NNAPI Delegate Errors:**
    - Ensure your device supports NNAPI and is running Android 9.0 or higher.
    - Verify that the TensorFlow Lite model is compatible with NNAPI.

- **Fallback to CPU:**
    - If NNAPI or GPU delegates fail, the application will automatically fallback to CPU execution.

## License

This project is licensed under the GNU GPL v3 License.
```
