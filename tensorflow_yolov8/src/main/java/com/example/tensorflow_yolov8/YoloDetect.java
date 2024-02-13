package com.example.tensorflow_yolov8;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class YoloDetect {
    private static final int OBJECT_COUNT = 1;

    private Interpreter tfLite;
    private List<String> labels;

    private float[][] locations = new float[OBJECT_COUNT][84];
    private float[] labelIndices = new float[OBJECT_COUNT];
    private float[] scores = new float[OBJECT_COUNT];

    private int size;
    private int INPUT_SIZE = -1;
    private ByteBuffer imgData;
    private int[] intValues ;
    private Map<Integer, Object> outputBuffer = new HashMap<>();
    private final float IMAGE_MEAN = 0;
    private final float IMAGE_STD = 255.0f;

    private float inp_scale;
    private int inp_zero_point;
    Map<Integer, Object> output_Buffer = new HashMap<>();
    boolean isQuantized = false;
    float[][][] array1 = new float[1][84][4116];

    public YoloDetect(Interpreter tfLite, List<String> labels, int size,Yolov8Classfier d) {
        this.tfLite = tfLite;
        this.labels = labels;
        this.size = size;

        outputBuffer.put(0, locations);
        outputBuffer.put(1, labelIndices);
        outputBuffer.put(2, scores);
        outputBuffer.put(3, new float[1]);
        this.INPUT_SIZE = size;

        intValues = new int[size * size];
        this.imgData = ByteBuffer.allocateDirect(this.INPUT_SIZE * this.INPUT_SIZE * 3 * d.numBytesPerChannel);
        this.imgData.order(ByteOrder.nativeOrder());

        output_Buffer.put(0, array1);
        isQuantized = d.isQuantized;
        this.inp_scale = d.inp_scale;
        this.inp_zero_point = d.inp_zero_point;
    }

    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                if (isQuantized) {
                    // Quantized model
                    imgData.put((byte) ((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) ((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) (((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }


    public List<DetectedItem> detect(Bitmap image) {
        tfLite.runForMultipleInputsOutputs(new Object[]{convertBitmapToByteBuffer(image)},output_Buffer);
        return getPredictions();
    }
    public void get_input_shape(){
        int[] inputShape = tfLite.getInputTensor(0).shape();
        Log.d("YoloDetect", "Input shape: " + Arrays.toString(inputShape));
    }

    private List<DetectedItem> getPredictions() {
        List<DetectedItem> predictions = new ArrayList<>();
        for (int i = 0; i < OBJECT_COUNT; i++) {
            RectF location = new RectF(locations[i][1], locations[i][0], locations[i][3], locations[i][2]);
            String label = labels.get(1 + (int) labelIndices[i]);
            float confidence = scores[i];
            predictions.add(new DetectedItem(location, label, confidence));
        }
        return predictions;
    }

    public void debug() {
        Log.d("YoloDetect", "locations: " + locations);
        Log.d("YoloDetect", "labelIndices: " + labelIndices);
        Log.d("YoloDetect", "scores: " + scores);
    }

    public static class DetectedItem {
        private RectF location;
        private String label;
        private float confidence;

        public DetectedItem(RectF location, String label, float confidence) {
            this.location = location;
            this.label = label;
            this.confidence = confidence;
        }

        // getters and setters
    }

}