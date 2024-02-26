package com.example.tensorflow_yolov8;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;


import com.example.tensorflow_yolov8.Utils.BoundingBox;
import com.example.tensorflow_yolov8.Utils.DetectedItem;

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

    private final Interpreter tfLite;
    private final List<String> labels;

    private int size;
    private int INPUT_SIZE = -1;
    private final ByteBuffer imgData;
    private final int[] intValues ;
    private final Map<Integer, Object> outputBuffer = new HashMap<>();
    private final float inp_scale;
    private final int inp_zero_point;
    Map<Integer, Object> output_Buffer_float = new HashMap<>();
    int OBJECT_COUNT = 10;
    private float[][][] locations = new float[1][OBJECT_COUNT][4];
    private float[][] labelIndices = new float[1][OBJECT_COUNT];
    private float[][] scores = new float[1][OBJECT_COUNT];
    boolean isQuantized;
    float[][][] Array_def = new float[1][84][3024];
    private List<BoundingBox> Box;
    public YoloDetect(Interpreter tfLite, List<String> labels, int size,Yolov8Classfier d){
        this.tfLite = tfLite;
        this.labels = labels;
        this.size = size;

        this.INPUT_SIZE = size;

        intValues = new int[size * size];
        this.imgData = ByteBuffer.allocateDirect(this.INPUT_SIZE * this.INPUT_SIZE * 3 * d.numBytesPerChannel);
        this.imgData.order(ByteOrder.nativeOrder());

        outputBuffer.put(0, Array_def);
        isQuantized = d.isQuantized;
        this.inp_scale = d.inp_scale;
        this.inp_zero_point = d.inp_zero_point;
        this.output_Buffer_float.put(0 , locations);
        this.output_Buffer_float.put(1,labelIndices);
        this.output_Buffer_float.put(2,scores);
        this.output_Buffer_float.put(3,new float[1]);
    }

    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                float IMAGE_MEAN = 0;
                float IMAGE_STD = 255.0f;
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


    public void detect(Bitmap image) {
        tfLite.runForMultipleInputsOutputs(new Object[]{convertBitmapToByteBuffer(image)}, outputBuffer);
        getPredictions();
    }
    public List<DetectedItem> detect(TensorImage image){
        tfLite.runForMultipleInputsOutputs(new Object[]{image.getBuffer()},output_Buffer_float);
        return (getPredictions());
    }
    public void get_input_shape(){
        int[] inputShape = tfLite.getInputTensor(0).shape();
        Log.d("YoloDetect", "Input shape: " + Arrays.toString(inputShape));
        Log.d("YoloDetect", "Input Resolution : " + inputShape[1] + "x" + inputShape[2]);
    }

    public void getOutputShape() {
        int [] outputShape = tfLite.getOutputTensor(0).shape();
        Log.d("YoloDetect", "Output shape: " + Arrays.toString(outputShape));
    }

    private List<DetectedItem> getPredictions() {

        // Initialize list to store detected items
        List<DetectedItem> detectedItems = new ArrayList<>();

        // Loop over all detected objects
        for (int i = 0; i < OBJECT_COUNT; i++) {
            // Get label index and score for current object
            int labelIndex = (int) labelIndices[0][i];
            float score = scores[0][i];

            // If score is zero, there are no more objects
            if (score == 0) {
                break;
            }

            // Get location for current object
            float[] location = locations[0][i];

            // Create RectF object for bounding box
            RectF boundingBox = new RectF(location[0], location[1], location[2], location[3]);

            // Check if labelIndex is within the bounds of the labels list
            if (labelIndex < labels.size()) {
                // Get label for current object
                String label = labels.get(labelIndex);

                // Create DetectedItem object and add it to list
                DetectedItem item = new DetectedItem(boundingBox, label, score);
                detectedItems.add(item);
            }
        }

        return detectedItems;
    }

    public void debug() {

        Log.d("YoloDetect", "Input shape: " + Arrays.toString(tfLite.getInputTensor(0).shape()));
        Log.d("YoloDetect", "Output shape: " + Arrays.toString(tfLite.getOutputTensor(0).shape()));
        Log.d("YoloDetect", "Output shape: " + Arrays.toString(tfLite.getOutputTensor(0).shape()));
    }



}