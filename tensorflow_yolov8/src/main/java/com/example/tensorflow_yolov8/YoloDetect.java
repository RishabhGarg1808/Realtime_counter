package com.example.tensorflow_yolov8;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import com.example.tensorflow_yolov8.Utils.Utils;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
    boolean isQuantized;
    byte[][][] Array_def = new byte[1][84][3024];

    public YoloDetect(Interpreter tfLite, List<String> labels, int size,Yolov8Classfier d){
        this.tfLite = tfLite;
        this.labels = labels;
        this.size = size;

        this.INPUT_SIZE = size;

        intValues = new int[size * size];
        this.imgData = ByteBuffer.allocateDirect(this.INPUT_SIZE * this.INPUT_SIZE * 3 * d.numBytesPerChannel);
        this.imgData.order(ByteOrder.nativeOrder());

        output_Buffer_float.put(0, Array_def);
        isQuantized = d.isQuantized;
        this.inp_scale = d.inp_scale;
        this.inp_zero_point = d.inp_zero_point;
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
        tfLite.runForMultipleInputsOutputs(new Object[]{convertBitmapToByteBuffer(image)}, output_Buffer_float);
        getPredictions();
    }
    public void detect(TensorImage image){
        tfLite.runForMultipleInputsOutputs(new Object[]{image.getBuffer()}, output_Buffer_float);
        getPredictions();
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

    private void getPredictions() {

    }

    public void debug() {
//        Log.d("YoloDetect", "locations: " + locations);
//        Log.d("YoloDetect", "labelIndices: " + labelIndices);
//        Log.d("YoloDetect", "scores: " + scores);
        this.get_input_shape();
        this.getOutputShape();
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