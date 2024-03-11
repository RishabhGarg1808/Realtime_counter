package com.example.tensorflow_yolov8;

import android.graphics.Bitmap;
import android.util.Log;


import com.example.tensorflow_yolov8.Utils.BoundingBox;
import com.example.tensorflow_yolov8.Detection_factory;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
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
    int OBJECT_COUNT = 84;
    private float[][][] locations = new float[1][OBJECT_COUNT][3024];
    private float[][] labelIndices = new float[1][OBJECT_COUNT];
    private float[][] scores = new float[1][OBJECT_COUNT];
    boolean isQuantized;
    float[][][] Array_def = new float[1][84][3024];
    private List<BoundingBox> Box;
    int numElements = 3024;
    int numChannel = 84;
    float CONFIDENCE_THRESHOLD = 0.75f;
    float IOU_THRESHOLD =0.75f;
    int image_width =0;
    int image_height =0;

    public YoloDetect(Interpreter tfLite, List<String> labels, int size,Yolov8Classfier d){
        this.tfLite = tfLite;
        this.labels = labels;
        this.size = size;
        this.INPUT_SIZE = size;

        intValues = new int[size * size];
        this.imgData = ByteBuffer.allocateDirect(this.INPUT_SIZE * this.INPUT_SIZE * 3 * d.numBytesPerChannel);
        this.imgData.order(ByteOrder.nativeOrder());
        this.image_width = d.bitmap.getWidth();
        this.image_height = d.bitmap.getHeight();
        outputBuffer.put(0, Array_def);
        isQuantized = d.isQuantized;
        this.inp_scale = d.inp_scale;
        this.inp_zero_point = d.inp_zero_point;
        this.output_Buffer_float.put(0 , locations);
        Detection_factory.tensor_shape= size;
        Detection_factory.img_shape = new int[]{image_width,image_height};
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


//    public List<BoundingBox> detect(Bitmap image) {
//        TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, numChannel, 8400}, DataType.FLOAT32);
//        tfLite.run(convertBitmapToByteBuffer(image),output.getBuffer());
//        return (getPredictions(output.getFloatArray()));
//    }
    public List<BoundingBox> detect(TensorImage image){
        tfLite.run(image.getBuffer(), Array_def);
        return getPredictions(Array_def);
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

   private List<BoundingBox> getPredictions(float[][][] output){
        float[][] trans_output = Detection_factory.getTranspose(output);
        ArrayList<float[]> boxes = new ArrayList<>();

        for (float[] floats : trans_output) {
           for (int j = 4; j < floats.length;j++){
               if (floats[j] >= CONFIDENCE_THRESHOLD){
                   float[] temporaryArray = {
                           floats[0], floats[1], floats[2], floats[3], j-4, floats[j]
                   };
                   boxes.add(temporaryArray);
               }
           }
       }
        return Detection_factory.getBoundingBox(boxes,labels,IOU_THRESHOLD);
   }

    public void debug() {

        Log.d("YoloDetect", "Input shape: " + Arrays.toString(tfLite.getInputTensor(0).shape()));
        Log.d("YoloDetect", "Output shape: " + Arrays.toString(tfLite.getOutputTensor(0).shape()));
        Log.d("YoloDetect", "Output shape: " + Arrays.toString(tfLite.getOutputTensor(0).shape()));
    }



}