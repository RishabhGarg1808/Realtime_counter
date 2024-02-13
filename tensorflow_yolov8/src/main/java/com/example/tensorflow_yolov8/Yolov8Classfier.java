package com.example.tensorflow_yolov8;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import com.example.tensorflow_yolov8.Utils.Utils;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.MappedByteBuffer;

public class Yolov8Classfier {
     CompatibilityList compatList = new CompatibilityList();
     boolean isNNAPI = false;
     boolean isGPU = false;
     boolean isQuantized = false;
     int NUM_THREADS = 4;
     AssetManager assetManager = null;
     String modelFilename ;
     String labelFilename = "labels.txt";
     int inputSize ;
     Interpreter tfLite = null;
    private MappedByteBuffer tfliteModel;

    GpuDelegate gpuDelegate = null;
    /**
     * holds an nnapi delegate
     */
    NnApiDelegate nnapiDelegate = null;

     float inp_scale;
     int inp_zero_point;
    float oup_scale;
    int oup_zero_point;

    int numBytesPerChannel ;
    public Yolov8Classfier() {
    }

    public  void create(AssetManager assetManager, String modelFilename, String labelFilename, int inputSize,int NUM_THREADS){
        this.assetManager = assetManager;
        this.modelFilename = modelFilename;
        this.labelFilename = labelFilename;
        this.inputSize = inputSize;

        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                this.nnapiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    this.nnapiDelegate = new NnApiDelegate();
                    options.addDelegate(this.nnapiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                Log.d("Yolov8Classfier", "++++++++++++++++++++++++++++++create: Trying to create GPU Delegate +++++++++++++++++++++++++++++++++++++++++");
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                delegateOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
                this.gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(this.gpuDelegate);
            }
            this.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            this.tfLite = new Interpreter(this.tfliteModel, options);

            if(this.tfLite == null){
                Log.d("Yolov8Classfier", "create: tfLite is null");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if(this.isQuantized){
            numBytesPerChannel = 4;
            Log.d("Yolov8Classfier", "create: Model is quantized");
            Tensor inpten = this.tfLite.getInputTensor(0);
            this.inp_scale = inpten.quantizationParams().getScale();
            this.inp_zero_point = inpten.quantizationParams().getZeroPoint();
            Tensor oupten = this.tfLite.getOutputTensor(0);
            this.oup_scale = oupten.quantizationParams().getScale();
            this.oup_zero_point = oupten.quantizationParams().getZeroPoint();
        }else{
            numBytesPerChannel = 4;
            Log.d("Yolov8Classfier", "create: Model is not quantized");
        }
    }
    public void useNNAPI(boolean isNNAPI) {
        this.isNNAPI = isNNAPI;
    }

    public void useGPU(boolean isGPU) {
        this.isGPU = isGPU;
    }
    public void setQuantized(boolean isQuantized) {
        this.isQuantized = isQuantized;
    }
    public boolean isQuantized() {
        return isQuantized;
    }

    public void setNUM_THREADS(int NUM_THREADS) {
        this.NUM_THREADS = NUM_THREADS;
    }
    public void close() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnapiDelegate != null) {
            nnapiDelegate.close();
            nnapiDelegate = null;
        }
    }

    public void detect(Bitmap bitmap){
        //TensorImage image = Utils.img_process(bitmap, inputSize);
        if (labelFilename !=null && this.tfLite != null) {
            YoloDetect yolodetect = new YoloDetect(this.tfLite, Utils.readLabels(assetManager, labelFilename),inputSize,this);
            Log.d("Yolov8Classfier", "detect: Input image shape: " + yolodetect.convertBitmapToByteBuffer(Utils.processBitmap(bitmap, inputSize)).capacity() );
            yolodetect.get_input_shape();
            yolodetect.detect(Utils.processBitmap(bitmap, inputSize));
            //yolodetect.debug();
        }else{
            if (labelFilename == null) {
                Log.d("Yolov8Classfier", "detect: labelFilename is null");
            }
            if (tfLite == null) {
                Log.d("Yolov8Classfier", "detect: tfLite is null");
            }
        }
    }

}
