package com.example.tensorflow_yolov8;

import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.DelegateFactory;

import java.nio.MappedByteBuffer;
import java.util.List;

import com.example.tensorflow_yolov8.Utils.Utils;
public class Yolov8Classfier {
    List<Boolean> list_bool;
    CompatibilityList compatList = new CompatibilityList();
    static boolean isNNAPI = false;
     static boolean isGPU = false;
     int NUM_THREADS = 4;
     static AssetManager assetManager = null;
     String modelFilename ;
     String labelFilename;
     int inputSize ;
     Interpreter tfLite;
    private MappedByteBuffer tfliteModel;

    GpuDelegate gpuDelegate = null;
    /**
     * holds an nnapi delegate
     */
    NnApiDelegate nnapiDelegate = null;

    public Yolov8Classfier() {
    }

    public  Yolov8Classfier create(AssetManager assetManager, String modelFilename, String labelFilename, int inputSize,List<Boolean> list_bool,int NUM_THREADS){
        final Yolov8Classfier d = new Yolov8Classfier();
        Yolov8Classfier.assetManager = assetManager;
        d.modelFilename = modelFilename;
        d.labelFilename = labelFilename;
        d.inputSize = inputSize;
        d.list_bool = list_bool;

        isNNAPI = list_bool.get(0);
        isGPU = list_bool.get(1);


        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                d.nnapiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    d.nnapiDelegate = new NnApiDelegate();
                    options.addDelegate(d.nnapiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                Log.d("Yolov8Classfier", "++++++++++++++++++++++++++++++create: Trying to create GPU Delegate +++++++++++++++++++++++++++++++++++++++++");
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                delegateOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
                d.gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(d.gpuDelegate);
            }
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return d;
    }

}
