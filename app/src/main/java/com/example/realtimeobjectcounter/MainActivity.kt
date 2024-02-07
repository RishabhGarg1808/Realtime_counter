package com.example.realtimeobjectcounter;

import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.view.TextureView
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.objcounter.CameraHandler
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

import com.example.tensorflow_yolov8.Yolov8Classfier
class MainActivity : AppCompatActivity() {

    private var play_services_flag = false
    private var standalone_flag = false
    private var cpu_flag = false
    lateinit var interpreter: Interpreter
    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onCreate(savedInstanceState: Bundle?) {

        if(checkPermission()){

            super.onCreate(savedInstanceState)
            setContentView(R.layout.activity_main)

//            play_services_flag = false
//            standalone_flag = false
//            cpu_flag = false
//
//            val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(this@MainActivity)
//            if(useGpuTask.isSuccessful){
//                Log.d("GPU", "=====================Play Services GPU Delegate Supported=====================")
//                play_services_flag = true
//            }else{
//                Log.d("GPU", "===================Play Services GPU Delegate Not supported===================")
//                play_services_flag = false
//            }
//
//            if (play_services_flag){
//                val interpreterTask = useGpuTask.continueWithTask { useGpuTask ->
//                    TfLite.initialize(
//                        this,
//                        TfLiteInitializationOptions.builder()
//                            .setEnableGpuDelegateSupport(useGpuTask.result)
//                            .build()
//                    )
//                }.addOnFailureListener { exception ->
//                    // Handle the error here
//                    Log.e("TfLite", "Initialization failed", exception)
//                }.continueWith { task ->
//                    if (task.isSuccessful) {
//                        try {
//                            val options = InterpreterApi.Options()
//                                .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
//                                .addDelegateFactory(GpuDelegateFactory())
//
//                            interpreterapi = InterpreterApi.create(
//                                getFileFromAssets(
//                                    this@MainActivity,
//                                    "yolov8n_float32.tflite"
//                                ), options
//                            )
//                        } catch (e: Exception) {
//                            Log.e("Interpreter", "Error during initialization", e)
//                        }
//                    } else {
//                        Log.d("Interpreter", "Initialization failed", task.exception)
//                    }
//                }
//            }
//
////          Standalone Fallback GPU Delegate
//            val compatList = CompatibilityList()
//            if (!play_services_flag) {
//                val options = Interpreter.Options().apply {
//                    if (compatList.isDelegateSupportedOnThisDevice) {
//                        // if the device has a supported GPU, add the GPU delegate
//                        val delegateOptions = compatList.bestOptionsForThisDevice
//                        Log.d("GPU", "=====================Standalone GPU delegate Supported=====================")
//                        this.addDelegate(GpuDelegate(delegateOptions))
//                        standalone_flag = true
//                    } else {
//                        Log.d("GPU", "===================Standalone GPU Delegate Not supported :: Switching to CPU (THREADS :8 ) ===================")
//                        this.setNumThreads(8)
//                        standalone_flag = false
//                        cpu_flag = true
//                    }
//                }
//                try {
//                    interpreter = Interpreter(
//                        getFileFromAssets(this@MainActivity, "yolov8n_float32.tflite"),
//                        options
//                    )
//                } catch (e: Exception) {
//                    Log.e("Interpreter", "Error during initialization", e)
//                }
//            }

            var gpu_options_list = mutableListOf<Boolean>(false,true,true);
            val assetManager: AssetManager = this.assets
            val yolov8 = Yolov8Classfier()
            yolov8.create(assetManager, "yolov8n_float32.tflite", "labels.txt",
                416,  gpu_options_list,4)

            var bitmap : Bitmap
            val texView = findViewById<TextureView>(R.id.textureView)
            var imgView = findViewById<ImageView>(R.id.imageView)
            val cameraManager: CameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            texView.surfaceTextureListener = object :TextureView.SurfaceTextureListener{
                override fun onSurfaceTextureAvailable(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) {
                        val cameraHandler = CameraHandler(0,cameraManager, texView)
                        cameraHandler.openCamera()
                }

                override fun onSurfaceTextureSizeChanged(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) {

                }

                override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                    return false
                }

                override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {

                }

            }
        }else{
            askPermission()
        }
    }

    private fun askPermission(){
        if(ContextCompat.checkSelfPermission(this,android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA),101)
        }
    }

    private fun checkPermission() : Boolean {
        return if(ContextCompat.checkSelfPermission(this,android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            Toast.makeText(this, "Permission Denied! Allow Permissions ", Toast.LENGTH_SHORT).show()
            false
        }else{
            true
        }
    }

    private fun getFileFromAssets(context: Context, fileName: String): File {
        val file = File(context.cacheDir, fileName)

        if (!file.exists()) {
            try {
                val asset = context.assets.open("/assets/ml/$fileName")
                val output = FileOutputStream(file)
                val buffer = ByteArray(1024)
                var read = asset.read(buffer)
                while (read != -1) {
                    output.write(buffer, 0, read)
                    read = asset.read(buffer)
                }
                asset.close()
                output.flush()
                output.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }

        return file
    }

}



