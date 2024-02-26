package com.example.realtimeobjectcounter;

import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.TextureView
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.objcounter.CameraHandler


import com.example.tensorflow_yolov8.Yolov8Classfier
import org.tensorflow.lite.support.image.TensorImage

class MainActivity : AppCompatActivity() {

    val yolov8 = Yolov8Classfier()
    override fun onDestroy() {
        super.onDestroy()
        yolov8.close()
    }
    override fun onCreate(savedInstanceState: Bundle?) {

        if(checkPermission()){

            super.onCreate(savedInstanceState)
            setContentView(R.layout.activity_main)

            val assetManager: AssetManager = this.assets

            yolov8.setNUM_THREADS(8)
            yolov8.useGPU(true)
            yolov8.useNNAPI(true)
            yolov8.setQuantized(false)
            yolov8.create(assetManager, "ssd.tflite", "labels.txt",
                300, 8)

            var bitmap : Bitmap
            var image : TensorImage;
            val texView = findViewById<TextureView>(R.id.textureView)
            var imgView = findViewById<ImageView>(R.id.imageView)
            val button = findViewById<android.widget.Button>(R.id.button)

            val cameraManager: CameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraHandler = CameraHandler(1,cameraManager, texView)
            cameraHandler.openCamera()

            val workerThread = HandlerThread("workerThread").apply { start() }
            val workerHandler = Handler(workerThread.looper)


            texView.surfaceTextureListener = object :TextureView.SurfaceTextureListener{
                override fun onSurfaceTextureAvailable(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) {

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
                    workerHandler.post {
                    bitmap = texView.bitmap!!
                    val detectedItems = yolov8.detect(bitmap)
                        for (item in detectedItems) {
                            Log.d("DetectedItem", "Label: ${item.label}, Score: ${item.confidence}, BoundingBox: ${item.location}")
                        }
                }
                    runOnUiThread {
                    button.setText("FPS : ${yolov8.getFPS()}")
                    }
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
}



