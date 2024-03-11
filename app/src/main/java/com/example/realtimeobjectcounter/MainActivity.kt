package com.example.realtimeobjectcounter;

import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.TextureView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.objcounter.CameraHandler
import com.example.realtimeobjectcounter.utils.Rectangle_ImgView

import com.example.tensorflow_yolov8.Yolov8Classfier

class MainActivity : AppCompatActivity() {
    private lateinit var canvas: Canvas

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

            yolov8.setNUM_THREADS(32)
            yolov8.useGPU(true)
            yolov8.useNNAPI(true)
            yolov8.setQuantized(true)
            yolov8.create(assetManager, "yolov8n_int8.tflite", "labels.txt",
                384, 32)

            var bitmap : Bitmap
            val texView = findViewById<TextureView>(R.id.textureView)
            val imgview : Rectangle_ImgView = findViewById(R.id.imageView)
            val fps_button = findViewById<android.widget.Button>(R.id.button)
            val count_text = findViewById<android.widget.TextView>(R.id.textView)

            val cameraManager: CameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraHandler = CameraHandler(0,cameraManager, texView)
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
                       val Bbox : MutableList<com.example.tensorflow_yolov8.Utils.BoundingBox>? = yolov8.detect(bitmap)
                        runOnUiThread {
                            fps_button.setText("${yolov8.getFPS()}")
                            count_text.setText("Count : ${Bbox?.size}")
                            imgview.setRectangles(Bbox as List<com.example.tensorflow_yolov8.Utils.BoundingBox>)
                        }
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



