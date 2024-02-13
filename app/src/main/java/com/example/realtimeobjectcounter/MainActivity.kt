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
            //yolov8.useNNAPI(true)
            yolov8.setQuantized(true)
            yolov8.create(assetManager, "yolov8n_integer_quant.tflite", "labels.txt",
                448, 4)

            var bitmap : Bitmap
            var image : TensorImage;
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
                    bitmap = texView.bitmap!!
                    yolov8.detect(bitmap)
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



