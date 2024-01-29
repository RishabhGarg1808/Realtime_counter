package com.example.realtimeobjectcounter;

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Log
import android.view.TextureView
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.objcounter.CameraHandler
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if(checkPermission()){

            val compatList = CompatibilityList()
            val options = Interpreter.Options().apply{
                if(compatList.isDelegateSupportedOnThisDevice){
                    // if the device has a supported GPU, add the GPU delegate
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    Log.d("GPU", "=====================Supported=====================")
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    Log.d("GPU", "===================Not supported==================")
                    this.setNumThreads(4)
                }
            }
            val interpreter = Interpreter(getFileFromAssets(this@MainActivity,"yolov8n_float32.tflite") ,options)

            var bitmap : Bitmap
            val imageProcessor = ImageProcessor.Builder().add(ResizeOp(1280, 720, ResizeOp.ResizeMethod.BILINEAR)).build()
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
                val asset = context.assets.open("ml/$fileName")
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
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(bitmap.byteCount)
        bitmap.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()
        return byteBuffer
    }

}

