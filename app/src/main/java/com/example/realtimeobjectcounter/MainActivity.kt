package com.example.realtimeobjectcounter;

import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.view.MenuItem
import android.view.TextureView
import android.widget.Button
import android.widget.ImageButton
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import com.example.objcounter.CameraHandler
import com.example.realtimeobjectcounter.utils.Rectangle_ImgView
import android.content.Intent
import android.provider.MediaStore
import android.graphics.BitmapFactory
import android.net.Uri
import android.view.View

import com.example.tensorflow_yolov8.Yolov8Classfier
import com.google.android.material.navigation.NavigationView
private const val PICK_IMAGE = 1

class MainActivity : AppCompatActivity() {

    private lateinit var canvas: Canvas
    private var isFlashlightOn : Boolean = false
    private var isCaptureOn : Boolean = false
    private var isCaptureOff : Boolean = false
    private var surfaceReady = false
    val yolov8 = Yolov8Classfier()

    val workerThread = HandlerThread("workerThread").apply { start() }
    val workerHandler = Handler(workerThread.looper)

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var toggle: ActionBarDrawerToggle
    private val IOU_thres: Float = 0.5f
    private val CONFI_thres: Float = 0.5f
    override fun onDestroy() {
        super.onDestroy()
        yolov8.close()
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        if(checkPermission()){
            super.onCreate(savedInstanceState)
            setContentView(R.layout.activity_main)

            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            var bitmap : Bitmap
            val texView = findViewById<TextureView>(R.id.textureView)
            val imgview : Rectangle_ImgView = findViewById(R.id.imageView)
            val fps_button = findViewById<android.widget.Button>(R.id.button)
            val count_text = findViewById<android.widget.TextView>(R.id.textView)
            val cameraHandler = CameraHandler(0,cameraManager, texView)

            yolov8.setIOU_THRESHOLD(IOU_thres)
            yolov8.setCONFIDENCE_THRESHOLD(CONFI_thres)

            val camera_button : ImageButton = findViewById(R.id.capture);

            drawerLayout = findViewById(R.id.drawer_layout)
            val navigationView: NavigationView = findViewById(R.id.nav_view)
            toggle = ActionBarDrawerToggle(this, drawerLayout, R.string.open, R.string.close)
            drawerLayout.addDrawerListener(toggle)
            toggle.syncState()

            supportActionBar?.setDisplayHomeAsUpEnabled(true)

            val settingsButton: ImageButton = findViewById(R.id.settings)
            settingsButton.setOnClickListener {
                drawerLayout.openDrawer(GravityCompat.START)
            }

            val flashButton: ImageButton = findViewById(R.id.flash)
            flashButton.setOnClickListener {
                if(isFlashlightOn){
                    runOnUiThread {
                        cameraHandler.capture_no_flash();
                        isFlashlightOn = false
                        flashButton.setImageResource(R.drawable.baseline_flash_off_24)
                    }
                }else{
                    runOnUiThread {
                        cameraHandler.capture_flash();
                        isFlashlightOn = true
                        flashButton.setImageResource(R.drawable.baseline_flash_on_24)
                    }
                }
            }

            runOnUiThread{
                navigationView.setNavigationItemSelectedListener { menuItem ->
                    when (menuItem.itemId) {
                        R.id.threshold_slider -> {
                            val thresholdSlider: SeekBar =
                                menuItem.actionView!!.findViewById(R.id.slider)
                            thresholdSlider.setProgress((IOU_thres * 100).toInt())
                            thresholdSlider.setOnSeekBarChangeListener(object :
                                SeekBar.OnSeekBarChangeListener {
                                override fun onProgressChanged(
                                    seekBar: SeekBar?,
                                    progress: Int,
                                    fromUser: Boolean
                                ) {
                                    val threshold = progress / 100.0
                                    yolov8.setIOU_THRESHOLD(threshold.toFloat())
                                }

                                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
                            })

                        }

                        R.id.confidence_slider -> {
                            val confidenceSlider: SeekBar =
                                menuItem.actionView!!.findViewById(R.id.slider)
                            confidenceSlider.setProgress((CONFI_thres * 100).toInt())
                            confidenceSlider.setOnSeekBarChangeListener(object :
                                SeekBar.OnSeekBarChangeListener {
                                override fun onProgressChanged(
                                    seekBar: SeekBar?,
                                    progress: Int,
                                    fromUser: Boolean
                                ) {
                                    val confidence = progress / 100.0
                                    yolov8.setCONFIDENCE_THRESHOLD(confidence.toFloat())
                                }

                                override fun onStartTrackingTouch(seekBar: SeekBar?) {}

                                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
                            })

                        }
                    }
                    true
                }
            }

            val assetManager: AssetManager = this.assets
            val numThreads: Int = 8
            yolov8.setNUM_THREADS(numThreads)
            yolov8.useGPU(true)
            yolov8.useNNAPI(true)
            yolov8.setQuantized(true)
            yolov8.create(assetManager, "yolov8n_int8.tflite", "labels.txt",
                384, numThreads)

            runOnUiThread {
                val navigationView: NavigationView = findViewById(R.id.nav_view)
                val title_menu = navigationView.menu
                val title_menuItem = title_menu.findItem(R.id.INFO_HASH) // replace with your menu item id

                val spanString = SpannableString(title_menuItem.title.toString())
                spanString.setSpan(ForegroundColorSpan(Color.RED), 0, spanString.length, 0)// Change color to whatever you want
                title_menuItem.title = spanString

                val menu = navigationView.menu
                val menuItem = menu.findItem(R.id.gpu)
                menuItem.title = "GPU: ${yolov8.isGPU}"
                val menuItem2 = menu.findItem(R.id.nnapi)
                menuItem2.title = "NNAPI: ${yolov8.isNNAPI}"
                val menuItem3 = menu.findItem(R.id.quantized)
                menuItem3.title = "Quantized: ${yolov8.isQuantized}"
                val menuItem4 = menu.findItem(R.id.threads)
                menuItem4.title = "Threads: ${yolov8.NUM_THREADS}"
            }

            cameraHandler.openCamera()
            camera_button.setOnClickListener{
                isCaptureOn = true
                    runOnUiThread{

                        if(isCaptureOn && !isCaptureOff){
                            cameraHandler.closeCamera()
                            Toast.makeText(this,"Captured Temporarily ", Toast.LENGTH_SHORT).show()
                            isCaptureOff =true
                        }else if(isCaptureOff){
                            cameraHandler.openCamera()
                            isCaptureOff =false
                        }
                    }
            }

            texView.surfaceTextureListener = object :TextureView.SurfaceTextureListener{
                override fun onSurfaceTextureAvailable(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) {
                    surfaceReady = true
                }
                override fun onSurfaceTextureSizeChanged(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) {
                    surfaceReady = true
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
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (toggle.onOptionsItemSelected(item)) {
            return true
        }
        return super.onOptionsItemSelected(item)
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



