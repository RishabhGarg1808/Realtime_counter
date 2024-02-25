package com.example.objcounter

import android.annotation.SuppressLint
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.view.Surface
import android.view.TextureView
import android.widget.Toast


class CameraHandler(
    private val cameraId : Int, private var cameraManager : CameraManager,
    var textureView: TextureView) {
    private lateinit var cameraDevice : CameraDevice
    @SuppressLint("MissingPermission")
    fun openCamera(){
        cameraManager.openCamera(cameraManager.cameraIdList[cameraId], object: CameraDevice.StateCallback()
        {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                var captureRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(Surface(textureView.surfaceTexture))
                cameraDevice.createCaptureSession(listOf(Surface(textureView.surfaceTexture)), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(captureRequest.build(), null, null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(textureView.context, "Configuration Failed", Toast.LENGTH_SHORT).show()
                    }

                }, null)
            }

            override fun onDisconnected(camera: CameraDevice) {
                Toast.makeText(textureView.context, "Camera Disconnected", Toast.LENGTH_SHORT).show()
            }

            override fun onError(camera: CameraDevice, error: Int) {
                Toast.makeText(textureView.context, "Camera Error", Toast.LENGTH_SHORT).show()
            }

        }, null)
    }
}

