package com.example.realtimeobjectcounter.utils

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import androidx.appcompat.widget.AppCompatImageView

class Rectangle_ImgView(context: Context, attrs: AttributeSet?) : AppCompatImageView(context, attrs) {

    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val rectangles = mutableListOf<com.example.tensorflow_yolov8.Utils.BoundingBox>()

    fun setRectangles(rectangles: List<com.example.tensorflow_yolov8.Utils.BoundingBox>) {
        this.rectangles.clear()
        this.rectangles.addAll(rectangles)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (rect in rectangles) {
            canvas.drawRect(rect.x1,rect.y1,rect.x2,rect.y2, paint)
        }
    }
}
