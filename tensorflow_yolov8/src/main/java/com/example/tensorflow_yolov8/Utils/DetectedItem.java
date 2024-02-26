package com.example.tensorflow_yolov8.Utils;

import android.graphics.RectF;

public class DetectedItem {
    public RectF location;
    public String label;
    public float confidence;

    public DetectedItem(RectF location, String label, float confidence) {
        this.location = location;
        this.label = label;
        this.confidence = confidence;
    }

    // getters and setters
}