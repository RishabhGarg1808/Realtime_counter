package com.example.tensorflow_yolov8.Utils;

public class BoundingBox {
    public  float x1, y1,x2,y2;
    public String label;
    public float confidence;

    public BoundingBox(float x1, float y1, float x2, float y2, String label, float confidence) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.label = label;
        this.confidence = confidence;
    }
    public BoundingBox() {

    }

}