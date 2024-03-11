package com.example.tensorflow_yolov8;

import com.example.tensorflow_yolov8.Utils.BoundingBox;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Detection_factory {
    public static int tensor_shape;
    public static int[] img_shape;
    private static float threshold = 0.5f;
    private static float[] calculateIOU(float[] box, List<float[]> boxes) {
        float[] iou = new float[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            float[] box_i = boxes.get(i);
            float x1 = Math.max(box[0], box_i[0]);
            float y1 = Math.max(box[1], box_i[1]);
            float x2 = Math.min(box[0] + box[2], box_i[0] + box_i[2]);
            float y2 = Math.min(box[1] + box[3], box_i[1] + box_i[3]);
            float intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            float areaBox = box[2] * box[3];
            float areaBoxes = box_i[2] * box_i[3];
            float union = areaBox + areaBoxes - intersection;

            iou[i] = intersection / union;
        }

        return iou;
    }

    private static ArrayList<float[]> non_maximum_suppression(ArrayList<float[]> boxes) {
        ArrayList<float[]> selectedBoxes = new ArrayList<>();
        Collections.sort(boxes,new Comparator<float[]>(){
            public int compare(float[] a, float[] b){
                return Float.compare(a[5],b[5]);
            }
        });
        while(!boxes.isEmpty()){
            int selected_index = boxes.indexOf(boxes.get(0));
            int selected_class = (int) boxes.get(selected_index)[4];
            selectedBoxes.add(boxes.get(selected_index));
            float[] selected_box = boxes.get(selected_index);
            float[] ious = calculateIOU(selected_box, boxes.subList(1,boxes.size()));
            ArrayList<float[]> filtered_boxes = new ArrayList<>();
            for (int i = 1; i < boxes.size(); i++) {
                if (ious[i-1] <= threshold || (int)boxes.get(i)[4] != selected_class) {
                    filtered_boxes.add(boxes.get(i));
                }
            }
            boxes = filtered_boxes;
        }
        return selectedBoxes;
    }

    public static float[] scale_coords(float[] coords ){
        float widthScaleFactor = (float) img_shape[0]/tensor_shape;
        float heightScaleFactor = (float) img_shape[1]/tensor_shape;

        coords[0] = coords[0] * widthScaleFactor * tensor_shape;
        coords[1] = coords[1] * heightScaleFactor * tensor_shape;
        coords[2] = coords[2] * widthScaleFactor * tensor_shape;
        coords[3] = coords[3] * heightScaleFactor * tensor_shape;

        return coords;
    }

    public static float[][] getTranspose(float[][][] input){
        float [][] output = new float[input[0][0].length][input[0].length];
        float [][] tmp = new float[input[0].length][input[0][0].length];
        //shrink the array from (1,num_classes,num_elements) to (num_classes,num_elements)
        for (int i = 0; i < input[0].length; i++) {
            for (int j = 0; j < input[0][0].length; j++) {
                tmp[i][j] = input[0][i][j];
            }
        }
        //transpose the array form (num_classes,num_elements) to (num_elements,num_classes)
        for (int x = 0; x < tmp.length; x++) {
            for (int y = 0; y < tmp[0].length; y++) {
                output[y][x] = tmp[x][y];
            }
        }
        return output;
    }

    public static List<BoundingBox> getBoundingBox(ArrayList<float[]> boxes, List<String> labels,float IOU_THRESHOLD){
           boxes =  non_maximum_suppression(boxes);
           int x1,y1,x2,y2 = 0;
           float confidence = 0;
           String label = "";
           List<BoundingBox> boundingBoxes = new ArrayList<>();

        for (float[] box : boxes) {
            scale_coords(box);
            //x1 = (x - w / 2);
            //y1 = (y - h / 2);
            //x2 = (x + w / 2);
            //y2 = (y + h / 2);
            x1 = (int)(box[0]- box[2] / 2);
            y1 = (int)(box[1] - box[3] / 2);
            x2 = (int)(box[0] + box[2] / 2);
            y2 = (int)(box[1] + box[3] / 2);
            label = labels.get((int)box[4]);
            confidence = box[5];
            boundingBoxes.add(new BoundingBox(x1,y1,x2,y2,label,confidence));
        }

        return boundingBoxes;
    }
}
