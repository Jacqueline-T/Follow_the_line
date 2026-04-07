//Refactor 
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

const float CONF_THRESH = 0.25f;
const float NMS_THRESH  = 0.45f;

int slider_top_w = 200, slider_bot_w = 450, slider_height = 180, slider_offset = 300;

int main() {
    ncnn::Net net;    
    net.opt.use_vulkan_compute = false; // Keep false for WSL/Nano 2GB RAM safety
    if (net.load_param("/home/dalw/ros2_ws/src/lane_detection/model.ncnn.param") ||
        net.load_model("/home/dalw/ros2_ws/src/lane_detection/model.ncnn.bin")) {
        cerr << "Model loading failed!" << endl;
        return -1;
    }

    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);

    namedWindow("Lane Calibration");
    createTrackbar("Top Width", "Lane Calibration", &slider_top_w, 600);
    createTrackbar("Bot Width", "Lane Calibration", &slider_bot_w, 640);
    createTrackbar("Height", "Lane Calibration", &slider_height, 480);
    createTrackbar("Y-Offset", "Lane Calibration", &slider_offset, 480);

    Mat frame, warped, gray, binary;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            frame.data, ncnn::Mat::PIXEL_BGR2RGB,
            frame.cols, frame.rows, 640, 640
        );

        const float mean[3] = {0.f, 0.f, 0.f};
        const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(mean, norm);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        float sx = (float)frame.cols / 640.f;
        float sy = (float)frame.rows / 640.f;

        vector<Rect> boxes;
        vector<float> scores;

        for (int i = 0; i < out.w; i++) {
            float conf = out.channel(0).row(4)[i];
            if (conf < CONF_THRESH) continue;
            float cx = out.channel(0).row(0)[i] * sx;
            float cy = out.channel(0).row(1)[i] * sy;
            float w  = out.channel(0).row(2)[i] * sx;
            float h  = out.channel(0).row(3)[i] * sy;
            boxes.push_back({(int)(cx-w/2), (int)(cy-h/2), (int)w, (int)h});
            scores.push_back(conf);
        }

        vector<int> idx;
        dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, idx);

        for (int i : idx) {
            Rect b = boxes[i] & Rect(0, 0, frame.cols, frame.rows);
            rectangle(frame, b, {0, 255, 0}, 2); // Green for signs
            putText(frame, "SIGN", b.tl(), FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);
        }

        Point2f src[] = {
            Point2f(320 - slider_top_w/2, slider_offset),
            Point2f(320 + slider_top_w/2, slider_offset),
            Point2f(320 + slider_bot_w/2, slider_offset + slider_height),
            Point2f(320 - slider_bot_w/2, slider_offset + slider_height)
        };
        Point2f dst[] = { {150,0}, {490,0}, {490,480}, {150,480} };

        Mat M = getPerspectiveTransform(src, dst);
        warpPerspective(frame, warped, M, Size(640, 480));

        cvtColor(warped, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 150, 255, THRESH_BINARY); 

        line(frame, src[0], src[1], {255,0,0}, 2);
        line(frame, src[1], src[2], {255,0,0}, 2);
        line(frame, src[2], src[3], {255,0,0}, 2);
        line(frame, src[3], src[0], {255,0,0}, 2);

        imshow("Master Vision (Signs + ROI)", frame);
        imshow("Lane - Birdseye", warped);
        imshow("Lane - Binary Map", binary);

        if (waitKey(1) == 'q') break;
    }
    return 0;
}