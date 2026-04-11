#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <vector>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

const float CONF_THRESH = 0.5f;
const float NMS_THRESH  = 0.35f;
const vector<string> CLASS_NAMES  = {"car", "stop_sign"};
const vector<Scalar> CLASS_COLORS = {{0,0,255}, {0,255,0}};
const vector<float>  REAL_WIDTHS  = {0.35f, 0.75f};
const float FOV_H_DEG       = 70.0f;
const float FOCAL_LENGTH_PX = (640.f / 2.f) / tan(FOV_H_DEG * M_PI / 180.f / 2.f);

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("sign_node");

    auto pub_car  = node->create_publisher<std_msgs::msg::Float32>("/car_distance",  10);
    auto pub_stop = node->create_publisher<std_msgs::msg::Float32>("/stop_distance", 10);

    ncnn::Net net;
    net.opt.use_vulkan_compute    = false;
    // net.opt.num_threads        = 4;      // Uncomment on Jetson
    // net.opt.use_fp16_packed    = true;   
    // net.opt.use_fp16_storage   = true;
    // net.opt.use_fp16_arithmetic= true;

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

    Mat frame;

    while (rclcpp::ok()) {
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

        int num_classes = CLASS_NAMES.size();
        vector<vector<Rect>>  all_boxes(num_classes);
        vector<vector<float>> all_scores(num_classes);

        for (int i = 0; i < out.w; i++) {
            float cx = out.channel(0).row(0)[i] * sx;
            float cy = out.channel(0).row(1)[i] * sy;
            float w  = out.channel(0).row(2)[i] * sx;
            float h  = out.channel(0).row(3)[i] * sy;

            int   best_cls  = -1;
            float best_conf = CONF_THRESH;
            for (int c = 0; c < num_classes; c++) {
                float conf = out.channel(0).row(4 + c)[i];
                if (conf > best_conf) { best_conf = conf; best_cls = c; }
            }
            if (best_cls < 0) continue;

            all_boxes[best_cls].push_back({(int)(cx-w/2), (int)(cy-h/2), (int)w, (int)h});
            all_scores[best_cls].push_back(best_conf);
        }

        
        for (int c = 0; c < num_classes; c++) {
            vector<int> idx;
            dnn::NMSBoxes(all_boxes[c], all_scores[c], CONF_THRESH, NMS_THRESH, idx);

            float best_dist = -1.f;

            for (int i : idx) {
                Rect b = all_boxes[c][i] & Rect(0, 0, frame.cols, frame.rows);
                if (b.width <= 0 || b.height <= 0) continue;

                float dist_m = (REAL_WIDTHS[c] * FOCAL_LENGTH_PX) / (float)b.width;

                // Keep closest detection per class
                if (best_dist < 0 || dist_m < best_dist)
                    best_dist = dist_m;

                rectangle(frame, b, CLASS_COLORS[c], 2);
                putText(frame, format("%s %.2fm", CLASS_NAMES[c].c_str(), dist_m),
                        b.tl() + Point(0, -6),
                        FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[c], 1);
            }

            // Publish closest detection
            if (best_dist > 0) {
                std_msgs::msg::Float32 msg;
                msg.data = best_dist;
                if (c == 0) pub_car->publish(msg);
                else        pub_stop->publish(msg);
            }
        }

        imshow("Detection", frame);
        rclcpp::spin_some(node);
        if (waitKey(1) == 'q') break;
    }

    rclcpp::shutdown();
    return 0;
}