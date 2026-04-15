// Sign + Car Detection — ROS2 Node
// Subscribes to /image_raw, runs NCNN inference, publishes /robot/perception
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <vector>
#include <iostream>
#include <functional>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace cv;
using namespace std;

// ─────────────────────────────────────────────
//  DETECTION CONFIG
// ─────────────────────────────────────────────
namespace cfg {
    constexpr float CONF_THRESH = 0.5f;
    constexpr float NMS_THRESH  = 0.35f;
    constexpr float FOV_H_DEG    = 70.0f;
    constexpr float FOCAL_LEN_PX = (640.f / 2.f) / tan(FOV_H_DEG * M_PI / 180.f / 2.f);
}

// Class names, colors, and real-world widths for distance estimation
const vector<string> CLASS_NAMES  = { "car",       "stop_sign" };
const vector<Scalar> CLASS_COLORS = { {0, 0, 255}, {0, 255, 0} };
const vector<float>  REAL_WIDTHS  = { 0.75f,        0.35f      }; // metres

// ─────────────────────────────────────────────
//  ROS2 NODE
// ─────────────────────────────────────────────
class SignNode : public rclcpp::Node {
public:
    SignNode() : Node("sign_detection") {

        this->declare_parameter("model_path",
            "/home/regality/ros2_ws/src/lane_detection/model.ncnn");
        string model_path = this->get_parameter("model_path").as_string();

        net_.opt.use_vulkan_compute = false;
        string param = model_path + ".param";
        string bin   = model_path + ".bin";

        if (net_.load_param(param.c_str()) || net_.load_model(bin.c_str())) {
            RCLCPP_FATAL(this->get_logger(),
                "Failed to load NCNN model from: %s", model_path.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "NCNN model loaded from: %s", model_path.c_str());

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10,
            std::bind(&SignNode::image_callback, this, std::placeholders::_1)
        );

        // Publishes [car_dist, stop_dist]
        // -1.0 means not detected
        pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/robot/perception", 10
        );

        namedWindow("Perception");
        RCLCPP_INFO(this->get_logger(), "SignNode ready, waiting for /image_raw ...");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        if (frame.empty()) return;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            frame.data, ncnn::Mat::PIXEL_BGR2RGB,
            frame.cols, frame.rows, 640, 640
        );

        const float mean[3] = {0.f, 0.f, 0.f};
        const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(mean, norm);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;              // ← add this line
        ex.extract("out0", out);

        float sx = (float)frame.cols / 640.f;
        float sy = (float)frame.rows / 640.f;
        int nc = (int)CLASS_NAMES.size();

        vector<vector<Rect>>  boxes(nc);
        vector<vector<float>> scores(nc);

        for (int i = 0; i < out.w; i++) {
            float cx = out.channel(0).row(0)[i] * sx;
            float cy = out.channel(0).row(1)[i] * sy;
            float w  = out.channel(0).row(2)[i] * sx;
            float h  = out.channel(0).row(3)[i] * sy;

            int   best  = -1;
            float bconf = cfg::CONF_THRESH;
            for (int c = 0; c < nc; c++) {
                float conf = out.channel(0).row(4 + c)[i];
                if (conf > bconf) { bconf = conf; best = c; }
            }
            if (best < 0) continue;

            boxes[best].push_back({
                (int)(cx - w/2), (int)(cy - h/2), (int)w, (int)h
            });
            scores[best].push_back(bconf);
        }

        // ── NMS + distance estimation ─────────────
        float dist_car  = -1.f;
        float dist_stop = -1.f;

        for (int c = 0; c < nc; c++) {
            if (boxes[c].empty()) continue;

            vector<int> idx;
            dnn::NMSBoxes(boxes[c], scores[c], cfg::CONF_THRESH, cfg::NMS_THRESH, idx);

            for (int i : idx) {
                Rect b = boxes[c][i] & Rect(0, 0, frame.cols, frame.rows);
                if (b.width <= 0 || b.height <= 0) continue;

                float dist = (REAL_WIDTHS[c] * cfg::FOCAL_LEN_PX) / (float)b.width;

                if (c == 0) { if (dist_car  < 0 || dist < dist_car)  dist_car  = dist; }
                else        { if (dist_stop < 0 || dist < dist_stop) dist_stop = dist; }

                rectangle(frame, b, CLASS_COLORS[c], 2);
                putText(frame,
                    format("%s %.2fm", CLASS_NAMES[c].c_str(), dist),
                    b.tl() + Point(0, -6),
                    FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[c], 1, LINE_AA);
            }
        }

        // ── Publish distances ────────────────────
        // [0] = car_dist   (-1 if not detected)
        // [1] = stop_dist  (-1 if not detected)
        std_msgs::msg::Float32MultiArray perception_msg;
        perception_msg.data = { dist_car, dist_stop };
        pub_->publish(perception_msg);

        imshow("Perception", frame);
        waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
    ncnn::Net net_;
};

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SignNode>());
    rclcpp::shutdown();
    return 0;
}