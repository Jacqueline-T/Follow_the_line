#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

const float CONF_THRESH = 0.25f;
const float NMS_THRESH  = 0.45f;

int main() {
    ncnn::Net net;    
    net.opt.use_vulkan_compute = false;
    net.load_param("/home/dalw/ros2_ws/src/lane_detection/model.ncnn.param");
    net.load_model("/home/dalw/ros2_ws/src/lane_detection/model.ncnn.bin");

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::Mat frame;
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

        std::vector<cv::Rect> boxes;
        std::vector<float>    scores;

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

        std::vector<int> idx;
        cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, idx);

        for (int i : idx) {
            cv::Rect b = boxes[i] & cv::Rect(0, 0, frame.cols, frame.rows);
            cv::rectangle(frame, b, {0, 0, 255}, 2);
        }

        cv::imshow("signs", frame);
        if (cv::waitKey(1) == 'q') break;
    }
}