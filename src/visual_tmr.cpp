#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

using namespace cv;
using namespace std;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
namespace cfg {
    constexpr double ALPHA       = 0.3;
    constexpr int    N_VENTANAS  = 9;
    constexpr int    MARGEN      = 50;
    constexpr int    MIN_PIX     = 50;
    constexpr int    MAX_PERDIDO = 12;
    constexpr int    MIN_SEP     = 100;
    constexpr int    MIN_SEP_FIT = 150;

    constexpr float CONF_THRESH  = 0.5f;
    constexpr float NMS_THRESH   = 0.35f;
    constexpr float FOV_H_DEG    = 70.0f;
    const     float FOCAL_LEN_PX = (640.f / 2.f) / tanf(FOV_H_DEG * (float)M_PI / 180.f / 2.f);
}

const vector<string> CLASS_NAMES  = {"car", "stop_sign"};
const vector<Scalar> CLASS_COLORS = {{0,0,255}, {0,255,0}};
const vector<float>  REAL_WIDTHS  = {0.35f, 0.75f};

// ---------------------------------------------------------------------------
// Small structs
// ---------------------------------------------------------------------------
struct Cooldown {
    int restante = 0;
    static constexpr int MAX = 45;
    bool listo()   { return restante == 0; }
    void activar() { restante = MAX; }
    void tick()    { if (restante > 0) restante--; }
};

struct AjusteCarril {
    double A = 0, B = 0, C = 0;
    bool   valido           = false;
    int    cuadros_perdidos = 0;
};

// ---------------------------------------------------------------------------
// FrameSlot — single-slot buffer between lane thread and detection thread
// ---------------------------------------------------------------------------
struct FrameSlot {
    Mat   frame;
    bool  fresh = false;
    mutex mtx;
    condition_variable cv;

    // Producer: lane thread deposits a new frame
    void put(const Mat& src) {
        {
            lock_guard<mutex> lk(mtx);
            src.copyTo(frame);
            fresh = true;
        }
        cv.notify_one();
    }

    // Consumer: detection thread waits for a fresh frame (returns false on timeout)
    bool get(Mat& dst, int timeout_ms = 33) {
        unique_lock<mutex> lk(mtx);
        if (!cv.wait_for(lk, chrono::milliseconds(timeout_ms), [&]{ return fresh; }))
            return false;
        frame.copyTo(dst);
        fresh = false;
        return true;
    }
};

// ---------------------------------------------------------------------------
// ObjectDetector — plain class, no Node, owns the NCNN model
// ---------------------------------------------------------------------------
class ObjectDetector {
public:
    explicit ObjectDetector(rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub)
        : pub_(pub)
    {}

    bool loadModel(const string& param, const string& bin) {
        return net_.load_param(param.c_str()) == 0 &&
               net_.load_model(bin.c_str())   == 0;
    }

    // Runs inference, draws boxes on frame (in-place), publishes distances.
    void processFrame(Mat& frame) {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            frame.data, ncnn::Mat::PIXEL_BGR2RGB,
            frame.cols, frame.rows, 640, 640);

        const float mean[3] = {0, 0, 0};
        const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(mean, norm);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        float sx = (float)frame.cols / 640.f;
        float sy = (float)frame.rows / 640.f;
        int   nc = (int)CLASS_NAMES.size();

        vector<vector<Rect>>  boxes(nc);
        vector<vector<float>> scores(nc);

        for (int i = 0; i < out.w; i++) {
            float cx = out.channel(0).row(0)[i] * sx;
            float cy = out.channel(0).row(1)[i] * sy;
            float w  = out.channel(0).row(2)[i] * sx;
            float h  = out.channel(0).row(3)[i] * sy;

            int best = -1; float bconf = cfg::CONF_THRESH;
            for (int c = 0; c < nc; c++) {
                float conf = out.channel(0).row(4 + c)[i];
                if (conf > bconf) { bconf = conf; best = c; }
            }
            if (best < 0) continue;
            boxes [best].push_back({(int)(cx-w/2), (int)(cy-h/2), (int)w, (int)h});
            scores[best].push_back(bconf);
        }

        float dist_car = -1.f, dist_stop = -1.f;
        for (int c = 0; c < nc; c++) {
            vector<int> idx;
            dnn::NMSBoxes(boxes[c], scores[c], cfg::CONF_THRESH, cfg::NMS_THRESH, idx);
            for (int i : idx) {
                Rect b = boxes[c][i] & Rect(0, 0, frame.cols, frame.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                float d = (REAL_WIDTHS[c] * cfg::FOCAL_LEN_PX) / (float)b.width;
                if (c == 0) { if (dist_car  < 0 || d < dist_car)  dist_car  = d; }
                else        { if (dist_stop < 0 || d < dist_stop) dist_stop = d; }
                rectangle(frame, b, CLASS_COLORS[c], 2);
                putText(frame, format("%s %.2fm", CLASS_NAMES[c].c_str(), d),
                        b.tl() + Point(0, -6), FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[c], 1);
            }
        }

        std_msgs::msg::Float32MultiArray msg;
        msg.data = {0.f, 0.f,
                    dist_car  > 0 ? dist_car  : -1.f,
                    dist_stop > 0 ? dist_stop : -1.f};
        pub_->publish(msg);
    }

private:
    ncnn::Net net_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
};

// ---------------------------------------------------------------------------
// PerceptionNode — one node, two publishers, one camera, two threads
// ---------------------------------------------------------------------------
class PerceptionNode : public rclcpp::Node {
public:
    PerceptionNode()
        : Node("perception_node")
        , detector_(create_publisher<std_msgs::msg::Float32MultiArray>("robot/perception", 10))
    {
        lane_pub_ = create_publisher<geometry_msgs::msg::Vector3>("lane/info", 10);

        // Parameters
        declare_parameter("ancho_sup",  200);
        declare_parameter("ancho_inf",  450);
        declare_parameter("alto_roi",   180);
        declare_parameter("despl_y",    300);
        declare_parameter("blanco_min", 200);
        declare_parameter("canny_bajo", 40);
        declare_parameter("canny_alto", 120);

        int ancho_sup = get_parameter("ancho_sup").as_int();
        int ancho_inf = get_parameter("ancho_inf").as_int();
        int alto_roi  = get_parameter("alto_roi").as_int();
        int despl_y   = get_parameter("despl_y").as_int();
        blanco_min_   = get_parameter("blanco_min").as_int();
        canny_bajo_   = get_parameter("canny_bajo").as_int();
        canny_alto_   = get_parameter("canny_alto").as_int();

        // Camera
        cap_.open("/dev/video0", CAP_V4L2);
        cap_.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        cap_.set(CAP_PROP_FRAME_WIDTH,  640);
        cap_.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap_.set(CAP_PROP_FPS,          30);
        if (!cap_.isOpened())
            RCLCPP_FATAL(get_logger(), "No se pudo abrir la cámara");

        // Perspective transform
        const float c = 320.f;
        Point2f src[4] = {
            {c - ancho_sup/2.f, (float)despl_y},
            {c + ancho_sup/2.f, (float)despl_y},
            {c + ancho_inf/2.f, (float)(despl_y + alto_roi)},
            {c - ancho_inf/2.f, (float)(despl_y + alto_roi)}
        };
        Point2f dst[4] = {{150,0},{490,0},{490,480},{150,480}};
        M_    = getPerspectiveTransform(src, dst);
        Minv_ = getPerspectiveTransform(dst, src);

        s_Y_ = Mat(8000, 3, CV_64F);
        s_X_ = Mat(8000, 1, CV_64F);

        if (!detector_.loadModel(
                "/home/dalw/ros2_ws/src/lane_detection/model.ncnn.param",
                "/home/dalw/ros2_ws/src/lane_detection/model.ncnn.bin"))
            RCLCPP_FATAL(get_logger(), "Fallo al cargar el modelo NCNN");
    }

    // Lane thread — called from main()
    void laneLoop(atomic<bool>& running) {
        Mat frame, perspectiva, binaria, gris_chk, carril_bev, carril_orig, hist;
        vector<Point> curva_izq, curva_der, poligono;
        curva_izq.reserve(100); curva_der.reserve(100); poligono.reserve(200);

        while (running) {
            cap_ >> frame;
            if (frame.empty()) { running = false; break; }

            warpPerspective(frame, perspectiva, M_, Size(640, 480));
            crearMascaraBinaria(perspectiva, binaria);

            Rect cw_rect = detectarCrosswalk(binaria);

            cvtColor(frame, gris_chk, COLOR_BGR2GRAY);
            if (estaSobreexpuesto(gris_chk)) {
                aj_izq_ = degradarAjuste(aj_izq_);
                aj_der_ = degradarAjuste(aj_der_);
                slot_.put(frame);
                continue;
            }

            Mat mitad = binaria(Range(binaria.rows/2, binaria.rows), Range::all());
            reduce(mitad, hist, 0, REDUCE_SUM, CV_32S);

            double minV, maxV; Point minL, maxL;
            minMaxLoc(hist.colRange(0,   320), &minV, &maxV, &minL, &maxL);
            int x_izq = maxL.x;
            minMaxLoc(hist.colRange(320, 640), &minV, &maxV, &minL, &maxL);
            int x_der = maxL.x + 320;

            const int y_base = binaria.rows - 1;
            if (aj_izq_.valido) x_izq = evalAjuste(aj_izq_, y_base);
            if (aj_der_.valido) x_der = evalAjuste(aj_der_, y_base);

            ventanasDeslizantes(binaria, x_izq, x_der);

            double lA,lB,lC,rA,rB,rC;
            bool di = ajustarPoli(pts_izq_, lA,lB,lC);
            bool dd = ajustarPoli(pts_der_, rA,rB,rC);

            if (di && dd) {
                int lx = (int)(lA*y_base*y_base + lB*y_base + lC);
                int rx = (int)(rA*y_base*y_base + rB*y_base + rC);
                if (rx - lx < cfg::MIN_SEP_FIT) {
                    aj_izq_ = degradarAjuste(aj_izq_);
                    aj_der_ = degradarAjuste(aj_der_);
                } else {
                    aj_izq_ = suavizar(aj_izq_, lA,lB,lC);
                    aj_der_ = suavizar(aj_der_, rA,rB,rC);
                }
            } else {
                aj_izq_ = di && plausible(aj_izq_, lA,lB,lC, y_base)
                          ? suavizar(aj_izq_, lA,lB,lC) : degradarAjuste(aj_izq_);
                aj_der_ = dd && plausible(aj_der_, rA,rB,rC, y_base)
                          ? suavizar(aj_der_, rA,rB,rC) : degradarAjuste(aj_der_);
            }

            if (crosswalk_detectado_)
                dibujarCrosswalk(frame, cw_rect);

            if (aj_izq_.valido && aj_der_.valido) {
                curva_izq.clear(); curva_der.clear();
                for (int y = 0; y < perspectiva.rows; y += 5) {
                    curva_izq.push_back({clamp(evalAjuste(aj_izq_,y), 0, perspectiva.cols-1), y});
                    curva_der.push_back({clamp(evalAjuste(aj_der_,y), 0, perspectiva.cols-1), y});
                }
                if (carril_bev.size() != perspectiva.size())
                    carril_bev = Mat::zeros(perspectiva.size(), CV_8UC3);
                else
                    carril_bev.setTo(0);

                poligono.clear();
                poligono.insert(poligono.end(), curva_izq.begin(), curva_izq.end());
                poligono.insert(poligono.end(), curva_der.rbegin(), curva_der.rend());
                fillPoly(carril_bev, vector<vector<Point>>{poligono}, Scalar(0,200,0));
                warpPerspective(carril_bev, carril_orig, Minv_, frame.size());
                addWeighted(frame, 1.0, carril_orig, 0.35, 0, frame);

                int lx     = evalAjuste(aj_izq_, y_base);
                int rx     = evalAjuste(aj_der_, y_base);
                int desvio = (lx + rx) / 2 - frame.cols / 2;
                double rumbo = (evalPendiente(aj_izq_, y_base) +
                                evalPendiente(aj_der_, y_base)) / 2.0 * (180.0 / CV_PI);

                geometry_msgs::msg::Vector3 msg;
                msg.x = round(desvio * 10.0) / 10.0;
                msg.y = round(rumbo  * 10.0) / 10.0;
                msg.z = 0.0;
                lane_pub_->publish(msg);
            }

            slot_.put(frame);
        }
        cap_.release();
    }

    // Detection thread — called from main()
    void detectionLoop(atomic<bool>& running) {
        Mat frame;
        while (running) {
            if (!slot_.get(frame)) continue;  // wait up to 33 ms
            detector_.processFrame(frame);
            imshow("Perception", frame);
            if (waitKey(1) == 'q') { running = false; break; }
        }
    }

private:
    // --- Helpers ---
    static int    evalAjuste   (const AjusteCarril& f, int y) { return (int)(f.A*y*y + f.B*y + f.C); }
    static double evalPendiente(const AjusteCarril& f, int y) { return 2.0*f.A*y + f.B; }

    static AjusteCarril suavizar(const AjusteCarril& p, double A, double B, double C) {
        if (!p.valido) return {A, B, C, true, 0};
        return {cfg::ALPHA*A + (1-cfg::ALPHA)*p.A,
                cfg::ALPHA*B + (1-cfg::ALPHA)*p.B,
                cfg::ALPHA*C + (1-cfg::ALPHA)*p.C, true, 0};
    }
    static AjusteCarril degradarAjuste(const AjusteCarril& p) {
        if (!p.valido) return p;
        AjusteCarril f = p;
        if (++f.cuadros_perdidos > cfg::MAX_PERDIDO) f.valido = false;
        return f;
    }
    static bool plausible(const AjusteCarril& p, double A, double B, double C, int yb) {
        if (!p.valido) return true;
        return abs((int)(A*yb*yb + B*yb + C) - evalAjuste(p, yb)) < 60;
    }
    static bool estaSobreexpuesto(const Mat& g, double umbral = 0.60) {
        Mat sat; compare(g, 250, sat, CMP_GT);
        return (double)countNonZero(sat) / (g.rows * g.cols) > umbral;
    }

    void crearMascaraBinaria(const Mat& src, Mat& dst) {
        cvtColor(src, gris_, COLOR_BGR2GRAY);
        GaussianBlur(gris_, blur_, Size(5,5), 0);
        threshold(blur_, mask_w_, blanco_min_, 255, THRESH_BINARY);
        Canny(blur_, edges_, canny_bajo_, canny_alto_);
        bitwise_or(mask_w_, edges_, dst);
        morphologyEx(dst, dst, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3,3)));
        double d = (double)countNonZero(dst) / (dst.rows * dst.cols);
        if (d > 0.25) dst.setTo(0);
    }

    Rect detectarCrosswalk(const Mat& bin) {
        cd_.tick();
        Rect roi_search(bin.cols*0.1, bin.rows*0.2, bin.cols*0.8, bin.rows*0.7);
        Mat roi = bin(roi_search);

        vector<vector<Point>> contours;
        findContours(roi, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Rect> candidates;
        for (const auto& cnt : contours) {
            Rect r = boundingRect(cnt);
            if (r.width > 30 && r.height > 5 && r.height < 50)
                candidates.push_back(r + roi_search.tl());
        }

        if (candidates.size() >= 3 && cd_.listo()) {
            Rect bounds = candidates[0];
            for (const auto& r : candidates) bounds |= r;
            crosswalk_detectado_ = true;
            return bounds;
        }

        crosswalk_detectado_ = false;
        return Rect();
    }

    void dibujarCrosswalk(Mat& frame, const Rect& box_bev) {
        if (box_bev.area() <= 0) return;
        vector<Point2f> pts_bev = {
            {(float)box_bev.x,                  (float)box_bev.y},
            {(float)(box_bev.x+box_bev.width),  (float)box_bev.y},
            {(float)(box_bev.x+box_bev.width),  (float)(box_bev.y+box_bev.height)},
            {(float)box_bev.x,                  (float)(box_bev.y+box_bev.height)}
        };
        vector<Point2f> pts_orig;
        perspectiveTransform(pts_bev, pts_orig, Minv_);
        vector<Point> pts_int;
        for (auto p : pts_orig) pts_int.push_back(p);
        polylines(frame, vector<vector<Point>>{pts_int}, true, Scalar(0,255,255), 3);
        putText(frame, "CROSSWALK", pts_int[0] - Point(0,10),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
    }

    void ventanasDeslizantes(const Mat& bin, int& xi, int& xd) {
        pts_izq_.clear(); pts_der_.clear();
        const int alto = bin.rows / cfg::N_VENTANAS;
        const int med  = bin.cols / 2;
        xi = min(xi, med - cfg::MIN_SEP/2);
        xd = max(xd, med + cfg::MIN_SEP/2);

        vector<Point> found; found.reserve(2000);
        for (int i = 0; i < cfg::N_VENTANAS; i++) {
            int yb = bin.rows - (i+1)*alto;
            int ml = max(cfg::MARGEN - i*2, 20);
            auto proc = [&](int& cx, vector<Point>& pts) {
                Rect win(cx-ml, yb, ml*2, alto);
                win &= Rect(0, 0, bin.cols, bin.rows);
                if (win.width <= 0 || win.height <= 0) return;
                found.clear(); findNonZero(bin(win), found);
                if ((int)found.size() > cfg::MIN_PIX) {
                    int sx = 0, paso = max(1,(int)found.size()/400), cnt = 0;
                    for (int j = 0; j < (int)found.size(); j += paso) {
                        Point p = found[j] + win.tl(); pts.push_back(p);
                        sx += p.x; cnt++;
                    }
                    cx = sx / cnt;
                }
            };
            proc(xi, pts_izq_); proc(xd, pts_der_);
            if (xd - xi < cfg::MIN_SEP) {
                int c = (xi+xd)/2; xi = c-cfg::MIN_SEP/2; xd = c+cfg::MIN_SEP/2;
            }
        }
    }

    bool ajustarPoli(const vector<Point>& pts, double& A, double& B, double& C) {
        int n = (int)pts.size(); if (n < 25) return false;
        vector<int> xs; xs.reserve(n);
        for (auto& p : pts) xs.push_back(p.x);
        nth_element(xs.begin(), xs.begin()+n/2, xs.end());
        int mx = xs[n/2];
        vector<Point> pf; pf.reserve(n);
        for (auto& p : pts) if (abs(p.x-mx) < 80) pf.push_back(p);
        int nf = (int)pf.size(); if (nf < 20) return false;
        int ymin = pf[0].y, ymax = pf[0].y;
        for (auto& p : pf) { ymin = min(ymin,p.y); ymax = max(ymax,p.y); }
        if (ymax - ymin < 80) return false;
        if (nf > 2000) nf = 2000;
        Mat Y = s_Y_.rowRange(0,nf), X = s_X_.rowRange(0,nf);
        for (int i = 0; i < nf; i++) {
            double y = pf[i].y;
            Y.at<double>(i,0) = y*y; Y.at<double>(i,1) = y; Y.at<double>(i,2) = 1.0;
            X.at<double>(i,0) = pf[i].x;
        }
        Mat coefs; solve(Y, X, coefs, DECOMP_SVD);
        A = coefs.at<double>(0); B = coefs.at<double>(1); C = coefs.at<double>(2);
        return true;
    }

    // --- Members ---
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr lane_pub_;
    ObjectDetector detector_;
    FrameSlot      slot_;

    VideoCapture cap_;
    Mat M_, Minv_;
    AjusteCarril aj_izq_, aj_der_;
    Cooldown cd_;
    bool crosswalk_detectado_ = false;

    vector<Point> pts_izq_, pts_der_;
    Mat s_Y_, s_X_;
    Mat gris_, blur_, mask_w_, edges_;

    int blanco_min_, canny_bajo_, canny_alto_;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = make_shared<PerceptionNode>();

    atomic<bool> running{true};

    thread det_thread([&]() {
        node->detectionLoop(running);
    });

    node->laneLoop(running);

    running = false;
    det_thread.join();

    destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}