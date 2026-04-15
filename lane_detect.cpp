// Lane Detection — ROS2 Node (no VideoCapture)
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace cv;
using namespace std;

// ─────────────────────────────────────────────
//  CONFIG
// ─────────────────────────────────────────────
namespace cfg {
    constexpr double ALPHA       = 0.3;
    constexpr int    N_VENTANAS  = 9;
    constexpr int    MARGEN      = 50;
    constexpr int    MIN_PIX     = 50;
    constexpr int    MAX_PERDIDO = 12;
    constexpr int    MIN_SEP     = 100;
    constexpr int    MIN_SEP_FIT = 150;

    int ancho_sup  = 200, ancho_inf = 450;
    int alto_roi   = 180, despl_y   = 300;
    int blanco_min = 200;
    int canny_bajo = 40,  canny_alto = 120;
}

// ─────────────────────────────────────────────
//  LANE FIT
// ─────────────────────────────────────────────
struct AjusteCarril {
    double A = 0, B = 0, C = 0;
    bool   valido           = false;
    int    cuadros_perdidos = 0;
};

inline int evaluarAjuste(const AjusteCarril& f, int y) {
    return (int)(f.A * y * y + f.B * y + f.C);
}

inline double evaluarPendiente(const AjusteCarril& f, int y) {
    return 2.0 * f.A * y + f.B;
}

AjusteCarril suavizarAjuste(const AjusteCarril& previo, double A, double B, double C) {
    if (!previo.valido) return {A, B, C, true, 0};
    return {
        cfg::ALPHA * A + (1 - cfg::ALPHA) * previo.A,
        cfg::ALPHA * B + (1 - cfg::ALPHA) * previo.B,
        cfg::ALPHA * C + (1 - cfg::ALPHA) * previo.C,
        true, 0
    };
}

AjusteCarril degradarAjuste(const AjusteCarril& previo) {
    if (!previo.valido) return previo;
    AjusteCarril f = previo;
    f.cuadros_perdidos++;
    if (f.cuadros_perdidos > cfg::MAX_PERDIDO) f.valido = false;
    return f;
}

// ─────────────────────────────────────────────
//  POLYNOMIAL FIT
// ─────────────────────────────────────────────
static Mat s_Y(8000, 3, CV_64F);
static Mat s_X(8000, 1, CV_64F);
static Mat s_coefs;

bool ajustarPolinomio(const vector<Point>& pts, double& A, double& B, double& C) {
    int n = (int)pts.size();
    if (n < 25) return false;
    if (n > 2000) n = 2000;

    int y_min = pts[0].y, y_max = pts[0].y;
    for (int i = 0; i < n; i++) {
        y_min = min(y_min, pts[i].y);
        y_max = max(y_max, pts[i].y);
    }
    if (y_max - y_min < 80) return false;

    Mat Y = s_Y.rowRange(0, n);
    Mat X = s_X.rowRange(0, n);

    for (int i = 0; i < n; i++) {
        double y = pts[i].y;
        Y.at<double>(i, 0) = y * y;
        Y.at<double>(i, 1) = y;
        Y.at<double>(i, 2) = 1.0;
        X.at<double>(i, 0) = pts[i].x;
    }
    solve(Y, X, s_coefs, DECOMP_SVD);
    A = s_coefs.at<double>(0);
    B = s_coefs.at<double>(1);
    C = s_coefs.at<double>(2);
    return true;
}

bool esAjustePlausible(const AjusteCarril& previo, double A, double B, double C, int y_base) {
    if (!previo.valido) return true;
    return abs((int)(A * y_base * y_base + B * y_base + C) - evaluarAjuste(previo, y_base)) < 60;
}

// ─────────────────────────────────────────────
//  BINARY MASK
// ─────────────────────────────────────────────
static Mat s_gris, s_borroso, s_bordes, s_mascara_blanca;

void crearMascaraBinaria(const Mat& perspectiva, Mat& salida) {
    cvtColor(perspectiva, s_gris, COLOR_BGR2GRAY);
    GaussianBlur(s_gris, s_borroso, Size(5, 5), 0);
    threshold(s_borroso, s_mascara_blanca, cfg::blanco_min, 255, THRESH_BINARY);
    Canny(s_borroso, s_bordes, cfg::canny_bajo, cfg::canny_alto);
    bitwise_or(s_mascara_blanca, s_bordes, salida);
    morphologyEx(salida, salida, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(3, 3)));
    double densidad = (double)countNonZero(salida) / (double)(salida.rows * salida.cols);
    if (densidad > 0.25) salida.setTo(0);
}

bool estaSobreexpuesto(const Mat& gris, double umbral = 0.60) {
    Mat saturado;
    compare(gris, 250, saturado, CMP_GT);
    return (double)countNonZero(saturado) / (double)(gris.rows * gris.cols) > umbral;
}

// ─────────────────────────────────────────────
//  SLIDING WINDOWS
// ─────────────────────────────────────────────
static vector<Point> s_pts_izq, s_pts_der;

void ventanasDeslizantes(const Mat& binaria, int& x_izq, int& x_der,
                         const AjusteCarril& aj_izq, const AjusteCarril& aj_der) {
    s_pts_izq.clear();
    s_pts_der.clear();

    const int alto_vacia = binaria.rows / cfg::N_VENTANAS;
    const int medio      = binaria.cols / 2;

    x_izq = min(x_izq, medio - cfg::MIN_SEP / 2);
    x_der = max(x_der, medio + cfg::MIN_SEP / 2);

    static vector<Point> encontrados;
    encontrados.reserve(2000);

    for (int i = 0; i < cfg::N_VENTANAS; i++) {
        const int y_bajo = binaria.rows - (i + 1) * alto_vacia;

        auto procesarVentana = [&](int& cx, vector<Point>& pts) {
            Rect win(cx - cfg::MARGEN, y_bajo, cfg::MARGEN * 2, alto_vacia);
            win &= Rect(0, 0, binaria.cols, binaria.rows);
            if (win.width <= 0 || win.height <= 0) return;

            encontrados.clear();
            findNonZero(binaria(win), encontrados);

            if ((int)encontrados.size() > cfg::MIN_PIX) {
                int sx    = 0;
                int paso  = max(1, (int)encontrados.size() / 400);
                int cuenta = 0;
                for (int j = 0; j < (int)encontrados.size(); j += paso) {
                    Point p = encontrados[j] + win.tl();
                    pts.push_back(p);
                    sx += p.x;
                    cuenta++;
                }
                cx = sx / cuenta;
            }
        };

        procesarVentana(x_izq, s_pts_izq);
        procesarVentana(x_der, s_pts_der);

        if (x_der - x_izq < cfg::MIN_SEP) {
            int centro = (x_izq + x_der) / 2;
            x_izq = centro - cfg::MIN_SEP / 2;
            x_der = centro + cfg::MIN_SEP / 2;
        }
    }
}

// ─────────────────────────────────────────────
//  CROSSWALK DETECTION
// ─────────────────────────────────────────────
bool detectarPasoDePersonas(const Mat& perspectiva, Mat& gris_out) {
    // 1. Isolate bright white areas only
    Mat gris, blanco;
    cvtColor(perspectiva, gris, COLOR_BGR2GRAY);
    threshold(gris, blanco, 180, 255, THRESH_BINARY);

    // 2. Collapse each row to a single value (avg brightness across width)
    Mat perfil;
    reduce(blanco, perfil, 1, REDUCE_AVG, CV_32F); // shape: (rows, 1)

    // 3. Count rows that are "mostly white" (crosswalk stripe)
    int franjas = 0;
    bool en_franja = false;
    vector<int> alturas_franjas;

    for (int y = 0; y < perfil.rows; y++) {
        float val = perfil.at<float>(y, 0);
        if (val > 40.f) {  // row is bright enough to be a stripe
            if (!en_franja) {
                franjas++;
                alturas_franjas.push_back(y);
                en_franja = true;
            }
        } else {
            en_franja = false;
        }
    }

    // 4. Crosswalk = at least 3 distinct horizontal stripes
    gris_out = blanco;
    return franjas >= 3;
}

// ─────────────────────────────────────────────
//  ROS2 NODE
// ─────────────────────────────────────────────
class LaneNode : public rclcpp::Node {
public:
    LaneNode() : Node("lane_detection") {

        // ── Perspective matrices (built once from cfg) ──
        rebuildTransform();

        // ── Sub / Pub ────────────────────────────
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10,
            std::bind(&LaneNode::image_callback, this, std::placeholders::_1)
        );
        pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("lane/info", 10);

        // ── Windows & trackbars ──────────────────
        namedWindow("Lane Detection");
        namedWindow("Ajustes");

        auto cb = [](int, void* userdata) {
            // Rebuild transform whenever a slider moves
            reinterpret_cast<LaneNode*>(userdata)->rebuildTransform();
        };
        createTrackbar("Ancho Sup",  "Ajustes", &cfg::ancho_sup,  600, cb, this);
        createTrackbar("Ancho Inf",  "Ajustes", &cfg::ancho_inf,  640, cb, this);
        createTrackbar("Alto",       "Ajustes", &cfg::alto_roi,   480, cb, this);
        createTrackbar("Despl. Y",   "Ajustes", &cfg::despl_y,    480, cb, this);
        createTrackbar("Blanco Min", "Ajustes", &cfg::blanco_min, 255, cb, this);
        createTrackbar("Canny Bajo", "Ajustes", &cfg::canny_bajo, 255, cb, this);
        createTrackbar("Canny Alto", "Ajustes", &cfg::canny_alto, 255, cb, this);

        RCLCPP_INFO(this->get_logger(), "LaneNode ready, waiting for /image_raw ...");
    }

private:
    // ── Rebuild perspective transform from current cfg ──
    void rebuildTransform() {
        const float cx = 320.f; 
        Point2f src[4] = {
            {cx - cfg::ancho_sup / 2.f, (float)cfg::despl_y},
            {cx + cfg::ancho_sup / 2.f, (float)cfg::despl_y},
            {cx + cfg::ancho_inf / 2.f, (float)(cfg::despl_y + cfg::alto_roi)},
            {cx - cfg::ancho_inf / 2.f, (float)(cfg::despl_y + cfg::alto_roi)}
        };
        const Point2f dst[4] = {{150,0},{490,0},{490,480},{150,480}};
        M_    = getPerspectiveTransform(src, dst);
        Minv_ = getPerspectiveTransform(dst, src);
    }

    // ── Main pipeline ────────────────────────────
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        if (frame.empty()) return;

        // Bird's-eye warp
        Mat perspectiva, binaria;
        warpPerspective(frame, perspectiva, M_, Size(640, 480));
        // ── Crosswalk detection ──────────────────────
        Mat debug_blanco;
        bool hay_cruce = detectarPasoDePersonas(perspectiva, debug_blanco);
        if (hay_cruce) {
            putText(frame, "CROSSWALK DETECTED", Point(10, 60),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
        }
        crearMascaraBinaria(perspectiva, binaria);
        imshow("Bird's Eye", perspectiva);
        imshow("Binary", binaria);


        // Overexposure guard
        Mat gris_chequeo;
        cvtColor(frame, gris_chequeo, COLOR_BGR2GRAY);
        if (estaSobreexpuesto(gris_chequeo)) {
            aj_izq_ = degradarAjuste(aj_izq_);
            aj_der_ = degradarAjuste(aj_der_);
            imshow("Lane Detection", frame);
            waitKey(1);
            return;
        }

        // Histogram seed
        Mat hist;
        Mat mitad_inferior = binaria(Range(binaria.rows / 2, binaria.rows), Range::all());
        reduce(mitad_inferior, hist, 0, REDUCE_SUM, CV_32S);

        double minV, maxV; Point minLoc, maxLoc;
        minMaxLoc(hist.colRange(0, 320),   &minV, &maxV, &minLoc, &maxLoc);
        int x_izq_actual = maxLoc.x;
        minMaxLoc(hist.colRange(320, 640), &minV, &maxV, &minLoc, &maxLoc);
        int x_der_actual = maxLoc.x + 320;

        const int y_base = binaria.rows - 1;
        if (aj_izq_.valido) x_izq_actual = evaluarAjuste(aj_izq_, y_base);
        if (aj_der_.valido) x_der_actual = evaluarAjuste(aj_der_, y_base);

        // Sliding windows
        ventanasDeslizantes(binaria, x_izq_actual, x_der_actual, aj_izq_, aj_der_);

        // Polynomial fit
        double lA, lB, lC, rA, rB, rC;
        const bool det_izq = ajustarPolinomio(s_pts_izq, lA, lB, lC);
        const bool det_der = ajustarPolinomio(s_pts_der, rA, rB, rC);

        if (det_izq && det_der) {
            int lx = (int)(lA * y_base * y_base + lB * y_base + lC);
            int rx = (int)(rA * y_base * y_base + rB * y_base + rC);
            if (rx - lx < cfg::MIN_SEP_FIT) {
                aj_izq_ = degradarAjuste(aj_izq_);
                aj_der_ = degradarAjuste(aj_der_);
            } else {
                aj_izq_ = suavizarAjuste(aj_izq_, lA, lB, lC);
                aj_der_ = suavizarAjuste(aj_der_, rA, rB, rC);
            }
        } else {
            aj_izq_ = det_izq && esAjustePlausible(aj_izq_, lA, lB, lC, y_base)
                      ? suavizarAjuste(aj_izq_, lA, lB, lC) : degradarAjuste(aj_izq_);
            aj_der_ = det_der && esAjustePlausible(aj_der_, rA, rB, rC, y_base)
                      ? suavizarAjuste(aj_der_, rA, rB, rC) : degradarAjuste(aj_der_);
        }

        // Lane overlay + publish
        if (aj_izq_.valido && aj_der_.valido) {
            vector<Point> curva_izq, curva_der, poligono;
            for (int y = 0; y < perspectiva.rows; y += 5) {
                curva_izq.push_back({clamp(evaluarAjuste(aj_izq_, y), 0, perspectiva.cols - 1), y});
                curva_der.push_back({clamp(evaluarAjuste(aj_der_, y), 0, perspectiva.cols - 1), y});
            }

            Mat carril_bev = Mat::zeros(perspectiva.size(), CV_8UC3);
            poligono.insert(poligono.end(), curva_izq.begin(), curva_izq.end());
            poligono.insert(poligono.end(), curva_der.rbegin(), curva_der.rend());
            fillPoly(carril_bev, vector<vector<Point>>{poligono}, Scalar(0, 200, 0));

            Mat carril_orig;
            warpPerspective(carril_bev, carril_orig, Minv_, frame.size());
            addWeighted(frame, 1.0, carril_orig, 0.35, 0, frame);

            const int    lx     = evaluarAjuste(aj_izq_, y_base);
            const int    rx     = evaluarAjuste(aj_der_, y_base);
            const int    desvio = (lx + rx) / 2 - frame.cols / 2;
            const double rumbo  = (evaluarPendiente(aj_izq_, y_base) +
                                   evaluarPendiente(aj_der_, y_base)) / 2.0 * (180.0 / CV_PI);

            geometry_msgs::msg::Vector3 out;
            out.x = std::round((double)desvio * 10.0) / 10.0;
            out.y = std::round(rumbo           * 10.0) / 10.0;
            out.z = hay_cruce ? 1.0 : 0.0;  // was hardcoded 0.0
            pub_->publish(out);

            string info = "Desvio: " + to_string(desvio) + "px  Rumbo: " + to_string((int)rumbo) + "deg";
            putText(frame, info, Point(10, 35), FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(0, 255, 0), 1, LINE_AA);
        }

        imshow("Lane Detection", frame);
        waitKey(1);
    }

    // ── Members ──────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_;

    Mat M_, Minv_;
    AjusteCarril aj_izq_, aj_der_;
};

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaneNode>());
    rclcpp::shutdown();
    return 0;
}