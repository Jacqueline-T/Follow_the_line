#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>

using namespace cv;
using namespace std;

namespace cfg {
    constexpr double ALPHA       = 0.3;
    constexpr int    N_VENTANAS  = 9;
    constexpr int    MARGEN      = 50;
    constexpr int    MIN_PIX     = 50;
    constexpr int    MAX_PERDIDO = 12;
    constexpr int    MIN_SEP     = 100;
    constexpr int    MIN_SEP_FIT = 150;

    constexpr int ancho_sup  = 200, ancho_inf = 450;
    constexpr int alto_roi   = 180, despl_y   = 300;
    constexpr int blanco_min = 200;
    constexpr int canny_bajo = 40,  canny_alto = 120;
}

struct Cooldown {
    int restante = 0;
    static constexpr int MAX = 45;
    bool listo()   { return restante == 0; }
    void activar() { restante = MAX; }
    void tick()    { if (restante > 0) restante--; }
};

int8_t detectarEventoHorizontal(const Mat& binaria, Cooldown& cd) {
    cd.tick();
    if (!cd.listo()) return 0;
    Mat roi = binaria(Range(binaria.rows*2/3, binaria.rows), Range::all());
    Mat hist; reduce(roi, hist, 1, REDUCE_SUM, CV_32S);
    int umbral_alto = (int)(binaria.cols*0.55f);
    int umbral_med  = (int)(binaria.cols*0.40f);
    int filas_alto=0, filas_med=0;
    for (int i=0; i<hist.rows; i++) {
        int v = hist.at<int>(i,0);
        if      (v > umbral_alto) filas_alto++;
        else if (v > umbral_med)  filas_med++;
    }
    if (filas_alto>=1 && filas_alto<=2 && filas_med<=2) { cd.activar(); return 1; }
    if (filas_med>=4)                                    { cd.activar(); return 2; }
    return 0;
}

struct AjusteCarril {
    double A = 0, B = 0, C = 0;
    bool   valido           = false;
    int    cuadros_perdidos = 0;
};

inline int    evaluarAjuste   (const AjusteCarril& f, int y) { return (int)(f.A*y*y + f.B*y + f.C); }
inline double evaluarPendiente(const AjusteCarril& f, int y) { return 2.0*f.A*y + f.B; }

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

static Mat s_Y(8000, 3, CV_64F);
static Mat s_X(8000, 1, CV_64F);
static Mat s_coefs;

bool ajustarPolinomio(const vector<Point>& pts, double& A, double& B, double& C) {
    int n = (int)pts.size();
    if (n < 25) return false;

    vector<int> xs;
    xs.reserve(n);
    for (auto& p : pts) xs.push_back(p.x);
    nth_element(xs.begin(), xs.begin() + n/2, xs.end());
    int med_x = xs[n/2];

    vector<Point> pts_f;
    pts_f.reserve(n);
    for (auto& p : pts)
        if (abs(p.x - med_x) < 80)
            pts_f.push_back(p);

    int nf = (int)pts_f.size();
    if (nf < 20) return false;

    int y_min = pts_f[0].y, y_max = pts_f[0].y;
    for (auto& p : pts_f) { y_min = min(y_min, p.y); y_max = max(y_max, p.y); }
    if (y_max - y_min < 80) return false;

    if (nf > 2000) nf = 2000;

    Mat Y = s_Y.rowRange(0, nf);
    Mat X = s_X.rowRange(0, nf);
    for (int i = 0; i < nf; i++) {
        double y = pts_f[i].y;
        Y.at<double>(i, 0) = y * y;
        Y.at<double>(i, 1) = y;
        Y.at<double>(i, 2) = 1.0;
        X.at<double>(i, 0) = pts_f[i].x;
    }
    solve(Y, X, s_coefs, DECOMP_SVD);
    A = s_coefs.at<double>(0);
    B = s_coefs.at<double>(1);
    C = s_coefs.at<double>(2);
    return true;
}

bool esAjustePlausible(const AjusteCarril& previo, double A, double B, double C, int y_base) {
    if (!previo.valido) return true;
    return abs((int)(A*y_base*y_base + B*y_base + C) - evaluarAjuste(previo, y_base)) < 60;
}

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

static vector<Point> s_pts_izq, s_pts_der;

void ventanasDeslizantes(const Mat& binaria, int& x_izq, int& x_der) {
    s_pts_izq.clear();
    s_pts_der.clear();

    const int alto_vacia = binaria.rows / cfg::N_VENTANAS;
    const int medio      = binaria.cols / 2;

    x_izq = min(x_izq, medio - cfg::MIN_SEP / 2);
    x_der = max(x_der, medio + cfg::MIN_SEP / 2);

    static vector<Point> encontrados;
    encontrados.reserve(2000);

    for (int i = 0; i < cfg::N_VENTANAS; i++) {
        const int y_bajo      = binaria.rows - (i + 1) * alto_vacia;
        int margen_local      = max(cfg::MARGEN - (i * 2), 20);

        auto procesarVentana = [&](int& cx, vector<Point>& pts) {
            Rect win(cx - margen_local, y_bajo, margen_local * 2, alto_vacia);
            win &= Rect(0, 0, binaria.cols, binaria.rows);
            if (win.width <= 0 || win.height <= 0) return;

            encontrados.clear();
            findNonZero(binaria(win), encontrados);

            if ((int)encontrados.size() > cfg::MIN_PIX) {
                int sx = 0, paso = max(1, (int)encontrados.size() / 400), cuenta = 0;
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

rclcpp::Node::SharedPtr g_node;
rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr g_pub;

int main(int argc, char** argv) {
    VideoCapture cap(1, CAP_V4L2);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS,          30);
    if (!cap.isOpened()) { cerr << "Error al abrir la camara\n"; return -1; }

    const Point2f dst[4] = {{150,0},{490,0},{490,480},{150,480}};
    const float   cx_mid = 320.f;
    const Point2f src[4] = {
        {cx_mid - cfg::ancho_sup/2.f, (float)cfg::despl_y},
        {cx_mid + cfg::ancho_sup/2.f, (float)cfg::despl_y},
        {cx_mid + cfg::ancho_inf/2.f, (float)(cfg::despl_y + cfg::alto_roi)},
        {cx_mid - cfg::ancho_inf/2.f, (float)(cfg::despl_y + cfg::alto_roi)}
    };

    Mat M    = getPerspectiveTransform(src, dst);
    Mat Minv = getPerspectiveTransform(dst, src);

    Mat frame, perspectiva, binaria, gris_chequeo, carril_bev, carril_orig;
    AjusteCarril aj_izq, aj_der;

    vector<Point> curva_izq, curva_der, poligono;
    curva_izq.reserve(100);
    curva_der.reserve(100);
    poligono.reserve(200);

    rclcpp::init(argc, argv);
    g_node = rclcpp::Node::make_shared("lane_detection");
    g_pub  = g_node->create_publisher<geometry_msgs::msg::Vector3>("lane/info", 10);

    Cooldown cd_horizontal;
    Mat hist;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        warpPerspective(frame, perspectiva, M, Size(640, 480));
        crearMascaraBinaria(perspectiva, binaria);
        detectarEventoHorizontal(binaria, cd_horizontal);

        cvtColor(frame, gris_chequeo, COLOR_BGR2GRAY);
        if (estaSobreexpuesto(gris_chequeo)) {
            aj_izq = degradarAjuste(aj_izq);
            aj_der = degradarAjuste(aj_der);
            imshow("Lane Detection", frame);
            if (waitKey(1) == 'q') break;
            continue;
        }

        Mat mitad_inferior = binaria(Range(binaria.rows/2, binaria.rows), Range::all());
        reduce(mitad_inferior, hist, 0, REDUCE_SUM, CV_32S);

        double minV, maxV; Point minLoc, maxLoc;
        minMaxLoc(hist.colRange(0, 320),   &minV, &maxV, &minLoc, &maxLoc);
        int x_izq_actual = maxLoc.x;
        minMaxLoc(hist.colRange(320, 640), &minV, &maxV, &minLoc, &maxLoc);
        int x_der_actual = maxLoc.x + 320;

        const int y_base = binaria.rows - 1;
        if (aj_izq.valido) x_izq_actual = evaluarAjuste(aj_izq, y_base);
        if (aj_der.valido) x_der_actual = evaluarAjuste(aj_der, y_base);

        ventanasDeslizantes(binaria, x_izq_actual, x_der_actual);

        double lA, lB, lC, rA, rB, rC;
        const bool det_izq = ajustarPolinomio(s_pts_izq, lA, lB, lC);
        const bool det_der = ajustarPolinomio(s_pts_der, rA, rB, rC);

        if (det_izq && det_der) {
            int lx = (int)(lA*y_base*y_base + lB*y_base + lC);
            int rx = (int)(rA*y_base*y_base + rB*y_base + rC);
            if (rx - lx < cfg::MIN_SEP_FIT) {
                aj_izq = degradarAjuste(aj_izq);
                aj_der = degradarAjuste(aj_der);
            } else {
                aj_izq = suavizarAjuste(aj_izq, lA, lB, lC);
                aj_der = suavizarAjuste(aj_der, rA, rB, rC);
            }
        } else {
            aj_izq = det_izq && esAjustePlausible(aj_izq, lA, lB, lC, y_base)
                     ? suavizarAjuste(aj_izq, lA, lB, lC) : degradarAjuste(aj_izq);
            aj_der = det_der && esAjustePlausible(aj_der, rA, rB, rC, y_base)
                     ? suavizarAjuste(aj_der, rA, rB, rC) : degradarAjuste(aj_der);
        }

        if (aj_izq.valido && aj_der.valido) {
            curva_izq.clear();
            curva_der.clear();
            for (int y = 0; y < perspectiva.rows; y += 5) {
                curva_izq.push_back({clamp(evaluarAjuste(aj_izq, y), 0, perspectiva.cols-1), y});
                curva_der.push_back({clamp(evaluarAjuste(aj_der, y), 0, perspectiva.cols-1), y});
            }

            if (carril_bev.size() != perspectiva.size())
                carril_bev = Mat::zeros(perspectiva.size(), CV_8UC3);
            else
                carril_bev.setTo(0);

            poligono.clear();
            poligono.insert(poligono.end(), curva_izq.begin(), curva_izq.end());
            poligono.insert(poligono.end(), curva_der.rbegin(), curva_der.rend());
            fillPoly(carril_bev, vector<vector<Point>>{poligono}, Scalar(0, 200, 0));

            warpPerspective(carril_bev, carril_orig, Minv, frame.size());
            addWeighted(frame, 1.0, carril_orig, 0.35, 0, frame);

            const int lx     = evaluarAjuste(aj_izq, y_base);
            const int rx     = evaluarAjuste(aj_der, y_base);
            const int desvio = (lx + rx) / 2 - frame.cols / 2;
            const double rumbo = (evaluarPendiente(aj_izq, y_base) + evaluarPendiente(aj_der, y_base)) / 2.0 * (180.0 / CV_PI);

            geometry_msgs::msg::Vector3 msg;
            msg.x = round((double)desvio * 10.0) / 10.0;
            msg.y = round(rumbo * 10.0) / 10.0;
            msg.z = 0.0;
            g_pub->publish(msg);
            rclcpp::spin_some(g_node);
        }

        imshow("Lane Detection", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}