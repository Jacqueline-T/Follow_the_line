// CALIBRACIONN!!!!!
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

namespace cfg {
    constexpr double ALPHA       = 0.15;
    constexpr int    N_VENTANAS  = 12;
    constexpr int    MARGEN      = 50;
    constexpr int    MIN_PIX     = 8;
    constexpr int    MAX_PERDIDO = 25;
    constexpr int    MIN_SEP     = 100;
    constexpr int    MIN_SEP_FIT = 70;

    int ancho_sup = 190, ancho_inf = 450;
    int alto_roi  = 180, despl_y   = 300;
    int blanco_min = 190;
    int canny_bajo = 40,  canny_alto = 120;
}

struct Cooldown {
    int restante = 0;
    static constexpr int MAX = 45;
    bool listo() { return restante == 0; }
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
    for (int i=0;i<hist.rows;i++){
        int v=hist.at<int>(i,0);
        if(v>umbral_alto) filas_alto++;
        else if(v>umbral_med) filas_med++;
    }
    if(filas_alto>=1 && filas_alto<=2 && filas_med<=2){ cd.activar(); return 1; }
    if(filas_med>=4){ cd.activar(); return 2; }                                    
    return 0;
}

struct AjusteCarril {
    double A = 0, B = 0, C = 0;

    bool   valido          = false;
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


static Mat s_Y(8000, 3, CV_64F);
static Mat s_X(8000, 1, CV_64F);
static Mat s_coefs;

auto filtrarExtremos = [](vector<Point>& pts, int rows){
    pts.erase(remove_if(pts.begin(), pts.end(),
        [&](const Point& p){ return p.y < rows * 0.20f; }), pts.end());
};

bool ajustarPolinomio(const vector<Point>& pts, double& A, double& B, double& C) {
    int n = (int)pts.size();
    if (n < 25) return false;

    // --- Filtro de outlies por mediana en X ---
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

    // --- Check de rango vertical ---
    int y_min = pts_f[0].y, y_max = pts_f[0].y;
    for (auto& p : pts_f) {
        y_min = min(y_min, p.y);
        y_max = max(y_max, p.y);
    }
    if (y_max - y_min < 30) return false;

    // --- Polyfit SVD ---
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
    return abs((int)(A * y_base * y_base + B * y_base + C) - evaluarAjuste(previo, y_base)) < 90;
}


void dibujarPolinomio(Mat& img, const AjusteCarril& ajuste, Scalar color, int grosor = 2) {
    if (!ajuste.valido) return;
    
    double desvanecer = 1.0 - 0.6 * ((double)ajuste.cuadros_perdidos / cfg::MAX_PERDIDO);
    color *= desvanecer;
    
    Point previo(-1, -1);
    for (int y = 0; y < img.rows; y += 4) {
        int x = evaluarAjuste(ajuste, y);
        if (x < 0 || x >= img.cols) { previo.x = -1; continue; }
        
        Point actual(x, y);
        if (previo.x >= 0) line(img, previo, actual, color, grosor, LINE_AA);
        previo = actual;
    }
}


static Mat s_gris, s_borroso, s_bordes, s_mascara_blanca, s_binaria;

void crearMascaraBinaria(const Mat& perspectiva, Mat& salida) {
    cvtColor(perspectiva, s_gris, COLOR_BGR2GRAY);
    GaussianBlur(s_gris, s_borroso, Size(5, 5), 0);
    
    threshold(s_borroso, s_mascara_blanca, cfg::blanco_min, 255, THRESH_BINARY);
    Canny(s_borroso, s_bordes, cfg::canny_bajo, cfg::canny_alto);
    
    bitwise_or(s_mascara_blanca, s_bordes, salida);
    
    morphologyEx(salida, salida, MORPH_CLOSE, 
                 getStructuringElement(MORPH_RECT, Size(3, 3)));

    double densidad = (double)countNonZero(salida) / (double)(salida.rows * salida.cols);
    if (densidad > 0.40) salida.setTo(0);
}

bool estaSobreexpuesto(const Mat& gris, double umbral = 0.70) {
    Mat saturado;
    compare(gris, 250, saturado, CMP_GT);
    return (double)countNonZero(saturado) / (double)(gris.rows * gris.cols) > umbral;
}


static vector<Point> s_pts_izq, s_pts_der;

void ventanasDeslizantes(const Mat& binaria, int& x_izq, int& x_der, 
                         Mat& visualizacion, const AjusteCarril& aj_izq, const AjusteCarril& aj_der) {
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


        auto procesarVentana = [&](int& cx, vector<Point>& pts, Scalar color) {

            int margen_local = cfg::MARGEN - (i * 2); 
            margen_local = max(margen_local, 20);
            Rect win(cx - margen_local, y_bajo, margen_local * 2, alto_vacia);

            win &= Rect(0, 0, binaria.cols, binaria.rows);
            
            if (win.width <= 0 || win.height <= 0) return;

            encontrados.clear();
            findNonZero(binaria(win), encontrados);

            if ((int)encontrados.size() > cfg::MIN_PIX) {
                int sx = 0;
                int paso = max(1, (int)encontrados.size() / 400); 
                int cuenta = 0;
                for (int j = 0; j < (int)encontrados.size(); j += paso) {
                    Point p = encontrados[j] + win.tl();
                    pts.push_back(p);
                    sx += p.x;
                    cuenta++;
                }
                cx = sx / cuenta;
            }
            rectangle(visualizacion, win, color, 1);
        };

        procesarVentana(x_izq, s_pts_izq, Scalar(255, 0, 0));
        procesarVentana(x_der, s_pts_der, Scalar(0, 0, 255));

        if (x_der - x_izq < cfg::MIN_SEP) {
            int centro = (x_izq + x_der) / 2;
            x_izq = centro - cfg::MIN_SEP / 2;
            x_der = centro + cfg::MIN_SEP / 2;
        }
    }
}


int main() {
    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS,          30);
    
    if (!cap.isOpened()) { cerr << "Error al abrir la camara\n"; return -1; }

    namedWindow("Ajustes");
    auto cb = [](int, void*){}; 
    
    createTrackbar("Ancho Sup", "Ajustes", &cfg::ancho_sup,  600, cb);
    createTrackbar("Ancho Inf", "Ajustes", &cfg::ancho_inf,  640, cb);
    createTrackbar("Alto",  "Ajustes", &cfg::alto_roi,   480, cb);
    createTrackbar("Despl. Y",  "Ajustes", &cfg::despl_y,   480, cb);
    createTrackbar("Blanco Min","Ajustes", &cfg::blanco_min, 255, cb);
    createTrackbar("Canny Bajo","Ajustes", &cfg::canny_bajo, 255, cb);
    createTrackbar("Canny Alto","Ajustes", &cfg::canny_alto, 255, cb);

    Mat frame, perspectiva, binaria, vis, carril_bev, carril_orig;
    Mat gris_chequeo, M, Minv;
    int ant_as = -1, ant_ai = -1, ant_hr = -1, ant_dy = -1;

    AjusteCarril aj_izq, aj_der;

    const Point2f dst[4] = {{150,0}, {490,0}, {490,480}, {150,480}};
    Point2f src[4];

    vector<Point> curva_izq, curva_der, poligono;
    curva_izq.reserve(100);
    curva_der.reserve(100);
    poligono.reserve(200);

    Mat hist;

    Cooldown cd_horizontal;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (cfg::ancho_sup != ant_as || cfg::ancho_inf != ant_ai || 
            cfg::alto_roi  != ant_hr || cfg::despl_y   != ant_dy) {
            
            float cx = 320.f;
            src[0] = {cx - cfg::ancho_sup / 2.f, (float)cfg::despl_y};
            src[1] = {cx + cfg::ancho_sup / 2.f, (float)cfg::despl_y};
            src[2] = {cx + cfg::ancho_inf / 2.f, (float)(cfg::despl_y + cfg::alto_roi)};
            src[3] = {cx - cfg::ancho_inf / 2.f, (float)(cfg::despl_y + cfg::alto_roi)};
            
            M    = getPerspectiveTransform(src, dst);
            Minv = getPerspectiveTransform(dst, src);
            
            ant_as = cfg::ancho_sup; ant_ai = cfg::ancho_inf; 
            ant_hr = cfg::alto_roi;  ant_dy = cfg::despl_y;
        }

        warpPerspective(frame, perspectiva, M, Size(640, 480));
        crearMascaraBinaria(perspectiva, binaria);
        int8_t evento = detectarEventoHorizontal(binaria, cd_horizontal);        

        cvtColor(frame, gris_chequeo, COLOR_BGR2GRAY);
        if (estaSobreexpuesto(gris_chequeo)) {
            aj_izq = degradarAjuste(aj_izq);
            aj_der = degradarAjuste(aj_der);
            
            continue;
        }

        Mat mitad_inferior = binaria(Range(binaria.rows / 2, binaria.rows), Range::all());
        reduce(mitad_inferior, hist, 0, REDUCE_SUM, CV_32S);

        double minV, maxV; Point minLoc, maxLoc;
        minMaxLoc(hist.colRange(0, 320), &minV, &maxV, &minLoc, &maxLoc);
        int x_izq_actual = maxLoc.x;
        
        minMaxLoc(hist.colRange(320, 640), &minV, &maxV, &minLoc, &maxLoc);
        int x_der_actual = maxLoc.x + 320;

        const int y_base = binaria.rows - 1;
        if (aj_izq.valido) x_izq_actual = evaluarAjuste(aj_izq, y_base);
        if (aj_der.valido) x_der_actual = evaluarAjuste(aj_der, y_base);

        perspectiva.copyTo(vis);
        ventanasDeslizantes(binaria, x_izq_actual, x_der_actual, vis, aj_izq, aj_der);

        filtrarExtremos(s_pts_izq, binaria.rows);
        filtrarExtremos(s_pts_der, binaria.rows);

        double lA, lB, lC, rA, rB, rC;
        const bool det_izq = ajustarPolinomio(s_pts_izq, lA, lB, lC);
        const bool det_der = ajustarPolinomio(s_pts_der, rA, rB, rC);

        if (det_izq && det_der) {
            int lx = (int)(lA * y_base * y_base + lB * y_base + lC);
            int rx = (int)(rA * y_base * y_base + rB * y_base + rC);
            
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
                curva_izq.push_back ({clamp(evaluarAjuste(aj_izq, y), 0, perspectiva.cols - 1), y});
                curva_der.push_back ({clamp(evaluarAjuste(aj_der, y), 0, perspectiva.cols - 1), y});
            }

            if (carril_bev.size() != perspectiva.size())
                carril_bev = Mat::zeros(perspectiva.size(), CV_8UC3);
            else
                carril_bev.setTo(0);

            poligono.clear();
            poligono.insert(poligono.end(), curva_izq.begin(), curva_izq.end());
            poligono.insert(poligono.end(), curva_der.rbegin(), curva_der.rend());
            
            fillPoly(carril_bev, vector<vector<Point>>{poligono}, Scalar(0, 200, 0));

            dibujarPolinomio(vis, aj_izq, Scalar(255, 0, 0), 3);
            dibujarPolinomio(vis, aj_der, Scalar(0, 0, 255), 3);

            warpPerspective(carril_bev, carril_orig, Minv, frame.size());
            addWeighted(frame, 1.0, carril_orig, 0.35, 0, frame);

            const int lx      = evaluarAjuste(aj_izq, y_base);
            const int rx      = evaluarAjuste(aj_der, y_base);
            const int desvio  = (lx + rx) / 2 - frame.cols / 2;
            const double rumbo = (evaluarPendiente(aj_izq, y_base) + evaluarPendiente(aj_der, y_base)) / 2.0 * (180.0 / CV_PI);

            string info = "Desvio: " + to_string(desvio) + "px | Rumbo: " + to_string((int)rumbo) + "deg";
            putText(frame, info, Point(10, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, LINE_AA);
        }

        for (int i = 0; i < 4; i++)
            line(frame, src[i], src[(i + 1) % 4], Scalar(0, 255, 255), 2, LINE_AA);

        imshow("1. Original", frame);
        imshow("2. Vista Pajaro", vis);
        imshow("3. Binaria", binaria);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}