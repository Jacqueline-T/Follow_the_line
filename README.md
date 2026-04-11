# Que falta:
- Probar el nodo final fused_lane_detection
- Terminar de programar la detección de cruces peatonales
- Mejorar la deteccion de autos (falla demasiado, arreglar dataset)
- Mejorar la deteccion de carriles cuando hay lineas punteadas

# Dependencias

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git libopencv-dev v4l-utils libeigen3-dev libprotobuf-dev protobuf-compiler \
ros-humble-cv-bridge ros-humble-image-transport ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-std-msgs ros-humble-rclcpp ros-humble-vision-opencv
```

## NCNN

```bash
cd ~/ros2_ws
git clone https://github.com/Tencent/ncnn.git && cd ncnn
git submodule update --init
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_VULKAN=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
make -j2 && sudo make install
```

Verificar instalación:
```bash
ls /usr/local/include/ncnn/net.h
```

## Clonar y Compilar el nodo

```bash
cd ~/ros2_ws/src
git clone -b vision_sign_lane https://github.com/Jacqueline-T/Follow_the_line.git lane_detection
```

Compilar:
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select lane_detection
source install/setup.bash
```

## CONSIDERACIONES DE LA CAMARA

El código fue hecho y probado con una webcam USB. Si usas otra cámara hay que adaptar la inicialización manualmente en el `main()` de:

- `calibracion.cpp`
- `lane_detection.cpp`

Y eventualmente fused_lane_detection

---

### El código corre pero la imagen sale verde

Hay que abrir la cámara con un pipeline dif en lugar de `VideoCapture(0)`.

Ejemplo para Raspberry Pi Camera v2:

```cpp
// Webcam:
VideoCapture cap(0, CAP_V4L2);

// Pi Camera v2:
VideoCapture cap(
  "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 "
  "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink",
  CAP_GSTREAMER
);
```

Reemplaza también los `cap.set()` de resolución y FPS (eso depende de cada camara, no estoy muy segura...).

---

## Propósito de cada nodo

cal_node es para calibración. Córrelo primero y anota los valores que te da, los vas a necesitar para ajustar la perspectiva del carril.

lane_node es el de detección de carril. Publica la posición y ángulo del carril en tiempo real. Este es el que ustedes deberían usar ahorita para programar el movimiento básico.

sign_node es detección de señales y autos. publica a /car_distance — std_msgs/Float32 y
/stop_distance — std_msgs/Float32 

fused_lane_node es la fusión de `lane_node` + `sign_node` en un solo proceso. La idea es que eventualmente reemplace los dos nodos corriendo por separado. Aún no probado !!!

---

## PASO 1: Calibración

```bash
ros2 run lane_detection cal_node
```

Anota los valores resultantes.

## PASO 2: Detección

```bash
ros2 run lane_detection lane_node
```

Publica al tópico `lane/info` — `geometry_msgs/Vector3`

| campo | dato |
|-------|------|
| `x` | Desvío lateral del centro del carril (px) |
| `y` | Rumbo/ángulo promedio de las líneas (°) |
| `z` | Siempre `0.0` |

Este es el nodo de detección de objetos, detecta objetos y te manda la distancia de la camara al objeto.

```bash
ros2 run lane_detection sign_node
```
---

## Sobre fused_lane_node

Si se quieren arriesgar y probarlo:

```bash
ros2 run lane_detection fused_node
```

Van a necesitar corregir los errores que encuentren y optimizarlo. Por ahorita quédense con `lane_node` y programen el movimiento básico con eso, es lo más seguro.
