## 1. Dependencias

```
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git libopencv-dev v4l-utils libeigen3-dev libprotobuf-dev protobuf-compiler \
ros-humble-cv-bridge ros-humble-image-transport ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-std-msgs ros-humble-rclcpp ros-humble-vision-opencv
```

## 2. NCNN 

```
cd ~/ros2_ws
git clone https://github.com/Tencent/ncnn.git && cd ncnn
git submodule update --init
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_VULKAN=OFF
make -j$(nproc) && make install
```

## 3. Clonar y Compilar el nodo

```
cd ~/ros2_ws/src
git clone -b vision_sign_lane https://github.com/Jacqueline-T/Follow_the_line.git
```

# Configurar ruta de ncnn y compilar
```
export ncnn_DIR=~/ros2_ws/ncnn/build/install/lib/cmake/ncnn
cd ~/ros2_ws
colcon build --symlink-install --packages-select lane_detection
source install/setup.bash
```

##4. Permisos de Cámara (este no se muy bien como funcione en la jetson)

```
sudo chmod 777 /dev/video0
```

# PASO 1: Calibración
Ejecuta la calibracion primero. La parte de arriba tiene que estar debajo del horizonte
y las puntas de los trapecios tocando las esquinas de los carriles (no tiene que ser preciso)

```
ros2 run lane_detection cal_node
```

Anota los valores resultantes.

# PASO 2: Detección

Ejecuta el nodo principal e ingresa los datos de calibración en la terminal cuando los pida:

```
ros2 run lane_detection lane_node
```

# PASO 3: Verificar que si se publican los datos. 
Se publicaran los datos de offset (que tan alejado esta del centro del carril) y rumbo en grados. 

```
ros2 topic echo /lane/info
```
