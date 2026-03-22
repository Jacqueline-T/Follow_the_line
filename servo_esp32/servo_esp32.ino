#include <ESP32Servo.h>

Servo myServo;

const int SERVO_PIN = 18;

void setup() {
  myServo.attach(SERVO_PIN);
  myServo.write(90); // Start at center position
}

void loop() {
  // Sweep from 0° to 180°
  for (int angle = 0; angle <= 180; angle += 1) {
    myServo.write(angle);
    delay(15);
  }

  // Sweep back from 180° to 0°
  for (int angle = 180; angle >= 0; angle -= 1) {
    myServo.write(angle);
    delay(15);
  }
}
