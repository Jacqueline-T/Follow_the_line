#include <ESP32Servo.h>

// -----------------------------------------------------
// --- Servo ---
Servo myServo;
const int SERVO_PIN = 18;

// --- Steering Angles ---
const int SERVO_CENTER = 81;   // Straight
const int SERVO_LEFT   = 36;   // Left turn
const int SERVO_RIGHT  = 144;  // Right turn

// -----------------------------------------------------
// --- Motor A ---
const int ENA = 14;
const int IN1 = 27;
const int IN2 = 26;

// --- Motor B ---
const int IN3 = 25;
const int IN4 = 33;
const int ENB = 32;

// --- PWM Settings ---
const int PWM_FREQ = 1000;
const int PWM_CHANNEL_A = 2;
const int PWM_CHANNEL_B = 3;
const int PWM_RESO = 8;

// --- Speed ---
const int SPEED_A = 180;    // Can be from 0-255
const int SPEED_B = 100;

// -----------------------------------------------------
// --- Move Forward ---
void motorForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(PWM_CHANNEL_A, SPEED_A);
  ledcWrite(PWM_CHANNEL_B, SPEED_B);
}

// --- Move Backward ---
void motorBackward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(PWM_CHANNEL_A, SPEED_A);
  ledcWrite(PWM_CHANNEL_B, SPEED_B);
}

// --- Stop Moving ---
void motorStop() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(PWM_CHANNEL_A, 0);
  ledcWrite(PWM_CHANNEL_B, 0);
}

// -----------------------------------------------------
void setup() {
  Serial.begin(115200);

  // Motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // PWM Channel
  ledcSetup(PWM_CHANNEL_A, PWM_FREQ, PWM_RESO);
  ledcSetup(PWM_CHANNEL_B, PWM_FREQ, PWM_RESO);
  ledcAttachPin(ENA, PWM_CHANNEL_A);
  ledcAttachPin(ENB, PWM_CHANNEL_B);

  // Servo
  myServo.attach(SERVO_PIN);
  myServo.write(SERVO_CENTER);

  delay(1000);
}

void loop() {
  // Forward
  Serial.println("Forward...");
  myServo.write(SERVO_CENTER);
  motorForward();
  delay(2000);

  // Stop
  Serial.println("Stopping...");
  myServo.write(SERVO_LEFT);
  motorStop();
  delay(1000);

  // Backward
  Serial.println("Backward...");
  myServo.write(SERVO_CENTER);
  motorBackward();
  delay(2000);

  // Stop
  Serial.println("Stopping...");
  myServo.write(SERVO_RIGHT);
  motorStop();
  delay(1000);
}
