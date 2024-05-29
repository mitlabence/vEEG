
#define NOTE 16000  // in Hz
#define DELAY 30000 // in ms
#define SIGNAL_LENGTH 200 // in ms
#define SOUND_OUT_PIN 2
#define IR_OUT_PIN 3  // pin of infrared LED
#define VIS_OUT_PIN 4 // pin of visible LED
#define TRIGGER_OUT_PIN 5 // pin of EEG trigger

void setup() {
  pinMode(IR_OUT_PIN, OUTPUT);
  pinMode(IR_OUT_PIN, OUTPUT);
  pinMode(TRIGGER_OUT_PIN, OUTPUT);
}

void loop() {
  tone(SOUND_OUT_PIN, NOTE);
  digitalWrite(IR_OUT_PIN, HIGH);
  digitalWrite(VIS_OUT_PIN, HIGH);
  digitalWrite(TRIGGER_OUT_PIN, HIGH);
  delay(SIGNAL_LENGTH);
  noTone(SOUND_OUT_PIN);
  digitalWrite(IR_OUT_PIN, LOW);
  digitalWrite(VIS_OUT_PIN, LOW);
  digitalWrite(TRIGGER_OUT_PIN, LOW);
  delay(DELAY);
}