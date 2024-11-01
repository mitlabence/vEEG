#define NOTE 16000  // in Hz
#define DELAY 3600000 // in ms (Set for 1 Hour)
#define SIGNAL_LENGTH 500 // in ms
#define SOUND_OUT_PIN 2
#define IR_OUT_PIN 3  // pin of infrared LED
#define VIS_OUT_PIN 4 // pin of visible LED
#define TRIGGER_OUT_PIN 5 // pin of EEG trigger
#define IR_OUT_2_PIN 14  // pin of infrared LED
#define IR_OUT_3_PIN 16  // pin of infrared LED

long myTimer = -DELAY;
void setup() {
  pinMode(IR_OUT_PIN, OUTPUT);
  pinMode(IR_OUT_2_PIN, OUTPUT);
  pinMode(IR_OUT_3_PIN, OUTPUT);
  pinMode(VIS_OUT_PIN, OUTPUT);
  pinMode(TRIGGER_OUT_PIN, OUTPUT);
}
void loop() {
  if (millis() > DELAY + myTimer ) {
    myTimer = millis();
    tone(SOUND_OUT_PIN, NOTE);
    digitalWrite(IR_OUT_PIN, HIGH);
    digitalWrite(IR_OUT_2_PIN, HIGH);
    digitalWrite(IR_OUT_3_PIN, HIGH);
    digitalWrite(VIS_OUT_PIN, HIGH);
    digitalWrite(TRIGGER_OUT_PIN, HIGH);
    delay(SIGNAL_LENGTH);
    noTone(SOUND_OUT_PIN);
    digitalWrite(IR_OUT_PIN, LOW);
    digitalWrite(IR_OUT_2_PIN, LOW);
    digitalWrite(IR_OUT_3_PIN, LOW);
    digitalWrite(VIS_OUT_PIN, LOW);
    digitalWrite(TRIGGER_OUT_PIN, LOW);
  }
}