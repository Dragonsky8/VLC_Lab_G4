/*
  test_rx.ino: testing the VLC receiver
  Course: CESE4110 Visible Light Communication & Sensing
*/


/*
 * The VLC receiver is equipped with an OPT101 photodiode. 
 * Pin 5 of the OPT101 is connected to A0 of the Arduino Due
 * Pin 1 of the OPT101 is connected to 5V of the Arduino Due
 * Pin 8 of the OPT101 is connected to GND of the Arduino Due
 */
#define PD A0 // PD: Photodiode


/*
 * Some configurations
 */
void setup() {
  Serial.begin(115200);
}


/*
 * The Main function
 */
void loop() {
  while (1)
  {
    Serial.println(analogRead(PD));
    delay(200); // two times per second
  }
}
