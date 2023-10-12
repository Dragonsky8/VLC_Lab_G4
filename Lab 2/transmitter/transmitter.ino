/*
  test_tx.ino: testing the VLC transmitter
  Course: CESE4110 Visible Light Communication & Sensing
*/


/*
 * The VLC transmitter is equipped with an RGB LED. 
 * The LED's three channels, R, G, B, can be controlled individually.
 * The R channel is connected to Pin 38 of the Arduino Due
 * The G channel is connected to Pin 42 of the Arduino Due
 * The B channel is connected to Pin 34 of the Arduino Due
 */
const int ledR= 38; // GPIO for controlling R channel
const int ledG= 42; // GPIO for controlling G channel
const int ledB= 34; // GPIO for controlling B channel


/*
 * Brightness of each channel.
 * The range of the brightness is [0, 255].
 * *  0 represents the highest brightness
 * *  255 represents the lowest brightness
 */
int britnessR = 255; // Default: lowest brightness
int britnessG = 255; // Default: lowest brightness
int britnessB = 255; // Default: lowest brightness

uint32_t start, end, elapsedTime; //to record the time during

int counter = 1;


/*
 * Some configurations
 */
void setup() {
  Serial.begin(115200); // Set the Baud rate to 115200 bits/s
  while (Serial.available() > 0)
    Serial.read();

  pinMode(ledR, OUTPUT);
  pinMode(ledG, OUTPUT);
  pinMode(ledB, OUTPUT);

  analogWrite(ledR, britnessR); // Turn OFF the R channel
  analogWrite(ledG, britnessG); // Turn OFF the G channel
  analogWrite(ledB, britnessB); // Turn OFF the B channel

  start = micros();
}

/*
 * The Main function
 */
void loop() {
  /*
   * In this simple test, only the R channel is used to transmit data.
   * The R channel is turned ON and OFF alternatively to transmit 1 and 0. 
   * The TX frequency is set to 10 Hz, i.e., sending a symbol every 100 ms
   */
    end = micros();
    elapsedTime = end - start;

    if (elapsedTime >= 0 && elapsedTime < 1000000){
      /*
      * Pattern 1
      */
      analogWrite(ledR, britnessR);
      britnessR = (britnessR == 0 ? 255 : 0);
    
      delayMicroseconds(100000); // TX frequency:  1s/100ms = 10 Hz
    }else if (elapsedTime >= 1000000 && elapsedTime < 2000000){
      /*
      * Pattern 2, different flickering
      */  
      analogWrite(ledR, britnessR);
      if (counter % 2 == 1 && britnessR == 0){
        britnessR = 0;
        counter++;
      }else if (counter % 2 == 1 && britnessR == 255){
        britnessR = 255;
        counter++;
      }else if (counter % 2 == 0 && britnessR == 0){
        britnessR = 255;
        counter = 1;
      }else if (counter % 2 == 0 && britnessR == 255){
        britnessR = 0;
        counter = 1;
      }
   
      
      delayMicroseconds(1000000); // TX frequency:  1s/100ms = 10 Hz

    }else if (elapsedTime >= 2000000 && elapsedTime < 3000000){
      /*
      * Pattern 3, diifferent frequency
      */
      analogWrite(ledR, britnessR);
      britnessR = (britnessR == 0 ? 255 : 0);
    
      delayMicroseconds(33333); // TX frequency:  1s/33ms = 30 Hz
    }
  
}
