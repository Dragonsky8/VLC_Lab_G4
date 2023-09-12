/*
  test_tx.ino: testing the VLC transmitter
  Course: CESE4110 Visible Light Communication & Sensing
*/


// 设置曼彻斯特编码的时间间隔（微秒）
#define MANCHESTER_INTERVAL 500



/*
 * The VLC transmitter is equipped with an RGB LED. 
 * The LED's three channels, R, G, B, can be controlled individually.
 * The R channel is connected to Pin 38 of the Arduino Due
 * The G channel is connected to Pin 42 of the Arduino Due
 * The B channel is connected to Pin 34 of the Arduino Due
 */
const int ledR= 38; // GPIO for controlling R channel, which should be the transmitter pin
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



/*
 * Some configurations
 */
void setup() {
  
  Serial.begin(115200); // Set the Baud rate to 115200 bits/s
  while (Serial.available() > 0)
    Serial.read();

  // pinMode(ledR, OUTPUT);
  // pinMode(ledG, OUTPUT);
  // pinMode(ledB, OUTPUT);

  analogWrite(ledR, britnessR); // Turn OFF the R channel
  analogWrite(ledG, britnessG); // Turn OFF the G channel
  analogWrite(ledB, britnessB); // Turn OFF the B channel
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
  // analogWrite(ledR, britnessR);
  // britnessR = (britnessR == 0 ? 255 : 0);
  
  // delay(500); // TX frequency:  1s/400ms = 2.5 Hz

  String dataToEncode = "Hello VLC&S 2023-2024";

  String binaryData = stringToBinary(dataToEncode);
  Serial.println(binaryData);
  Serial.println(binaryData.length());

  int preamble_hex = 0xAAAAAA;
  String preamble_binary = hexToBinaryString(preamble_hex);

  String payload = "";
  // Manchester coding
  for (int i = 0; i < binaryData.length(); i++) {
    char bit = binaryData.charAt(i);
    if (bit == '0'){
      payload += '1';
      payload += '0';
    }else{
      payload += '0';
      payload += '1';
    }
  }

  int payload_length = payload.length();
  Serial.println(payload);
  Serial.println(payload_length);

  //send the frame based on red channel
  for (int i = 0; i < payload.length(); i++){
    char bit = binaryData.charAt(i);
    //send 0 or 1
    if (bit == '0') {
      digitalWrite(ledR, 255); // on-off
      
    } else {
      digitalWrite(ledR, 0); // off-on
    
    }
    delay(500);
  }
  Serial.println("done with payload");
  // delay for a while
  delay(1000000000000000000);
}

String stringToBinary(String input) {
  String binaryString = "";
  for (char c : input) {
    binaryString += String(c, BIN);
  }
  return binaryString;
}

String hexToBinaryString(int hexValue) {
  String binaryString = "";

  while (hexValue > 0) {
    int remainder = hexValue % 2;
    binaryString = String(remainder) + binaryString;
    hexValue /= 2;
  }

  return binaryString;
}
