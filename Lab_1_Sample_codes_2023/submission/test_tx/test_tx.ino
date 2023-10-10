
#include <CRC16.h>

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
const int ledR = 38;  // GPIO for controlling R channel, which should be the transmitter pin
const int ledG = 42;  // GPIO for controlling G channel
const int ledB = 34;  // GPIO for controlling B channel



/*
 * Brightness of each channel.
 * The range of the brightness is [0, 255].
 * *  0 represents the highest brightness
 * *  255 represents the lowest brightness
 */
int britnessR = 255;  // Default: lowest brightness
int britnessG = 255;  // Default: lowest brightness
int britnessB = 255;  // Default: lowest brightness

unsigned long startTime;  // 用于记录开始时间
uint32_t start, end;

/*
 * Some configurations
 */
void setup() {

  Serial.begin(115200);  // Set the Baud rate to 115200 bits/s
  while (Serial.available() > 0)
    Serial.read();

  // pinMode(ledR, OUTPUT);
  // pinMode(ledG, OUTPUT);
  // pinMode(ledB, OUTPUT);

  analogWrite(ledR, britnessR);  // Turn OFF the R channel
  analogWrite(ledG, britnessG);  // Turn OFF the G channel
  analogWrite(ledB, britnessB);  // Turn OFF the B channel
  startTime = micros();
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


  String dataToEncode = "Hello World";
  // String dataToEncode = "Hello";

  char preamble_arr[24];
  char length_arr[16];
  char crc_arr[16];


  //preamble part here
  int preamble_hex = 0xAAAAAA;
  String preamble_binary = hexToBinaryString(preamble_hex);
  for (int i = 0; i < preamble_binary.length(); i++) {
    char bit = preamble_binary.charAt(i);
    preamble_arr[i] = bit;
  }
  if (preamble_binary.length() < 24) {
    for (int i = preamble_binary.length(); i < 24; i++) {
      preamble_arr[i] = '0';
    }
  }

  // Serial.println("The preamble is:" + preamble_binary);
  //convert the data to binary sequence
  String binaryData = stringToBinary(dataToEncode);
  // Serial.println("The original BINARY sequence is: " + binaryData);
  // Serial.println(binaryData.length());


  //payload part here
  String payload = "";
  // Manchester coding
  for (int i = 0; i < binaryData.length(); i++) {
    char bit = binaryData.charAt(i);
    if (bit == '0') {
      payload += '1';
      payload += '0';
    } else {
      payload += '0';
      payload += '1';
    }
  }

  //length part here
  String binary_length = intToBinary(payload.length());
  for (int i = 0; i < binary_length.length(); i++) {
    char bit = binary_length.charAt(i);
    length_arr[i] = bit;
  }

  int payload_length = payload.length();
  char payload_arr[payload_length];

  // Serial.println("The payload length(binary) is:" + binary_length);
  // Serial.println("The payload is:"+payload);
  // Serial.println(payload_length);

  //convert the payload from stirng to array
  for (int i = 0; i < payload.length(); i++) {
    char bit = payload.charAt(i);
    payload_arr[i] = bit;
  }
  char merged[40 + payload_length];  //preamble+length+payload, used for calculating crc
  memcpy(merged, preamble_arr, sizeof(preamble_arr) / sizeof(preamble_arr[0]));
  memcpy(merged + 24, length_arr, sizeof(length_arr) / sizeof(length_arr[0]));
  memcpy(merged + 40, payload_arr, sizeof(payload_arr) / sizeof(payload_arr[0]));
  // for (int i = 0; i < sizeof(merged) / sizeof(merged[0]); i++) {
  //   if (i != sizeof(merged) / sizeof(merged[0]) - 1) {
  //     // Serial.print(merged[i]);
  //   } else {
  //     // Serial.println(merged[i]);
  //   }
  // }

  //below are the program for CRC calculating
  CRC16 crc;
  uint8_t pre_arr[24];
  uint8_t len_arr[16];
  uint8_t pay_arr[payload_length];
  for (int i = 0; i < 24; i++){
    if (preamble_arr[i] == '1'){
      pre_arr[i] = 0b1;
    }else{
      pre_arr[i] = 0b0;
    }
  }
  for (int i = 0; i < 16; i++){
    if (length_arr[i] == '1'){
      len_arr[i] = 0b1;
    }else{
      len_arr[i] = 0b0;
    }
  }
  for (int i = 0; i < payload_length; i++){
    if (payload_arr[i] == '1'){
      pay_arr[i] = 0b1;
    }else{
      pay_arr[i] = 0b0;
    }
  }

  crc.add((uint8_t *)pre_arr, 24);
  crc.add((uint8_t *)len_arr, 16);
  crc.add((uint8_t *)pay_arr, payload_length);
  // crc.add((uint8_t *) merged, 40+payload_length);
  uint16_t crcValue = crc.calc();
  Serial.println("CRC: ");
  Serial.print((crcValue));

  String crc_str = uint16ToBinary(crcValue);
  // Serial.println(crc_str);
  // Serial.println((crcValue));
  for (int i = 0; i < crc_str.length(); i++) {
    char bit = crc_str.charAt(i);
    crc_arr[i] = bit;
  }


  //merge all parts together to one frame (also char[])
  char frame[56 + payload_length];
  for (int i = 0; i < 40 + payload_length; i++) {
    frame[i] = merged[i];
  }
  for (int i = 40 + payload_length; i < 56 + payload_length; i++) {
    frame[i] = crc_arr[i - (40 + payload_length)];
  }

  for (int i = 0; i < crc_str.length(); i++) {
    char bit = crc_str.charAt(i);
    crc_arr[i] = bit;
  }

  for (int i = 0; i < sizeof(crc_arr) / sizeof(crc_arr[0]); i++) {
    Serial.print(crc_arr[i]);
  }

  //send the frame based on red channel
  for (int i = 0; i < sizeof(frame) / sizeof(frame[0]); i++) {
    start = micros();
    //send 0 or 1
    if (frame[i] == '0') {
      digitalWrite(ledR, 255);  // on-off

    } else {
      digitalWrite(ledR, 0);  // off-on
    }
    end = micros();
    uint32_t elapsedTime = end - start;  

    uint32_t remainder = (100000) - elapsedTime; //10Hz for now, since the capture rate of camera is 30fps, 1/3 of camera
    // Serial.println(remainder);

    delayMicroseconds(remainder);
  }
  Serial.println("done with sending payload");
  // delay for a while
  delay(1000);
}

String stringToBinary(String input) {
  String binaryString = "";
  for (char c : input) {
    if ((c>='a'&&c<='z') || (c>='A'&&c<='Z')){
      binaryString += "0";
      binaryString += String(c, BIN);
    }else{
      binaryString += "00";
      binaryString += String(c, BIN);
    }
    
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

String intToBinary(int decimalValue) {
  String binaryString = "";

  for (int i = 15; i >= 0; i--) {
    int bit = (decimalValue >> i) & 1;
    binaryString += String(bit);
  }

  return binaryString;
}

String uint16ToBinary(uint16_t value) {
  String binaryString = "";

  for (int i = 15; i >= 0; i--) {
    int bit = (value >> i) & 1;
    binaryString += String(bit);
  }

  return binaryString;
}
