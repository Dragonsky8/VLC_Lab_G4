/*
  test_rx.ino: testing the VLC receiver
  Course: CESE4110 Visible Light Communication & Sensing
*/
#include <RingBuf.h>
#include <CRC16.h>



/*
 * The VLC receiver is equipped with an OPT101 photodiode. 
 * Pin 5 of the OPT101 is connected to A0 of the Arduino Due
 * Pin 1 of the OPT101 is connected to 5V of the Arduino Due
 * Pin 8 of the OPT101 is connected to GND of the Arduino Due
 */
#define PD A1           // PD: Photodiode NOTE(broken wire )
#define loopDelay 200  // In mircroSeconds
#define threshold 230   // Define by light intensity
// Buffer used to check when the Preemple part of the frame is received
RingBuf<uint8_t, 24> preembleBuffer;
RingBuf<uint8_t, 24> preembleMessage;
unsigned long start, stop, bigStart;


CRC16 crc;



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
  // Store the preemble message in the RingBuffer, via this function
  hexToBinary(0xAAAAAA);
  while (1) {
    // Check if preemble is complete
    bigStart = micros();
    int signalVal = calculateThreshold(analogRead(PD));
    bool result = preembleBuffer.pushOverwrite(signalVal);

    // Serial.println(signalVal);
    // Serial.println(analogRead(PD));

    // for (int i=23; i>=0; i--){
    //   Serial.println(preembleMessage[i]);
    // }
    // delay(1000000);
    // If the buffer is full, check if the buffer equals the preemple message
    if (compareRingBuf(preembleBuffer, preembleMessage)) {
      start = micros();
      Serial.println("match found");
      // Wait for time to get next interval
      stop = micros();
      delayMicroseconds(loopDelay - (stop - start));
      delayMicroseconds(loopDelay);
      // Determine the payload length
      uint8_t lengthArray[16];
      for (int i = 0; i < 16; i++) {
        start = micros();
        lengthArray[i] = calculateThreshold(analogRead(PD));
        Serial.print((lengthArray[i]));
        stop = micros();
        delayMicroseconds(loopDelay - (stop - start));
      }
      // Calculate payload size
      int sizePayload = binToInt(lengthArray, 16);
      Serial.println((sizePayload));
      // Start reading the payload
      uint8_t payloadBuffer[sizePayload];
      for (int i = 0; i < sizePayload; i++) {
        start = micros();

        uint8_t signalVal = calculateThreshold(analogRead(PD));

        // TODO: Should keep track of the push computation time, and adjust loopDelay accordingly to keep consistent looptime.
        payloadBuffer[i] = signalVal;
        
        stop = micros();
        delayMicroseconds(loopDelay - (stop - start));
      }
      // Determine the crc value
      uint8_t crcArray[16];
      for (int i = 0; i < 16; i++) {
        start = micros();
        crcArray[i] = calculateThreshold(analogRead(PD));
        stop = micros();
        delayMicroseconds(loopDelay - (stop - start));
      }
      int crcReceived = binToInt(crcArray, 16);
      // Add crc of the payload part
      // Add preemble to CRC object
      addRingBufCRC(preembleBuffer);
      crc.add((uint8_t*)lengthArray, 16);
      crc.add((uint8_t*)payloadBuffer, sizePayload);
      Serial.println("show payload");
      for (int i=0; i< sizePayload; i++) {
        Serial.print((payloadBuffer[i]));
      }
      uint16_t crcCalc = crc.calc();
      // Check comparison
      if (crcReceived != crcCalc) {
        Serial.println(" Invalid CRC");
        // Serial.println((crcReceived));
        // Serial.println((crcCalc));
        delay(5000);
      }

   
      int* payload_decoded = manchester_decode(payloadBuffer);                //Manchester decoding
      String binaryString = intArrayToBinaryString(payload_decoded, sizePayload / 2);  //convert the array to string
      String information = binaryStringToString(binaryString);
      Serial.println("The information in this frame is: " + information);
      Serial.println("Done with payload");
      delete payload_decoded;
      crc.reset();
    } else {
      stop = micros();
      uint32_t looptime = stop - bigStart;
      uint32_t remainder = ((loopDelay)-looptime);
      if (remainder < 0) {
        remainder = loopDelay;
      }
      // Serial.println((remainder));
      delayMicroseconds((remainder));  // two times per second
    }
  }
}

uint8_t calculateThreshold(int input) {
  if (input < threshold) {
    return 0b0;
  } else {
    return 0b1;
  }
}

bool hexToBinary(int hexValue) {
  while (hexValue > 0) {
    uint8_t remainder = hexValue % 2;
    bool status = preembleMessage.push(remainder);
    hexValue /= 2;
    if (!status) {
      return false;
    }
  }
  return true;
}

int manchesterDecode(int A, int B) {
  if (A == 1 && B == 0) {
    return 0;
  } else if (A == 0 && B == 1) {
    return 1;
  }
  return 0;
}

bool compareRingBuf(RingBuf<uint8_t, 24> A, RingBuf<uint8_t, 24> B) {
  for (int i = A.size() - 1; i >= 0; i--) {
    if (A[i] != B[i]) {
      return false;
    }
    // Serial.print(A[i]);
  }
  return true;
}

void addRingBufCRC(RingBuf<uint8_t, 24> A) {
  int temp[24];
  for (int i = 23; i >= 0; i--) {
    temp[23 - i] = A[i];
  }
  Serial.println("preamble");
  for (int i = 0; i < 24; i++) {
    Serial.print((temp[i]));
  }
  crc.add((uint8_t*)temp, 24);
}

// function definition
int convert(long long n) {

  int dec = 0, i = 0, rem;

  while (n != 0) {

    // get remainder of n divided by 10
    rem = n % 10;

    // divide n by 10
    n /= 10;

    // multiply rem by (2 ^ i)
    // add the product to dec
    dec += rem * pow(2, i);

    // increment i
    ++i;
  }

  return dec;
}

// Calculate the length of the packet, based on the binary sequence
int binToInt(uint8_t array[], uint8_t len) {
  int output = 0;
  int power = 1;

  for (int i = 0; i < len; i++) {
    output += array[(len - 1) - i] * power;
    // output goes 1*2^0 + 0*2^1 + 0*2^2 + ...
    power *= 2;
  }


  return output;
}

//convert the binary sequence to words
// String binaryToString(int* binaryData, int length) {

//   String originalString = "";
//   for (int i = 0; i < length; i += 8) {
//     int byteValue = 0;
//     for (int j = 0; j < 8; j++) {
//       byteValue |= (binaryData[i + j] << (7 - j));
//     }
//     originalString += char(byteValue);
//   }
//   return originalString;
// }

String intArrayToBinaryString(int* intArray, int length) {
  String binaryString = "";
  for (int i = 0; i < length; i++) {
    binaryString += String(intArray[i]);
  }
  return binaryString;
}

String binaryStringToString(String binaryString) {
  String originalString = "";
  for (int i = 0; i < binaryString.length(); i += 8) {
    String byteString = binaryString.substring(i, i + 8);
    char charValue = char(strtol(byteString.c_str(), NULL, 2));
    originalString += charValue;
  }
  return originalString;
}

//Calculate the decoded binary sequence
int* manchester_decode(uint8_t payload[]) {
  int len = sizeof(payload) / sizeof(payload[0]);
  int* payload_decoded = new int[len / 2];
  int i = 0, j = 0;
  while (i < len - 1) {
    if (payload[i] == 0b0 && payload[i + 1] == 0b1) {
      payload_decoded[j] = 1;
    }
    if (payload[i] == 0b1 && payload[i + 1] == 0b0) {
      payload_decoded[j] = 0;
    }
    i += 2;
    j++;
  }
  return payload_decoded;
}
