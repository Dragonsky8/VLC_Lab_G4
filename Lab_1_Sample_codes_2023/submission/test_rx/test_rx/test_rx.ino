/*
  test_rx.ino: testing the VLC receiver
  Course: CESE4110 Visible Light Communication & Sensing
*/
#include <RingBuf.h>
#include "CRC16.h"



/*
 * The VLC receiver is equipped with an OPT101 photodiode. 
 * Pin 5 of the OPT101 is connected to A0 of the Arduino Due
 * Pin 1 of the OPT101 is connected to 5V of the Arduino Due
 * Pin 8 of the OPT101 is connected to GND of the Arduino Due
 */
#define PD A0          // PD: Photodiode
#define loopDelay 50   // In miliseconds
#define threshold 500  // Define by light intensity
// Buffer used to check when the Preemple part of the frame is received
RingBuf<int, 24> preembleBuffer;
RingBuf<int, 24> preembleMessage;

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
    int signalVal = calculateThreshold(analogRead(PD));
    bool result = preembleBuffer.pushOverwrite(signalVal);

    Serial.println(signalVal);
    // for (int i=23; i>=0; i--){
    //   Serial.println(preembleMessage[i]);
    // }
    // delay(1000000);
    // If the buffer is full, check if the buffer equals the preemple message
    if (compareRingBuf(preembleBuffer, preembleMessage)) {
      // Add preemble to CRC object
      addRingBufCRC(preembleBuffer);
      Serial.println("match found");

      // Determine the payload length
      int lengthArray[16];
      delay(loopDelay);
      delay(loopDelay);

      for (int i = 0; i < 16; i++) {
        lengthArray[i] = calculateThreshold(analogRead(PD));
        Serial.println(lengthArray[i]);
        delay(loopDelay);
      }
      crc.add((uint8_t *)lengthArray, sizeof(lengthArray) / sizeof(lengthArray[0]));
      // Calculate payload size
      int sizePayload = binToInt(lengthArray, 16);
      Serial.println((sizePayload));
      // Start reading the payload
      uint8_t payloadBuffer[sizePayload];
      for (int i = 0; i < sizePayload; i++) {
        int signalVal = calculateThreshold(analogRead(PD));

        // TODO: Should keep track of the push computation time, and adjust loopDelay accordingly to keep consistent looptime.
        payloadBuffer[i] = signalVal;
        delay(loopDelay);
      }
      Serial.println("Done with payload");
      // Add crc of the payload part
      crc.add(payloadBuffer, sizePayload);


      // Determine the crc value
      int crcArray[16];
      for (int i = 0; i < 16; i++) {
        crcArray[i] = calculateThreshold(analogRead(PD));
        delay(loopDelay);
      }
      int crcReceived = binToInt(crcArray, 16);
      uint16_t crcCalc = crc.calc();
      if (crcReceived != crcCalc) {
        Serial.println("Invalid CRC");
      }
      for (int i = 0; i < 16; i++) {
        Serial.println(payloadBuffer[i]);
      }
    }
    delay(loopDelay);  // two times per second
  }
}

int calculateThreshold(int input) {
  if (input < threshold) {
    return 0b0;
  } else {
    return 0b1;
  }
}

bool hexToBinary(int hexValue) {
  while (hexValue > 0) {
    int remainder = hexValue % 2;
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

bool compareRingBuf(RingBuf<int, 24> A, RingBuf<int, 24> B) {
  for (int i = A.size() - 1; i >= 0; i--) {
    if (A[i] != B[i]) {
      return false;
    }
  }
  return true;
}

void addRingBufCRC(RingBuf<int, 24> A) {
  for (int i = A.size() - 1; i >= 0; i--) {
    crc.add(A[i]);
  }
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
int binToInt(int array[], int len) {
  int output = 0;
  int power = 1;

  for (int i = 0; i < len; i++) {
    output += array[(len - 1) - i] * power;
    // output goes 1*2^0 + 0*2^1 + 0*2^2 + ...
    power *= 2;
  }


  return output;
}
