#include <vector>
using namespace std;
const int ldrPin_1 = A0;
const int ldrPin_2 = A2;
const int ldrPin_3 = A4;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); // Set the Baud rate to 115200 bits/s
}



void loop() {
  // put your main code here, to run repeatedly:
  int ldrValue_1 = analogRead(ldrPin_1); // read the value of LDR
  int ldrValue_2 = analogRead(ldrPin_2);
  int ldrValue_3 = analogRead(ldrPin_3);

  Serial.print(ldrPin_1);
  Serial.print(" ");
  Serial.print(ldrPin_2);
  Serial.print(" ");
  Serial.println(ldrPin_3);
  // cell_1.push_back(ldrValue_1);
  // for (int i = 0; i < cell_1.size(); i++){
  //   Serial.println(cell_1[i]);
  //   delay(100);
  // }
  delay(100);
}
