#include <vector>
using namespace std;
const int ldrPin_1 = A7;
const int ldrPin_2 = A2;
const int ldrPin_3 = A4;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); // Set the Baud rate to 115200 bits/s
  pinMode(A7, INPUT);
  pinMode(A2, INPUT);
  pinMode(A4, INPUT);
}



void loop() {
  // put your main code here, to run repeatedly:
  int ldrValue_1 = analogRead(A7); // read the value of LDR

  int ldrValue_2 = analogRead(A2);
 
  int ldrValue_3 = analogRead(A4);
    

  Serial.print(ldrValue_1);
  Serial.print(" ");
  Serial.print(ldrValue_2);
  Serial.print(" ");
  Serial.println(ldrValue_3);
  // cell_1.push_back(ldrValue_1);
  // for (int i = 0; i < cell_1.size(); i++){
  //   Serial.println(cell_1[i]);
  //   delay(100);
  // }
  delay(50); //frequency 500Hz
}
