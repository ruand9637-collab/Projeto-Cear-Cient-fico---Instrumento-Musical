const int pinoBuzzer = 25; 
const int DURACAO_BIP = 150;

void setup() {
  Serial.begin(9600);
  pinMode(pinoBuzzer, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char letraLibras = Serial.read();

    if (letraLibras == '\n' || letraLibras == '\r') return;

    switch (letraLibras) {
      case 'X': noTone(pinoBuzzer); break;

      
      case 'C': tone(pinoBuzzer, 262, DURACAO_BIP); break; // Dó
      case 'D': tone(pinoBuzzer, 294, DURACAO_BIP); break; // Ré
      case 'E': tone(pinoBuzzer, 330, DURACAO_BIP); break; // Mi
      case 'F': tone(pinoBuzzer, 349, DURACAO_BIP); break; // Fá
      case 'G': tone(pinoBuzzer, 392, DURACAO_BIP); break; // Sol
      case 'A': tone(pinoBuzzer, 440, DURACAO_BIP); break; // Lá
      case 'B': tone(pinoBuzzer, 494, DURACAO_BIP); break; // Si
    }
  }
}