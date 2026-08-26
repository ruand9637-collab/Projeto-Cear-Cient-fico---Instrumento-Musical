const int pinoBuzzer = 25; 

void setup() {
  Serial.begin(9600);
  pinMode(pinoBuzzer, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char letraLibras = Serial.read();

    // Remove resíduos de quebra de linha (\n ou \r) se houver
    if (letraLibras == '\n' || letraLibras == '\r') {
      return; 
    }

    switch (letraLibras) {
      case 'X': // Silêncio
        noTone(pinoBuzzer);
        break;
        
      case 'C': // Có -> Nota Dó
        tone(pinoBuzzer, 264); 
        break;
        
      case 'D': // Dó -> Nota Ré
        tone(pinoBuzzer, 300); 
        break;
        
      case 'E': // É -> Nota Mi
        tone(pinoBuzzer, 330); 
        break;
        
      case 'F': // Fá -> Nota Fá
        tone(pinoBuzzer, 352); 
        break;
        
      case 'G': // Gá -> Nota Sol
        tone(pinoBuzzer, 396); 
        break;
        
      case 'A': // Á -> Nota Lá
        tone(pinoBuzzer, 440); 
        break;
        
      case 'B': // Bí -> Nota Si
        tone(pinoBuzzer, 495); 
        break;
    }
  }
}