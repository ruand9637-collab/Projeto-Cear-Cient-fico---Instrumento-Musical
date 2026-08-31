#include <driver/i2s.h>
#include <math.h>

const i2s_port_t i2s_num = I2S_NUM_0;
const int DURACAO_BIP = 150; 

unsigned long ultimo_comando = 0;
const unsigned long TIMEOUT_MS = 2000; // 2 segundos sem comando = para tudo

// Se o módulo de vibração estiver em um pino digital separado (ex: GPIO 13)
//const int PINO_VIBRACAO = 32; 

const int PINO_SD = 14;

void setup() {
  Serial.begin(9600);

  pinMode(PINO_SD, OUTPUT);
  digitalWrite(PINO_SD, HIGH);

  // Configuração I2S (Mantenha a sua configuração atual aqui)
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = 44100,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false
  };

  // Configuração e indicação dos pinos(Definição do SD no início do código)
  i2s_pin_config_t pin_config = {
    .bck_io_num = 26, /*BCLK*/
    .ws_io_num = 27, /*LRC*/
    .data_out_num = 25, /*DIN*/
    
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(i2s_num, &i2s_config, 0, NULL);
  i2s_set_pin(i2s_num, &pin_config);
  
  
 /* digitalWrite(PINO_VIBRACAO, LOW);
  pinMode(PINO_VIBRACAO, OUTPUT);
  analogWrite(PINO_VIBRACAO, 0); 
}*/

void pararAudioI2S() {
  // Envia amostras ZERADAS para forçar o silêncio e desligar o amplificador
  int16_t buffer_zero[2] = {0, 0};
  size_t bytes_escritos;
  for (int i = 0; i < 100; i++) {
    i2s_write(i2s_num, &buffer_zero, sizeof(buffer_zero), &bytes_escritos, portMAX_DELAY);
  }
  /*analogWrite(PINO_VIBRACAO, 0); // Desliga motor externo no silêncio*/
}

void tocarNotaI2S(float frequencia, int duracao_ms, int intensidade_vibracao) {
  int sample_rate = 44100;
  int total_amostras = (sample_rate * duracao_ms) / 1000;
  size_t bytes_escritos;
/*
  // Aciona motor PWM externo (se houver)
  analogWrite(PINO_VIBRACAO, intensidade_vibracao);
*/
  for (int i = 0; i < total_amostras; i++) {
    int16_t sample = (int16_t)(sin(i * 2.0 * M_PI * frequencia / sample_rate) * 8000);
    int16_t buffer[2] = {sample, sample};
    i2s_write(i2s_num, &buffer, sizeof(buffer), &bytes_escritos, portMAX_DELAY);
  }

  // Corta a vibração e zera o I2S ao terminar o tempo do bip
  pararAudioI2S();
}

void loop() {
  if (millis() - ultimo_comando > TIMEOUT_MS) {
    pararAudioI2S();
  }

  if (Serial.available() > 0) {
    char letraLibras = Serial.read();

    if (letraLibras == '\n' || letraLibras == '\r') return;

    switch (letraLibras) {
      case 'X': 
        pararAudioI2S();
        ultimo_comando = millis();
        break;

      // tocarNotaI2S(Frequencia_Som, Duracao_ms, Intensidade_Vibracao_0_a_255)
      case 'C': tocarNotaI2S(262.0, DURACAO_BIP, 255); break; // Dó: Vibração Forte (Grave)
      case 'D': tocarNotaI2S(294.0, DURACAO_BIP, 220); break; // Ré
      case 'E': tocarNotaI2S(330.0, DURACAO_BIP, 180); break; // Mi
      case 'F': tocarNotaI2S(349.0, DURACAO_BIP, 150); break; // Fá
      case 'G': tocarNotaI2S(392.0, DURACAO_BIP, 120); break; // Sol
      case 'A': tocarNotaI2S(440.0, DURACAO_BIP, 90);  break; // Lá
      case 'B': tocarNotaI2S(494.0, DURACAO_BIP, 60);  break; // Si: Vibração Leve (Agudo)
    }
  }
}