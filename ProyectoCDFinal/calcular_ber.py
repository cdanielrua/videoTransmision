"""
Calcula el Bit Error Rate (BER) entre el mensaje transmitido
(mensaje_enviado.txt) y el mensaje recibido (mensaje_recibido.txt).

Ejecutar: python calcular_ber.py
"""
from transmisor.encoder import text_to_bits
from receptor.decoder import calculate_ber

ENVIADO  = "mensaje_enviado.txt"
RECIBIDO = "mensaje_recibido.txt"


def main():
    with open(ENVIADO, "r", encoding="utf-8") as f:
        texto_tx = f.read()
    with open(RECIBIDO, "r", encoding="utf-8") as f:
        texto_rx = f.read()

    bits_tx = text_to_bits(texto_tx)
    bits_rx = text_to_bits(texto_rx)

    n   = min(len(bits_tx), len(bits_rx))
    ber = calculate_ber(bits_tx, bits_rx)

    n_chars     = min(len(texto_tx), len(texto_rx))
    char_errors = sum(1 for a, b in zip(texto_tx, texto_rx) if a != b)
    char_errors += abs(len(texto_tx) - len(texto_rx))

    print(f"Mensaje transmitido : {len(texto_tx)} caracteres / {len(bits_tx)} bits")
    print(f"Mensaje recibido    : {len(texto_rx)} caracteres / {len(bits_rx)} bits")
    print(f"{'='*50}")
    print(f"Bits comparados     : {n}")
    print(f"BER                 : {ber:.6f}")
    print(f"Errores de caracter : {char_errors}/{len(texto_tx)}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
