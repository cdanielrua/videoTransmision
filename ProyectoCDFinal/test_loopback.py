# test_loopback.py — prueba sin cámara
import cv2
import numpy as np
from transmisor.encoder import text_to_bits, manchester_encode, bits_to_ook_symbols
from transmisor.frame_builder import build_frame
from receptor.capture import find_fiducials, rectify_frame, extract_symbols
from receptor.decoder import calibrate_threshold, symbols_to_bits_ook, \
                              manchester_decode, bits_to_text, calculate_ber

TEXTO = "Hola Universidad de Antioquia"

# --- Transmitir ---
bits_tx = manchester_encode(text_to_bits(TEXTO))
symbols_tx = bits_to_ook_symbols(bits_tx)
frame = build_frame(symbols_tx, 1, 1)
gray_ideal = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# --- Simular canal ideal (sin distorsión) ---
corners_ideal = [
    (50, 50), (750, 50), (50, 550), (750, 550)
]

# --- Recibir ---
threshold = calibrate_threshold(gray_ideal)
symbols_rx = extract_symbols(gray_ideal)
bits_rx = symbols_to_bits_ook(symbols_rx, threshold)
bits_dec = manchester_decode(bits_rx)
texto_rx = bits_to_text(bits_dec)

# --- Métricas ---
n = min(len(bits_tx), len(bits_rx))
ber = calculate_ber(bits_tx[:n], bits_rx[:n])

print(f"TX: '{TEXTO}'")
print(f"RX: '{texto_rx}'")
print(f"BER: {ber:.6f}")
print(f"{'✓ PERFECTO' if ber == 0 else '✗ HAY ERRORES'}")