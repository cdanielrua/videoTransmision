import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import *


def calibrate_threshold(rectified_gray: np.ndarray) -> int:
    """
    Usa la fila de pilotos (patrón conocido 255/0 alternado)
    para calcular el umbral óptimo de decisión.
    """
    from common.config import PILOT_ROW, CELL_SIZE, GRID_COLS
    margin = CELL_SIZE // 4
    whites, blacks = [], []

    for col in range(GRID_COLS):
        y = PILOT_ROW * CELL_SIZE + margin
        x = col * CELL_SIZE + margin
        h = w = CELL_SIZE - 2 * margin
        roi = rectified_gray[y:y+h, x:x+w]
        val = int(np.median(roi))
        if col % 2 == 0:
            whites.append(val)
        else:
            blacks.append(val)

    mean_white = np.mean(whites)
    mean_black = np.mean(blacks)
    threshold = int((mean_white + mean_black) / 2)
    print(f"[Calibración] Blanco={mean_white:.1f} Negro={mean_black:.1f} Umbral={threshold}")
    return threshold


def symbols_to_bits_ook(symbols: np.ndarray, threshold: int = 128) -> np.ndarray:
    """OOK: símbolo >= umbral → 1, < umbral → 0"""
    return (symbols >= threshold).astype(np.uint8)


def manchester_decode(bits: np.ndarray) -> np.ndarray:
    """
    Decodifica Manchester: cada par [1,0] → 0, [0,1] → 1
    Pares inválidos se marcan como error (valor 2).
    """
    decoded = []
    for i in range(0, len(bits) - 1, 2):
        pair = (bits[i], bits[i+1])
        if pair == (1, 0):
            decoded.append(0)
        elif pair == (0, 1):
            decoded.append(1)
        else:
            decoded.append(2)  # error de decodificación
    return np.array(decoded, dtype=np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    """Convierte array de bits a texto UTF-8."""
    text = []
    # Ignorar bits con error (valor 2)
    clean_bits = bits[bits <= 1]

    for i in range(0, len(clean_bits) - 7, 8):
        byte_bits = clean_bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_val = 0
        for b in byte_bits:
            byte_val = (byte_val << 1) | int(b)
        if byte_val == 0:  # byte nulo = terminador de mensaje
            break
        if 32 <= byte_val <= 126:  # solo ASCII imprimible
            text.append(chr(byte_val))
        else:
            text.append('?')
    return ''.join(text)


def calculate_ber(original_bits: np.ndarray, received_bits: np.ndarray) -> float:
    """Calcula Bit Error Rate entre transmitido y recibido."""
    n = min(len(original_bits), len(received_bits))
    errors = np.sum(original_bits[:n] != received_bits[:n])
    return errors / n if n > 0 else 1.0