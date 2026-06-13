import numpy as np

def text_to_bits(text: str) -> np.ndarray:
    """Convierte texto UTF-8 a array de bits."""
    bits = []
    for char in text:
        byte = ord(char)
        for i in range(7, -1, -1):  # MSB primero
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=np.uint8)

def manchester_encode(bits: np.ndarray) -> np.ndarray:
    """
    Codificación Manchester:
    bit 0 → [1, 0]
    bit 1 → [0, 1]
    Garantiza balance de brillo y facilita sincronización.
    """
    encoded = []
    for bit in bits:
        if bit == 0:
            encoded.extend([1, 0])
        else:
            encoded.extend([0, 1])
    return np.array(encoded, dtype=np.uint8)

def bits_to_ook_symbols(bits: np.ndarray) -> np.ndarray:
    """OOK: 0 → negro (0), 1 → blanco (255)"""
    return (bits * 255).astype(np.uint8)

def bits_to_4ask_symbols(bits: np.ndarray) -> np.ndarray:
    """
    4-ASK: agrupa bits de 2 en 2 → 4 niveles de gris
    00 → 0, 01 → 85, 10 → 170, 11 → 255
    Duplica la tasa de bits por celda.
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)  # padding
    symbols = []
    levels = [0, 85, 170, 255]
    for i in range(0, len(bits), 2):
        idx = (bits[i] << 1) | bits[i+1]
        symbols.append(levels[idx])
    return np.array(symbols, dtype=np.uint8)

# --- Test rápido ---
if __name__ == '__main__':
    texto = "Hola"
    bits = text_to_bits(texto)
    print(f"Texto: '{texto}' → {len(bits)} bits")
    print(f"Bits: {bits}")
    
    man = manchester_encode(bits)
    print(f"Manchester: {len(man)} bits (×2)")
    
    symbols = bits_to_ook_symbols(man)
    print(f"Símbolos OOK: {symbols[:16]}...")