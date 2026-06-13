"""
Define la estructura del protocolo de tramas.
Cada frame de datos tiene este layout en las celdas de datos:

[SEQ_HIGH] [SEQ_LOW] [TOTAL_HIGH] [TOTAL_LOW] [CRC8] [DATA...]

- SEQ   : número de frame (0-based), 2 bytes → hasta 65535 frames
- TOTAL : total de frames en la transmisión
- CRC8  : checksum de los datos de este frame
- DATA  : payload de bits codificados en Manchester → símbolos OOK
"""
import numpy as np


def crc8(data: np.ndarray) -> int:
    """CRC-8 simple para detección de errores por frame."""
    crc = 0
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


def build_packet(seq: int, total: int, payload_bits: np.ndarray) -> np.ndarray:
    """
    Construye el array de símbolos OOK para un frame:
    Header (5 bytes en bits) + payload Manchester.
    """
    from transmisor.encoder import manchester_encode, bits_to_ook_symbols

    # Header: 5 bytes = 40 bits → 80 bits Manchester
    header_bytes = np.array([
        (seq >> 8) & 0xFF,    # SEQ alto
        seq & 0xFF,            # SEQ bajo
        (total >> 8) & 0xFF,  # TOTAL alto
        total & 0xFF,          # TOTAL bajo
        crc8(payload_bits),   # CRC del payload
    ], dtype=np.uint8)

    header_bits = np.unpackbits(header_bytes)
    header_man  = manchester_encode(header_bits)
    payload_man = manchester_encode(payload_bits)

    all_bits = np.concatenate([header_man, payload_man])
    return bits_to_ook_symbols(all_bits)


def parse_packet(symbols: np.ndarray, threshold: int = 128):
    """
    Decodifica un frame y extrae header + payload.
    Retorna (seq, total, payload_bits, crc_ok) o None si hay error.
    """
    from receptor.decoder import symbols_to_bits_ook, manchester_decode

    bits_raw = symbols_to_bits_ook(symbols, threshold)
    bits_dec = manchester_decode(bits_raw)

    # Header: 5 bytes = 40 bits
    if len(bits_dec) < 40:
        return None

    header_bits = bits_dec[:40]
    payload_bits = bits_dec[40:]

    # Reconstruir bytes del header
    if len(header_bits) < 40 or 2 in header_bits[:40]:
        return None

    header_bytes = np.packbits(header_bits)
    seq   = (int(header_bytes[0]) << 8) | int(header_bytes[1])
    total = (int(header_bytes[2]) << 8) | int(header_bytes[3])
    crc_rx = int(header_bytes[4])

    # Verificar CRC
    payload_clean = payload_bits[payload_bits <= 1]
    crc_calc = crc8(payload_clean)
    crc_ok = (crc_rx == crc_calc)

    return seq, total, payload_clean, crc_ok