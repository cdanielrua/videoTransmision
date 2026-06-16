"""
Lógica de recepción reutilizable: decodificación de tramas, detección de
preámbulo/fin, reconstrucción del mensaje y una máquina de estados con
reinicio manual.

Usado por main_rx.py y bidireccional.py.
"""
import time
import cv2
import numpy as np
from receptor.capture import rectify_frame, extract_symbols
from receptor.decoder import calibrate_threshold, bits_to_text
from common.protocol import parse_packet, payload_capacity
from transmisor.frame_builder import cells_capacity
from common.config import *

PAYLOAD_BITS = payload_capacity(cells_capacity())


def decode_frame(gray, corners, bgr=None):
    """Usa canal Y de YCbCr para mayor robustez ante variaciones de color."""
    if bgr is not None:
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        luma  = ycbcr[:, :, 0]
    else:
        luma = gray

    rectified = rectify_frame(luma, corners)
    if rectified is None:
        return None, None

    threshold = calibrate_threshold(rectified)
    symbols   = extract_symbols(rectified)
    return symbols, threshold


def is_preamble(symbols, threshold):
    bits = (symbols >= threshold).astype(np.uint8)
    expected = np.array([1 if i % 2 == 0 else 0
                         for i in range(len(bits))], dtype=np.uint8)
    return np.mean(bits == expected) > 0.85


def is_end_frame(symbols):
    return np.mean(np.abs(symbols.astype(int) - 128) < 40) > 0.80


def reconstruct(received: dict, total_frames, elapsed: float,
               crc_errors: int = 0):
    """Reconstruye el texto a partir de los frames recibidos y lo guarda
    en mensaje_recibido.txt. Los frames faltantes se rellenan con ceros."""
    if total_frames is None:
        print("No se recibió ningún frame.")
        return None

    print(f"\n{'='*50}")
    print(f"Frames recibidos : {len(received)}/{total_frames}")
    print(f"Frames perdidos  : {total_frames - len(received)}")
    print(f"CRC errors       : {crc_errors}")
    print(f"{'='*50}")

    all_bits = []
    for seq in range(total_frames):
        if seq in received:
            all_bits.extend(received[seq])
        else:
            print(f"  ⚠ Frame {seq} faltante — rellenando con ceros")
            all_bits.extend([0] * PAYLOAD_BITS)

    all_bits = np.array(all_bits, dtype=np.uint8)
    text     = bits_to_text(all_bits)

    print(f"\nTexto reconstruido ({len(text)} caracteres):")
    print(f"{'='*50}")
    print(text)
    print(f"{'='*50}")
    print(f"Tiempo           : {elapsed:.2f}s")

    with open("mensaje_recibido.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Guardado en: mensaje_recibido.txt")

    return {"text": text, "elapsed": elapsed, "received": len(received),
            "total": total_frames, "crc_errors": crc_errors}


class Receiver:
    """
    Máquina de estados: WAITING -> SYNCED -> RECEIVING -> DONE.

    Una vez en DONE, permanece ahí (ignora nuevos preámbulos) hasta que
    se llame a restart() — disparado por la tecla R o un click derecho.
    """
    WAITING   = "WAITING"
    SYNCED    = "SYNCED"
    RECEIVING = "RECEIVING"
    DONE      = "DONE"

    def __init__(self):
        self.restart()

    def restart(self):
        self.state        = self.WAITING
        self.received     = {}
        self.total_frames = None
        self.crc_errors   = 0
        self.start_time   = None
        self.last_rx_time = None
        self.result       = None

    def process(self, symbols, threshold) -> bool:
        """Procesa un frame decodificado. Retorna True si la recepción
        terminó en esta llamada (estado pasa a DONE)."""
        if self.state == self.WAITING:
            if is_preamble(symbols, threshold):
                print("✓ Preámbulo detectado — sincronizado")
                self.state = self.SYNCED
            return False

        if self.state == self.SYNCED:
            if not is_preamble(symbols, threshold):
                print("✓ Inicio de datos detectado")
                self.state = self.RECEIVING
                self.last_rx_time = time.time()  # arranca el reloj de timeout desde ya
            else:
                return False

        if self.state == self.RECEIVING:
            # Aceptar frame de fin solo cuando ya ha transcurrido el tiempo
            # mínimo para que el transmisor haya enviado todos los frames.
            # Esto evita que un frame desenfocado (símbolos ~128) lo dispare
            # prematuramente, pero permite terminación aunque falten frames.
            min_tx_time = (self.total_frames * FRAME_DURATION
                           if self.total_frames is not None else float('inf'))
            elapsed_rx  = (time.time() - self.start_time
                           if self.start_time is not None else 0.0)
            end_guard   = elapsed_rx >= min_tx_time
            if is_end_frame(symbols) and end_guard:
                elapsed = (time.time() - self.start_time
                          if self.start_time else 0)
                print(f"\n✓ Frame de fin detectado ({elapsed:.2f}s)")
                self._finish(elapsed)
                return True

            result = parse_packet(symbols, threshold)
            if result is not None:
                seq, total, payload, crc_ok = result

                # El CRC protege el payload, no el header. Si el header llega
                # corrupto, total puede ser 0 o un valor sin sentido. Solo
                # actualizamos total_frames si el valor es positivo Y o bien
                # no teníamos uno todavía, o el packet pasó CRC (header fiable).
                if total > 0:
                    if self.total_frames is None:
                        self.total_frames = total
                    elif crc_ok:
                        self.total_frames = total

                if self.start_time is None:
                    self.start_time   = time.time()
                    self.last_rx_time = self.start_time
                    print("  ⏱ Tiempo iniciado")

                if not crc_ok:
                    self.crc_errors += 1
                    print(f"  ✗ Frame {seq}: CRC error")
                elif seq not in self.received:
                    self.received[seq] = payload
                    self.last_rx_time = time.time()
                    print(f"  ✓ Frame {seq+1}/{self.total_frames} "
                          f"({len(payload)} bits)")

                # Terminar solo si recibimos todos los frames esperados y el
                # total conocido es válido (> 0 protege contra headers corruptos
                # que hayan puesto total_frames = 0).
                if (self.total_frames is not None and
                        self.total_frames > 0 and
                        len(self.received) >= self.total_frames):
                    elapsed = time.time() - self.start_time
                    print(f"\n✓ Todos los frames recibidos ({elapsed:.2f}s)")
                    self._finish(elapsed)
                    return True

        return False

    def check_timeout(self) -> bool:
        """Aborta la recepción si pasan TIMEOUT_SECONDS sin frames nuevos.
        Aplica en cualquier momento dentro de RECEIVING, incluso si aún no se
        ha parseado ningún frame (total_frames puede ser None)."""
        if self.state != self.RECEIVING or self.last_rx_time is None:
            return False

        if (self.total_frames is not None and
                self.total_frames - len(self.received) <= 0):
            return False

        if time.time() - self.last_rx_time <= TIMEOUT_SECONDS:
            return False

        elapsed = time.time() - (self.start_time or self.last_rx_time)
        self._finish(elapsed)
        return True

    def _finish(self, elapsed):
        self.result = reconstruct(self.received, self.total_frames,
                                  elapsed, self.crc_errors)
        self.state  = self.DONE
