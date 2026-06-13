"""
Receptor multi-frame con timeout y YCbCr.
Ejecutar: python main_rx.py
"""
import cv2
import numpy as np
import time
from receptor.capture import find_fiducials_robust, rectify_frame, extract_symbols
from receptor.decoder import calibrate_threshold, symbols_to_bits_ook, \
                             manchester_decode, bits_to_text
from common.protocol import parse_packet, payload_capacity
from transmisor.frame_builder import cells_capacity
from common.config import *

# Timeout: si no llega un frame nuevo en X segundos, terminar igual
TIMEOUT_SECONDS = 3.0

# Bits de payload por frame (igual calculo que usa el transmisor)
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


def is_end_frame(symbols, threshold):
    return np.mean(np.abs(symbols.astype(int) - 128) < 40) > 0.80


def receive():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    print("Receptor multi-frame iniciado.")
    print("Esperando preámbulo del transmisor...")
    print("[D] debug  [R] reiniciar  [Q] salir\n")

    state        = "WAITING"
    received     = {}
    total_frames = None
    crc_errors   = 0
    start_time   = None
    last_rx_time = None   # último momento en que llegó un frame válido
    debug_mode   = False

    def reset_state():
        nonlocal state, received, total_frames, crc_errors
        nonlocal start_time, last_rx_time
        state        = "WAITING"
        received     = {}
        total_frames = None
        crc_errors   = 0
        start_time   = None
        last_rx_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, method = find_fiducials_robust(gray, bgr=frame,
                                                debug=debug_mode)
        display = frame.copy()

        # ── HUD ──────────────────────────────────────────────
        status_color = {
            "WAITING":   (0, 0,   255),
            "SYNCED":    (0, 165, 255),
            "RECEIVING": (0, 255,   0),
        }.get(state, (255, 255, 255))

        cv2.putText(display, f"Estado: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if corners is not None:
            pts = np.array([corners[0], corners[1],
                            corners[3], corners[2]], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, f"[{method}]", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if total_frames is not None:
            cv2.putText(display,
                        f"Frames: {len(received)}/{total_frames}  "
                        f"CRC errors: {crc_errors}",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Mostrar countdown del timeout
        if state == "RECEIVING" and last_rx_time is not None:
            elapsed_since = time.time() - last_rx_time
            remaining     = TIMEOUT_SECONDS - elapsed_since
            if remaining > 0:
                cv2.putText(display,
                            f"Timeout en: {remaining:.1f}s",
                            (10, display.shape[0] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 165, 255), 1)

        cv2.imshow("Receptor", display)

        # ── Teclas ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('r'):
            print("\n↺ Reinicio manual — volviendo a esperar preámbulo")
            reset_state()
            continue

        # ── Timeout: terminar si no llegan frames nuevos ─────
        if (state == "RECEIVING" and
                last_rx_time is not None and
                time.time() - last_rx_time > TIMEOUT_SECONDS):
            print(f"\n⚠ Timeout ({TIMEOUT_SECONDS}s sin frames nuevos)")
            print(f"  Frames recibidos: {len(received)}/{total_frames}")
            elapsed = time.time() - start_time if start_time else 0
            _reconstruct(received, total_frames, elapsed)
            reset_state()
            continue

        # ── Máquina de estados ───────────────────────────────
        if corners is None:
            continue

        symbols, threshold = decode_frame(gray, corners, bgr=frame)
        if symbols is None:
            continue

        if state == "WAITING":
            if is_preamble(symbols, threshold):
                print("✓ Preámbulo detectado — sincronizado")
                state = "SYNCED"

        elif state == "SYNCED":
            if not is_preamble(symbols, threshold):
                print("✓ Inicio de datos detectado")
                state = "RECEIVING"

        if state == "RECEIVING":
            if is_end_frame(symbols, threshold):
                elapsed = time.time() - start_time if start_time else 0
                print(f"\n✓ Frame de fin detectado ({elapsed:.2f}s)")
                _reconstruct(received, total_frames, elapsed)
                reset_state()

            else:
                result = parse_packet(symbols, threshold)
                if result is not None:
                    seq, total, payload, crc_ok = result
                    total_frames = total

                    if start_time is None:
                        start_time   = time.time()
                        last_rx_time = start_time
                        print("  ⏱ Tiempo iniciado")

                    if not crc_ok:
                        crc_errors += 1
                        print(f"  ✗ Frame {seq}: CRC error")
                    elif seq not in received:
                        received[seq]= payload
                        last_rx_time = time.time()  # resetear timeout
                        print(f"  ✓ Frame {seq+1}/{total} "
                              f"({len(payload)} bits)")

    cap.release()
    cv2.destroyAllWindows()


def _reconstruct(received: dict, total_frames: int, elapsed: float):
    if total_frames is None:
        print("No se recibió ningún frame.")
        return

    print(f"\n{'='*50}")
    print(f"Frames recibidos : {len(received)}/{total_frames}")
    print(f"Frames perdidos  : {total_frames - len(received)}")
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


if __name__ == '__main__':
    receive()