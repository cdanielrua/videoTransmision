"""
Receptor multi-frame con timeout, YCbCr y reinicio manual.
Ejecutar: python main_rx.py
"""
import cv2
import numpy as np
import time
from receptor.capture import find_fiducials_robust
from receptor.rx_state import Receiver, decode_frame
from common.config import *


def receive():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    print("Receptor multi-frame iniciado.")
    print("Esperando preámbulo del transmisor...")
    print("[D] debug  [R / click derecho] reiniciar  [Q] salir\n")

    rx         = Receiver()
    debug_mode = False

    restart_requested = [False]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            restart_requested[0] = True

    cv2.namedWindow("Receptor")
    cv2.setMouseCallback("Receptor", on_mouse)

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
            "DONE":      (255, 255, 0),
        }.get(rx.state, (255, 255, 255))

        cv2.putText(display, f"Estado: {rx.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if corners is not None:
            pts = np.array([corners[0], corners[1],
                            corners[3], corners[2]], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, f"[{method}]", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if rx.total_frames is not None:
            cv2.putText(display,
                        f"Frames: {len(rx.received)}/{rx.total_frames}  "
                        f"CRC errors: {rx.crc_errors}",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Mostrar countdown del timeout (solo si falta al menos 1 frame)
        if rx.state == "RECEIVING" and rx.last_rx_time is not None:
            missing = (rx.total_frames - len(rx.received)
                      if rx.total_frames is not None else None)
            if missing is not None and missing >= 1:
                remaining = TIMEOUT_SECONDS - (time.time() - rx.last_rx_time)
                if remaining > 0:
                    cv2.putText(display,
                                f"Timeout en: {remaining:.1f}s",
                                (10, display.shape[0] - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 165, 255), 1)

        if rx.state == "DONE":
            cv2.putText(display,
                        "Recepcion completa - [R]/click derecho para recibir otro",
                        (10, display.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        cv2.imshow("Receptor", display)

        # ── Teclas ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug: {'ON' if debug_mode else 'OFF'}")

        if (key == ord('r') or restart_requested[0]) and rx.state == "DONE":
            print("\n↺ Reinicio manual — volviendo a esperar preámbulo")
            rx.restart()
        restart_requested[0] = False

        # ── Si ya terminó, esperar reinicio manual ───────────
        if rx.state == "DONE":
            continue

        # ── Timeout: abortar si no llegan frames nuevos ──────
        if rx.check_timeout():
            print(f"\n⚠ Timeout ({TIMEOUT_SECONDS}s sin frames nuevos)")
            print(f"  Frames recibidos: {len(rx.received)}/{rx.total_frames}")
            continue

        # ── Máquina de estados ───────────────────────────────
        if corners is None:
            continue

        symbols, threshold = decode_frame(gray, corners, bgr=frame)
        if symbols is None:
            continue

        rx.process(symbols, threshold)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    receive()
