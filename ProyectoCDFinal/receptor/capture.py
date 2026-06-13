import cv2
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import *


def find_fiducials_robust(gray: np.ndarray, debug: bool = False):
    """
    Pipeline de detección con fallback geométrico:
    1. Buscar los 4 por contornos
    2. Si solo hay 3, inferir el 4to por geometría
    3. Si hay menos de 3, intentar template matching
    """
    candidates = _get_all_candidates(gray)

    if len(candidates) >= 4:
        candidates.sort(key=lambda x: x[2], reverse=True)
        top4 = [(x, y) for x, y, _ in candidates[:4]]
        corners = sort_corners(top4)
        return corners, "contornos_4"

    if len(candidates) == 3:
        pts3 = [(x, y) for x, y, _ in candidates]
        fourth = _infer_fourth_corner(pts3, gray.shape)  # <- agregar gray.shape
        if fourth is not None:
            all4 = pts3 + [fourth]
            corners = sort_corners(all4)
            return corners, "geometria_inferida"

    corners = _find_by_template(gray, debug)
    if corners is not None:
        return corners, "template"

    return None, "no_encontrado"


def _get_all_candidates(gray):
    """
    Extrae candidatos a fiducial.
    NUEVO: solo busca en las 4 esquinas de la imagen (25% de cada lado).
    Los fiduciales siempre aparecen cerca de las esquinas del frame transmitido.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates = []
    h, w = gray.shape

    # NUEVO: definir zonas de búsqueda — solo esquinas
    corner_frac = 0.35  # buscar en el 35% más cercano a cada esquina
    zones = [
        (0,           0,           int(w*corner_frac), int(h*corner_frac)),   # TL
        (int(w*(1-corner_frac)), 0, w,                 int(h*corner_frac)),   # TR
        (0,           int(h*(1-corner_frac)), int(w*corner_frac), h),         # BL
        (int(w*(1-corner_frac)), int(h*(1-corner_frac)), w,        h),        # BR
    ]

    for block_size in [11, 21]:
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 2
        )
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 150 or area > w * h * 0.08:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) != 4:
                continue

            child = hierarchy[i][2]
            if child == -1:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            if not (0.6 < aspect < 1.6):
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # NUEVO: verificar que el candidato está en alguna zona de esquina
            in_zone = any(
                zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                for zx1, zy1, zx2, zy2 in zones
            )
            if not in_zone:
                continue

            # Evitar duplicados
            is_dup = any(
                abs(cx - ex) < 20 and abs(cy - ey) < 20
                for ex, ey, _ in candidates
            )
            if not is_dup:
                candidates.append((cx, cy, area))

    return candidates

def _infer_fourth_corner(pts3, image_shape):
    """
    Dado 3 puntos de un rectángulo, infiere el 4to.
    Incluye validaciones estrictas de consistencia geométrica.
    """
    img_h, img_w = image_shape
    best_corner = None
    best_score = float('inf')

    for i in range(3):
        others = [pts3[j] for j in range(3) if j != i]
        p_opposite = pts3[i]
        p_a = others[0]
        p_b = others[1]

        p4 = (p_a[0] + p_b[0] - p_opposite[0],
              p_a[1] + p_b[1] - p_opposite[1])

        # --- Validaciones ---

        # 1. Dentro de la imagen con margen
        margin = 10
        if not (margin < p4[0] < img_w - margin and
                margin < p4[1] < img_h - margin):
            continue

        all4 = pts3 + [p4]

        # 2. No demasiado cerca de ningún punto existente
        min_dist = min(
            np.linalg.norm(np.array(p4) - np.array(p))
            for p in pts3
        )
        # La distancia mínima entre fiduciales debe ser al menos
        # 10% de la diagonal de la imagen
        min_allowed = 0.10 * np.sqrt(img_w**2 + img_h**2)
        if min_dist < min_allowed:
            continue

        # 3. Los 4 lados deben tener longitudes similares
        #    (es un cuadrado/rectángulo, no un rombo deformado)
        sorted4 = sort_corners(all4)
        tl, tr, bl, br = [np.array(p) for p in sorted4]
        top    = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left   = np.linalg.norm(bl - tl)
        right  = np.linalg.norm(br - tr)

        # Lados opuestos deben ser similares (ratio < 1.4)
        if top == 0 or bottom == 0 or left == 0 or right == 0:
            continue
        horiz_ratio = max(top, bottom) / min(top, bottom)
        vert_ratio  = max(left, right) / min(left, right)
        if horiz_ratio > 1.4 or vert_ratio > 1.4:
            continue

        # 4. El área del cuadrilátero debe ser razonable
        #    (al menos 5% del área de la imagen)
        area = _quad_area(sorted4)
        min_area = 0.05 * img_w * img_h
        max_area = 0.95 * img_w * img_h
        if not (min_area < area < max_area):
            continue

        # 5. Score de rectangularidad
        score = _rectangularity_score(all4)
        if score < best_score:
            best_score = score
            best_corner = p4

    # Aceptar solo si el resultado es suficientemente rectangular
    if best_score < 20.0:  # máx 20° de desviación promedio
        return best_corner
    return None


def _quad_area(pts4):
    """Área de un cuadrilátero por fórmula del zapato (shoelace)."""
    pts = np.array(pts4, dtype=np.float32)
    n = len(pts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2


def _rectangularity_score(pts4):
    """
    Mide qué tan rectangular es un cuadrilátero.
    Score bajo = más rectangular.
    """
    corners = sort_corners(pts4)
    tl, tr, bl, br = corners

    def angle_between(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 90
        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1, 1)
        return abs(np.degrees(np.arccos(cos_a)) - 90)

    angles = [
        angle_between(
            np.array(tr) - np.array(tl),
            np.array(bl) - np.array(tl)
        ),
        angle_between(
            np.array(tl) - np.array(tr),
            np.array(br) - np.array(tr)
        ),
        angle_between(
            np.array(br) - np.array(bl),
            np.array(tl) - np.array(bl)
        ),
        angle_between(
            np.array(bl) - np.array(br),
            np.array(tr) - np.array(br)
        ),
    ]
    return sum(angles)


def _find_by_template(gray, debug=False):
    """
    Fallback: busca fiduciales por template matching
    buscando en las 4 esquinas de la imagen.
    """
    fid_px = FIDUCIAL_SIZE * CELL_SIZE
    template = np.zeros((fid_px, fid_px), dtype=np.uint8)
    c = CELL_SIZE
    template[c:fid_px-c, c:fid_px-c] = 255
    c2 = 2 * CELL_SIZE
    template[c2:fid_px-c2, c2:fid_px-c2] = 0

    h, w = gray.shape
    centers = []

    regions = [
        (0,      0,      w//3, h//3),
        (2*w//3, 0,      w,    h//3),
        (0,      2*h//3, w//3, h   ),
        (2*w//3, 2*h//3, w,    h   ),
    ]

    for (x1, y1, x2, y2) in regions:
        roi = gray[y1:y2, x1:x2]
        best_val = -1
        best_loc = None

        for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            tw = max(10, int(fid_px * scale))
            th = max(10, int(fid_px * scale))
            t_scaled = cv2.resize(template, (tw, th))

            if roi.shape[0] < th or roi.shape[1] < tw:
                continue

            result = cv2.matchTemplate(roi, t_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = (max_loc[0] + tw//2 + x1,
                            max_loc[1] + th//2 + y1)

        if best_val > 0.4 and best_loc is not None:
            centers.append(best_loc)

    if len(centers) == 4:
        return sort_corners(centers)
    return None


def sort_corners(pts):
    """Ordena 4 puntos como [TL, TR, BL, BR]."""
    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[0], bot[1]]


def rectify_frame(image: np.ndarray, corners):
    """Rectifica perspectiva usando los 4 corners de fiduciales."""
    fid_half = (FIDUCIAL_SIZE * CELL_SIZE) // 2
    dst_pts = np.float32([
        [fid_half,           fid_half          ],
        [FRAME_W - fid_half, fid_half          ],
        [fid_half,           FRAME_H - fid_half],
        [FRAME_W - fid_half, FRAME_H - fid_half],
    ])
    src_pts = np.float32(corners)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None
    return cv2.warpPerspective(image, H, (FRAME_W, FRAME_H))


def extract_symbols(rectified_gray: np.ndarray) -> np.ndarray:
    """Lee el valor de cada celda de datos con margen para evitar bordes."""
    from transmisor.frame_builder import get_data_cells
    data_cells = get_data_cells()
    symbols = []
    margin = CELL_SIZE // 4

    for row, col in data_cells:
        y = row * CELL_SIZE + margin
        x = col * CELL_SIZE + margin
        h = w = CELL_SIZE - 2 * margin
        roi = rectified_gray[y:y+h, x:x+w]
        val = int(np.median(roi))
        symbols.append(val)

    return np.array(symbols, dtype=np.uint8)