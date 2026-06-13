# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A half-duplex optical "modem" for the Comunicaciones Digitales course (UdeA). It transmits a text
message by encoding it as a grid of black/white/gray cells displayed fullscreen on one computer's
screen (`main_tx.py`), and a webcam on a second computer reads that screen, decodes the grid back
into bits, and reconstructs the text (`main_rx.py`). There is no network connection between the two
machines — the "channel" is the screen-to-camera optical link, which introduces perspective
distortion, lighting variation, blur and sensor noise.

## Running

Requires `opencv-python` and `numpy` (no requirements.txt / venv is checked in).

```
python main_tx.py          # transmitter: shows fullscreen frames, SPACE to start, Q to cancel/close
python main_rx.py          # receiver: reads webcam index 1, D toggles debug view, Q to quit
python test_loopback.py     # no-camera sanity test (currently broken, see Known issues)
```

Quick module-level smoke tests (each file has an `if __name__ == '__main__'` block):
```
python -m transmisor.encoder         # prints bit/Manchester/OOK encoding of "Hola"
python -m transmisor.frame_builder   # builds one frame and writes frame_v2.png
```

## Architecture

### Grid / frame geometry (`common/config.py`)
Everything is built around a fixed 40×30 grid of 20px cells → an 800×600 px frame
(`GRID_COLS`, `GRID_ROWS`, `CELL_SIZE`, `FRAME_W`, `FRAME_H`). Key reserved regions, computed by
`transmisor/frame_builder.get_data_cells()`:
- **4 corner fiducials** (`FIDUCIAL_SIZE` = 5×5 cells): bright green ring on black, used by the
  receiver to find the frame and compute a homography. Built last in `build_frame` so they're never
  overwritten by data.
- **Pilot row** (`PILOT_ROW` = row 3): alternating white/black cells, used by the receiver to
  calibrate the OOK decision threshold per-frame (`receptor/decoder.calibrate_threshold`).
- All remaining cells are **data cells**, in row-major order as returned by `get_data_cells()`. This
  ordering is the contract between encoder and decoder — both sides must enumerate cells the same way.

### Bit/symbol pipeline (transmit side)
`transmisor/encoder.py`:
1. `text_to_bits` — UTF-8/ASCII text → bit array (MSB first).
2. `manchester_encode` — each bit → 2 symbols (`0→[1,0]`, `1→[0,1]`), giving DC balance and helping
   the receiver detect bit-sync errors.
3. `bits_to_ook_symbols` — bit → grayscale value (`0→0`, `1→255`) drawn into a cell.

`transmisor/frame_builder.build_frame(symbols, frame_number, total_frames)` paints data cells, then
the pilot row, then the fiducials (in that order, so later layers win).

### Packet/frame protocol (`common/protocol.py`)
Each data frame's payload begins with a 5-byte header, Manchester-encoded like the rest:
```
[SEQ_HIGH][SEQ_LOW][TOTAL_HIGH][TOTAL_LOW][CRC8] + payload bits
```
- `SEQ`/`TOTAL` are 16-bit (2 bytes each) → frame sequence number and total frame count.
- `CRC8` is computed over the *payload* bits only (`crc8`).
- `build_packet` Manchester-encodes header and payload separately, concatenates, and converts to OOK
  symbols. `parse_packet` reverses this: OOK→bits→Manchester-decode→split header/payload, validates
  CRC, and returns `(seq, total, payload_bits, crc_ok)`.

`main_tx.text_to_frames` chunks the message's bits to fit `payload_syms = total_cells - 80` (80 =
Manchester-encoded header bits), builds one packet per chunk via `build_packet`, pads the last frame
with gray (128) filler if needed, and renders each with `build_frame`.

### Special frames
- **Preamble** (`build_preamble_frame`, `frame_number=-1`): alternating 255/0 symbols across all data
  cells — a recognizable sync pattern. The receiver's `is_preamble` checks the decoded bits match
  this alternating pattern with >85% agreement.
- **End frame** (`build_end_frame`, `frame_number=-2`): all data cells set to 128 (mid-gray). The
  receiver's `is_end_frame` checks symbols are near 128 with >80% agreement.

### Transmit sequence (`main_tx.py`)
Shows the preamble fullscreen and waits for SPACE (lets the operator align the camera), then sends
`PREAMBLE_FRAMES` preamble frames, all data frames (each held for `FRAME_DURATION` seconds), repeats
the last 2 data frames 3× extra for reliability, pauses, then sends the end frame 6×.

### Receive pipeline (`main_rx.py`, `receptor/`)
State machine: `WAITING → SYNCED → RECEIVING → DONE`.
- Per camera frame: `receptor/capture.find_fiducials_robust` locates the 4 green corner fiducials
  (color-based `_find_by_green`, falling back to grayscale concentric-contour detection
  `_get_candidates_gray`, with `_infer_fourth` reconstructing a missing 4th corner via the
  parallelogram property). Returns corners as `[TL, TR, BL, BR]` (`sort_corners`).
- `rectify_frame` warps the perspective to a canonical 800×600 image via homography.
- `decode_frame` converts to YCbCr and uses the luma channel for robustness to color casts, then
  `extract_symbols` reads the median pixel value of each data cell (with an inward margin to avoid
  edge bleed).
- `calibrate_threshold` derives the OOK 0/1 threshold from the pilot row each frame (handles changing
  ambient light/exposure).
- `WAITING`→`SYNCED` on detecting the preamble pattern; `SYNCED`→`RECEIVING` once a non-preamble
  frame arrives; in `RECEIVING`, `parse_packet` extracts `(seq, total, payload, crc_ok)` and stores
  good frames in `received[seq]`; `is_end_frame` or a `TIMEOUT_SECONDS` (3s) gap with no new frames
  triggers `_reconstruct`, which fills missing frames with zero bits, joins all payload bits,
  converts to text via `bits_to_text` (non-printable bytes become `?`, Manchester errors are
  dropped), prints it, and writes `mensaje_recibido.txt`.
- Webcam is opened as `cv2.VideoCapture(1)` (second camera device) — change this index if testing on
  a machine with only one camera.

## Known issues
- `test_loopback.py` imports `find_fiducials` from `receptor.capture`, but the function is named
  `find_fiducials_robust` — this script is currently broken/outdated.
