import sys, os
try:
    marker = os.path.join(os.path.dirname(__file__), "MARKER_RAN.txt")
    with open(marker, "w") as f:
        f.write("Script ran successfully\n")

    # Import matcher
    sys.path.insert(0, os.path.dirname(__file__))
    from matcher import GameCharMatcher
    with open(marker, "a") as f:
        f.write("Import OK\n")

    m = GameCharMatcher()
    ok = m.load_database()
    with open(marker, "a") as f:
        f.write(f"load_database: {ok}\n")
        f.write(f"std_gray count: {len(m.std_gray)}\n")

    # Try matching on game1.png
    test_path = r"D:\项目\game1.png"
    if os.path.exists(test_path):
        from PIL import Image
        import cv2
        import numpy as np
        import imagehash

        pil_img = Image.open(test_path).convert('L')
        img = np.array(pil_img, dtype=np.uint8)

        # Process same way as app.py
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        kernel_close = np.ones((3, 5), np.uint8)
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_open)

        h, w = cleaned.shape
        h_proj = np.sum(cleaned > 0, axis=1)
        h_thresh = w * 0.03
        in_line = False
        line_ranges = []
        for y in range(h):
            if h_proj[y] > h_thresh and not in_line:
                line_start = y
                in_line = True
            elif h_proj[y] <= h_thresh and in_line:
                in_line = False
                if y - line_start > 5:
                    line_ranges.append((line_start, y))
        if in_line:
            line_ranges.append((line_start, h - 1))
        line_ranges = [(y1, y2) for y1, y2 in line_ranges if y2 - y1 > 15]

        with open(marker, "a") as f:
            f.write(f"Lines: {len(line_ranges)}\n")

        results = []
        for y1, y2 in line_ranges:
            line_h = y2 - y1
            line_bin = cleaned[y1:y2, :]
            v_proj = np.sum(line_bin > 0, axis=0)
            v_thresh = line_h * 0.05
            in_char = False
            char_starts, char_ends = [], []
            for x in range(w):
                if v_proj[x] > v_thresh and not in_char:
                    char_starts.append(x)
                    in_char = True
                elif v_proj[x] <= v_thresh and in_char:
                    char_ends.append(x)
                    in_char = False
            if in_char:
                char_ends.append(w - 1)

            if len(char_starts) > 1:
                ms, me = [char_starts[0]], []
                for i in range(1, len(char_starts)):
                    gap = char_starts[i] - char_ends[i-1]
                    avg_w = np.mean([char_ends[j] - char_starts[j] for j in range(i)])
                    if gap < avg_w * 0.33 and avg_w > 0:
                        continue
                    else:
                        me.append(char_ends[i-1])
                        ms.append(char_starts[i])
                me.append(char_ends[-1])
                char_starts, char_ends = ms, me

            for xs, xe in zip(char_starts, char_ends):
                cw = xe - xs
                if cw < 4 or cw > w * 0.7:
                    continue
                char_patch = blurred[y1:y2, xs:xe]
                hc, wc = char_patch.shape
                side = max(hc, wc) + 8
                square = np.zeros((side, side), dtype=np.uint8)
                xo = (side - wc) // 2
                yo = (side - hc) // 2
                square[yo:yo+hc, xo:xo+wc] = char_patch

                char, conf, details = m.match_single_char(square, threshold=0.0)
                with open(marker, "a") as f:
                    f.write(f"  char {len(results)+1}: ({xs},{y1}) {wc}x{hc} -> '{char}' conf={conf:.4f}\n")
                results.append(char if char else '?')

        with open(marker, "a") as f:
            f.write(f"\nRESULT: {''.join(results)}\n")
    else:
        with open(marker, "a") as f:
            f.write(f"Test image not found: {test_path}\n")
except Exception as e:
    import traceback
    with open(os.path.join(os.path.dirname(__file__), "MARKER_RAN.txt"), "a") as f:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
