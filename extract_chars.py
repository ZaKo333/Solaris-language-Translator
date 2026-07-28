"""
从已知文字的游戏截图提取字符，自动添加到标准库
用法: python extract_chars.py <截图路径> <对应的英文字符串>
示例: python extract_chars.py D:\项目\game.png happy
"""

import os, sys, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 3:
    print("用法: python extract_chars.py <截图路径> <对应的英文字符串>")
    print("示例: python extract_chars.py D:\项目\game.png happy")
    sys.exit(1)

img_path = sys.argv[1]
label_text = sys.argv[2]

if not os.path.exists(img_path):
    print(f"❌ 文件不存在: {img_path}")
    sys.exit(1)

import cv2
import numpy as np
from PIL import Image

DB_PATH = os.path.join(BASE_DIR, "char_db")
os.makedirs(DB_PATH, exist_ok=True)

# 读取并预处理
print(f"📷 读取: {img_path}")
pil_img = Image.open(img_path).convert('L')
img = np.array(pil_img, dtype=np.uint8)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
binary = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 4
)
kernel_close = np.ones((3, 5), np.uint8)
connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
kernel_open = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_open)

h_img, w_img = img.shape

# 水平投影找行
h_proj = np.sum(cleaned > 0, axis=1)
h_thresh = w_img * 0.03
in_line = False
line_ranges = []
for y in range(h_img):
    if h_proj[y] > h_thresh and not in_line:
        ls = y
        in_line = True
    elif h_proj[y] <= h_thresh and in_line:
        in_line = False
        if y - ls > 5:
            line_ranges.append((ls, y))
if in_line:
    line_ranges.append((ls, h_img - 1))

if not line_ranges:
    print("❌ 未检测到文字行")
    sys.exit(1)

# 过滤噪点行（高度太小的行是噪点）
line_ranges = [(y1, y2) for y1, y2 in line_ranges if y2 - y1 > 15]
if not line_ranges:
    print("❌ 过滤后无有效文字行")
    sys.exit(1)

print(f"📐 检测到 {len(line_ranges)} 行文字 (过滤后)")
y1, y2 = line_ranges[0]  # 取第一行有效文字
print(f"   行: y={y1}~{y2} (高{y2-y1}px)")

# 垂直投影
line_bin = cleaned[y1:y2, :]
line_h = y2 - y1
v_proj = np.sum(line_bin > 0, axis=0)
v_thresh = line_h * 0.05

in_char = False
char_starts, char_ends = [], []
for x in range(w_img):
    if v_proj[x] > v_thresh and not in_char:
        char_starts.append(x)
        in_char = True
    elif v_proj[x] <= v_thresh and in_char:
        char_ends.append(x)
        in_char = False
if in_char:
    char_ends.append(w_img - 1)

# 合并太近的
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

print(f"✂️ 切出 {len(char_starts)} 个字符, 标签文字: '{label_text}' ({len(label_text)}个字符)")

if len(char_starts) != len(label_text):
    print(f"⚠ 数量不匹配！切出了 {len(char_starts)} 个，但标签有 {len(label_text)} 个")
    print(f"   切出的位置:")
    for i, (xs, xe) in enumerate(zip(char_starts, char_ends)):
        print(f"     字符{i+1}: x={xs}~{xe} (宽{xe-xs}px)")
    ans = input("是否继续保存？(y/n): ")
    if ans.lower() != 'y':
        sys.exit(1)

# 保存每个字符
print(f"\n💾 保存到 {DB_PATH}/:")
for i, (xs, xe) in enumerate(zip(char_starts, char_ends)):
    if i >= len(label_text):
        break
    c = label_text[i]
    cw = xe - xs
    char_patch = blurred[y1:y2, xs:xe]

    # 填充到正方形
    hc, wc = char_patch.shape
    side = max(hc, wc) + 8
    square = np.zeros((side, side), dtype=np.uint8)
    xo = (side - wc) // 2
    yo = (side - hc) // 2
    square[yo:yo+hc, xo:xo+wc] = char_patch

    # 生成文件名：字符_来源_序号.png
    # 检查是否已有同名文件
    count = 1
    while True:
        fname = f"{c}_game_{count}.png"
        fpath = os.path.join(DB_PATH, fname)
        if not os.path.exists(fpath):
            break
        count += 1

    # 用 PIL 保存（支持中文路径）
    Image.fromarray(square).save(fpath)
    print(f"   字符{i+1} '{c}' → {fname}  ({wc}x{hc}px)")

print(f"\n✅ 已保存 {min(len(char_starts), len(label_text))} 个字符")
print("\n📝 下一步：编辑 char_mapping.py 添加映射")
for i, (xs, xe) in enumerate(zip(char_starts, char_ends)):
    if i >= len(label_text):
        break
    c = label_text[i]
    print(f"   '{c}_game_{i+1}': '{c}',")
