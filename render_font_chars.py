"""
从游戏字体文件批量渲染多尺寸字符模板
用法: python render_font_chars.py

会生成每个字符在 50~150px 多个尺寸的模板，
覆盖游戏可能使用的各种渲染尺寸。
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ===== 配置 =====
FONT_PATH = r"C:\Users\ThinkPad\Downloads\鸣潮字体\WuWa Lahai-Roi Regular.ttf"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_db")
os.makedirs(DB_PATH, exist_ok=True)

# 需要渲染的字符（大写、小写、数字、标点）
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.%"

# 多尺寸覆盖
SIZES = [50, 65, 80, 95, 110, 125, 140]

# 输出画布大小
CANVAS = 200
# 字符在画布中的目标比例
CHAR_SCALE = 0.75


def render_char(font, char, target_px):
    """
    用PIL渲染单个字符到指定像素大小。
    返回 (灰度图array, 实际渲染尺寸)
    """
    # 先用大尺寸渲染再缩放到目标大小，保证抗锯齿质量
    render_size = target_px * 2
    font_large = ImageFont.truetype(FONT_PATH, render_size)

    # 获取字符边界
    bbox = font_large.getbbox(char)
    if bbox is None:
        return None, 0

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    # 创建透明底图像
    padding = render_size // 2
    img_w = bw + padding * 2
    img_h = bh + padding * 2
    img = Image.new('L', (img_w, img_h), 255)  # 白底
    draw = ImageDraw.Draw(img)

    # 居中绘制
    draw.text((-bbox[0] + padding, -bbox[1] + padding), char,
              fill=0, font=font_large)

    # 裁剪到字符边界框
    arr = np.array(img, dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(255 - binary)  # 找黑色字符像素
    if coords is None:
        return None, 0

    x, y, w, h = cv2.boundingRect(coords)
    char_img = arr[y:y+h, x:x+w]

    # 缩放到目标像素大小
    scale = target_px / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, max(h, w)


def place_on_canvas(char_img, canvas_size=CANVAS):
    """将字符居中放在画布上，占画布一定比例"""
    h, w = char_img.shape

    # 目标字符大小 = 画布 * CHAR_SCALE
    target_char = int(canvas_size * CHAR_SCALE)
    scale = target_char / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    scaled = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 灰底画布
    canvas = np.full((canvas_size, canvas_size), 128, dtype=np.uint8)
    xo = (canvas_size - new_w) // 2
    yo = (canvas_size - new_h) // 2
    canvas[yo:yo+new_h, xo:xo+new_w] = scaled

    return canvas


def main():
    if not os.path.exists(FONT_PATH):
        print(f"❌ 字体文件不存在: {FONT_PATH}")
        return

    print(f"🔤 字体: {FONT_PATH}")
    print(f"📐 渲染尺寸: {SIZES}")
    print(f"🔢 字符数: {len(CHARS)}")
    print(f"📦 总计模板: {len(CHARS) * len(SIZES)} 个\n")

    total = 0
    for char in CHARS:
        for size in SIZES:
            try:
                # 渲染字符到目标像素大小
                font = ImageFont.truetype(FONT_PATH, size)
                bbox = font.getbbox(char)
                if bbox is None:
                    continue

                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # 带padding渲染
                pad = 20
                img_w = bw + pad * 2
                img_h = bh + pad * 2
                img = Image.new('L', (img_w, img_h), 255)
                draw = ImageDraw.Draw(img)
                draw.text((-bbox[0] + pad, -bbox[1] + pad), char,
                          fill=0, font=font)

                arr = np.array(img, dtype=np.uint8)

                # 找字符边界
                _, binary = cv2.threshold(arr, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                coords = cv2.findNonZero(255 - binary)
                if coords is None:
                    continue

                x, y, bw, bh = cv2.boundingRect(coords)
                char_cut = arr[y:y+bh, x:x+bw]

                # 放到200x200画布
                canvas = place_on_canvas(char_cut)

                # 生成文件名
                safe_char = char
                if char == '%':
                    safe_char = 'percent'
                elif char == '.':
                    safe_char = 'dot'

                fname = f"{safe_char}_font_{size}px.png"
                fpath = os.path.join(DB_PATH, fname)

                # 保存
                Image.fromarray(canvas).save(fpath)
                total += 1

            except Exception as e:
                print(f"  ⚠ '{char}'@{size}px 失败: {e}")

    print(f"\n✅ 已生成 {total} 个模板到 {DB_PATH}/")
    print(f"\n📝 char_mapping.py 添加代码:")
    print("    复制以下内容到 CHAR_MAP 中：")

    # 只输出一个尺寸的映射作为示例（其他尺寸复用相同映射）
    for char in CHARS:
        safe_char = char
        if char == '%':
            safe_char = 'percent'
        elif char == '.':
            safe_char = 'dot'

        for size in SIZES[:1]:  # 只示例第一个尺寸
            if char == '.':
                print(f"    '{safe_char}_font_{size}px': '{char}',")
            elif char == '%':
                print(f"    '{safe_char}_font_{size}px': '{char}',")
            elif char.isdigit():
                print(f"    '{safe_char}_font_{size}px': '{char}',")
            else:
                print(f"    '{safe_char}_font_{size}px': '{char}',")


if __name__ == "__main__":
    main()
