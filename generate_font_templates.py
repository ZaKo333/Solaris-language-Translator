"""
从TTF字体文件自动生成全部64个字符模板
用法：python generate_font_templates.py

会用TTF渲染 A-Z a-z 0-9 . % 共64个字符，
保存到 char_db/，自动清理旧模板和映射。
"""
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_DB = os.path.join(BASE_DIR, "char_db")
MAPPING_FILE = os.path.join(BASE_DIR, "char_mapping.py")

# ===== 配置 =====
TTF_PATH = r"C:\Users\ThinkPad\Downloads\鸣潮字体\WuWa Lahai-Roi Regular.ttf"
FONT_SIZE = 140         # 渲染尺寸（越大细节越丰富，匹配器会缩放到140px）
CANVAS_SIZE = 200
MARGIN = 6

# 索拉里斯64字符集
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.%"


def get_filename(char):
    """返回文件名（不含扩展名）"""
    if char == '.':
        return "dot"
    elif char == '%':
        return "percent"
    elif char.isupper():
        return f"{char}1"
    else:
        return char


def render_char(char, font):
    """渲染单个字符，返回裁剪+填充为正方形的图像"""
    canvas = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (CANVAS_SIZE - tw) // 2 - bbox[0]
    y = (CANVAS_SIZE - th) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=255)

    img = np.array(canvas, dtype=np.uint8)

    coords = cv2.findNonZero(img)
    if coords is None:
        return None

    cx, cy, cw, ch = cv2.boundingRect(coords)
    if cw < 2 or ch < 2:
        return None

    x1 = max(0, cx - MARGIN)
    y1 = max(0, cy - MARGIN)
    x2 = min(CANVAS_SIZE, cx + cw + MARGIN)
    y2 = min(CANVAS_SIZE, cy + ch + MARGIN)
    cropped = img[y1:y2, x1:x2]

    h, w = cropped.shape
    side = max(h, w) + 8
    square = np.zeros((side, side), dtype=np.uint8)
    xo = (side - w) // 2
    yo = (side - h) // 2
    square[yo:yo+h, xo:xo+w] = cropped

    return square


def build_char_map_entries():
    """生成CHAR_MAP字典条目的字符串"""
    lines = []
    # 大写 A-Z
    lines.append("    # ===== 大写字母 A-Z (TTF生成) =====")
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        lines.append(f"    '{ch}1': '{ch}',")
    # 小写 a-z
    lines.append("")
    lines.append("    # ===== 小写字母 a-z (TTF生成) =====")
    for ch in "abcdefghijklmnopqrstuvwxyz":
        lines.append(f"    '{ch}': '{ch}',")
    # 数字
    lines.append("")
    lines.append("    # ===== 数字 0-9 (TTF生成) =====")
    for ch in "0123456789":
        lines.append(f"    '{ch}': '{ch}',")
    # 标点
    lines.append("")
    lines.append("    # ===== 标点符号 (TTF生成) =====")
    lines.append("    'dot': '.',")
    lines.append("    'percent': '%',")
    return '\n'.join(lines)


def rewrite_mapping_file(new_entries_str):
    """重写 char_mapping.py，保留文件头尾但替换 CHAR_MAP 内容"""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到 CHAR_MAP = { 和匹配的 }
    start_marker = 'CHAR_MAP = {'
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("❌ 找不到 CHAR_MAP = {")
        return False

    # 找匹配的 } — 数大括号深度
    depth = 0
    i = start_idx + len(start_marker)
    while i < len(content):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            if depth == 0:
                end_idx = i
                break
            depth -= 1
        i += 1
    else:
        print("❌ 找不到 CHAR_MAP 的结尾 }")
        return False

    # 重建文件：start_marker 前的内容 + 新entries + 补上'}' + end之后的内容
    before = content[:start_idx + len(start_marker)]
    after = content[end_idx + 1:]

    new_content = before + '\n' + new_entries_str + '\n}\n' + after

    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


def main():
    print("=" * 60)
    print("  TTF字体 → 字符模板 生成器")
    print("=" * 60)

    # 检查字体
    if not os.path.exists(TTF_PATH):
        print(f"❌ 字体文件不存在: {TTF_PATH}")
        sys.exit(1)

    try:
        font = ImageFont.truetype(TTF_PATH, FONT_SIZE)
        print(f"✅ 加载字体: {os.path.basename(TTF_PATH)} ({FONT_SIZE}pt)")
    except Exception as e:
        print(f"❌ 加载字体失败: {e}")
        sys.exit(1)

    # 验证可渲染
    test_img = Image.new('L', (100, 100), 0)
    test_draw = ImageDraw.Draw(test_img)
    printable = 0
    for ch in CHARS:
        try:
            bbox = test_draw.textbbox((0, 0), ch, font=font)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                printable += 1
        except:
            pass
    print(f"📝 字符集: {len(CHARS)}个字符（可渲染: {printable}个）")

    # 清理旧的 game 模板
    os.makedirs(CHAR_DB, exist_ok=True)
    removed = 0
    for fname in list(os.listdir(CHAR_DB)):
        if '_game_' in fname:
            os.remove(os.path.join(CHAR_DB, fname))
            removed += 1
    old_pct = os.path.join(CHAR_DB, "%.png")
    if os.path.exists(old_pct):
        os.remove(old_pct)
        removed += 1
    print(f"🧹 清理: {removed}个旧模板文件")

    # 生成模板
    saved = 0
    failed = []
    for ch in CHARS:
        fname = get_filename(ch)
        img = render_char(ch, font)
        if img is None:
            failed.append(ch)
            continue

        save_path = os.path.join(CHAR_DB, f"{fname}.png")
        Image.fromarray(img).save(save_path)
        print(f"  [{saved+1}] '{ch}' → {fname}.png")
        saved += 1

    print(f"\n✅ 生成 {saved} 个模板")

    # 重写映射文件
    entries = build_char_map_entries()
    if rewrite_mapping_file(entries):
        print(f"✅ 映射文件已更新: {MAPPING_FILE}")
    else:
        print(f"❌ 映射文件更新失败!")
        sys.exit(1)

    print(f"\n下一步：运行 python test_image.py D:\\项目\\banana.png 验证")


if __name__ == "__main__":
    main()
