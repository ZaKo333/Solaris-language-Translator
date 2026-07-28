"""
数据库构建工具
验证 + 从大图切分字符
"""

import os
import cv2
from char_mapping import get_char_map


def verify_database(db_path="char_db"):
    """验证数据库完整性"""
    char_map = get_char_map()
    if not char_map:
        print("[!] CHAR_MAP 为空！请先在 char_mapping.py 中添加映射")
        return

    if not os.path.exists(db_path):
        print(f"[!] 目录 '{db_path}' 不存在！")
        return

    files = os.listdir(db_path)

    print("=" * 60)
    print("字符数据库完整性检查")
    print("=" * 60)

    ok = 0
    missing = []
    for fname, char in char_map.items():
        found = False
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            if (fname + ext) in files or fname in files:
                found = True
                break
        if found:
            ok += 1
        else:
            missing.append((fname, char))

    print(f"✓ 已匹配: {ok}")
    print(f"✗ 缺失图片: {len(missing)}")
    for fname, char in missing:
        print(f"   - 缺少 '{fname}' (→ '{char}')")

    mapped_names = set(char_map.keys())
    base_names = set(os.path.splitext(f)[0] for f in files)
    extra = base_names - mapped_names
    if extra:
        print(f"\n⚠ 以下文件在 char_db 中但未在映射表中:")
        for f in sorted(extra):
            for ff in files:
                if os.path.splitext(ff)[0] == f:
                    print(f"   - {ff}")
    print("=" * 60)


def split_sheet(sheet_path, rows, cols, output_dir="char_db/training"):
    """
    从字符表大图切分单个字符
    用法: python build_db.py split 图片.png 5 13
    """
    img = cv2.imread(sheet_path)
    if img is None:
        print(f"[!] 无法读取图片: {sheet_path}")
        return

    h, w = img.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    os.makedirs(output_dir, exist_ok=True)

    print(f"切分 {rows}行×{cols}列, 每格 {cell_w}×{cell_h}")
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = (c + 1) * cell_w, (r + 1) * cell_h
            cell = img[y1:y2, x1:x2]
            out_path = os.path.join(output_dir, f"char_r{r}_c{c}.png")
            cv2.imwrite(out_path, cell)
            print(f"   [{r},{c}] → {os.path.basename(out_path)}")

    print(f"\n✓ 已切分 {rows*cols} 个字符到 {output_dir}/")
    print("请重命名文件并更新 CHAR_MAP")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python build_db.py verify              # 验证数据库")
        print("  python build_db.py split <图> <行> <列> # 从大图切分")
    elif sys.argv[1] == "verify":
        verify_database()
    elif sys.argv[1] == "split" and len(sys.argv) >= 5:
        split_sheet(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        print("参数错误，请使用 python build_db.py 查看帮助")
