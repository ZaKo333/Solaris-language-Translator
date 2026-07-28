"""
从游戏截图中提取真实字符模板（复用 app.py 的最新分割引擎）

用法：
  python extract_templates.py <图片路径> <已知文字>

示例：
  python extract_templates.py banana.png Banana

会保存到 char_db/{char}_game_{id}.png
_matcher 的 load_database() 会自动识别 _game_ 命名的模板
"""
import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
CHAR_DB = os.path.join(BASE_DIR, "char_db")

# 复用 app.py 的最新分割引擎
from app import split_into_chars


def get_filename(char):
    """与 generate_font_templates.py 一致的文件名规则"""
    if char == '.':
        return "dot"
    elif char == '%':
        return "percent"
    elif char.isupper():
        return f"{char}1"
    else:
        return char


def get_next_game_id(save_key):
    """获取下一个可用的 _game_ 编号"""
    max_id = 0
    if not os.path.exists(CHAR_DB):
        return 1
    for fname in os.listdir(CHAR_DB):
        if fname.startswith(f"{save_key}_game_"):
            try:
                idx = int(fname.split('_game_')[1].split('.')[0])
                max_id = max(max_id, idx)
            except:
                pass
    return max_id + 1


def main():
    if len(sys.argv) < 3:
        print("用法: python extract_templates.py <图片路径> <已知文字>")
        print("示例: python extract_templates.py banana.png Banana")
        sys.exit(1)

    img_path = sys.argv[1]
    expected_text = sys.argv[2]

    if not os.path.isabs(img_path):
        img_path = os.path.join(BASE_DIR, img_path)

    if not os.path.exists(img_path):
        print(f"❌ 文件不存在: {img_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"📷 截图: {os.path.basename(img_path)}")
    print(f"📝 已知文字 ({len(expected_text)}字): {expected_text}")
    print("=" * 60)

    # 用 app.py 的最新分割引擎
    chars = split_into_chars(img_path)
    if not chars:
        print("❌ 未能分割出任何字符")
        sys.exit(1)

    print(f"\n✂️ 分割出 {len(chars)} 个字符")
    if len(chars) != len(expected_text):
        print(f"⚠️  分割数({len(chars)}) ≠ 文本长度({len(expected_text)})")
        for i in range(min(len(chars), len(expected_text))):
            h, w = chars[i].shape
            print(f"  [{i}] '{expected_text[i]}' → {w}×{h}")
        if len(chars) > len(expected_text):
            for i in range(len(expected_text), len(chars)):
                h, w = chars[i].shape
                print(f"  [{i}] (多余) → {w}×{h}")
        proceed = input("  继续？(y/N): ").strip().lower()
        if proceed != 'y':
            print("❌ 已取消")
            sys.exit(1)
    else:
        for i in range(len(chars)):
            h, w = chars[i].shape
            print(f"  [{i}] '{expected_text[i]}' → {w}×{h}")

    os.makedirs(CHAR_DB, exist_ok=True)

    # 保存模板（无需更新 char_mapping.py，load_database 会自动识别 _game_ 命名）
    new_entries = []
    for i in range(len(chars)):
        char = expected_text[i]
        save_key = get_filename(char)
        game_id = get_next_game_id(save_key)
        game_key = f"{save_key}_game_{game_id}"
        save_path = os.path.join(CHAR_DB, f"{game_key}.png")

        Image.fromarray(chars[i]).save(save_path)
        print(f"  ✅ [{i}] '{char}' → {game_key}.png ({chars[i].shape[1]}×{chars[i].shape[0]})")

    print(f"\n✅ 完成！提取了 {len(chars)} 个游戏模板到 {CHAR_DB}")
    print(f"   重启服务后用 test_image.py 测试")


if __name__ == "__main__":
    main()
