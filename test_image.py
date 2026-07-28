"""测试工具 - 测试一张游戏截图识别全流程"""
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    print("用法: python test_image.py <游戏截图路径>")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"❌ 文件不存在: {img_path}")
    sys.exit(1)

sys.path.insert(0, BASE_DIR)
from matcher import GameCharMatcher
from PIL import Image
import cv2
import numpy as np
import requests

print("=" * 70)
print(f"📷 测试截图: {img_path}")
print("=" * 70)

# 加载数据库
print("\n🔃 加载字符数据库...")
from char_mapping import get_char_map
cm = get_char_map()
matcher = GameCharMatcher(db_path=os.path.join(BASE_DIR, "char_db"))
ok = matcher.load_database()
print(f"   数据库: {len(matcher.std_gray)}/{len(cm)} 个模板加载成功")
if not ok or len(matcher.std_gray) == 0:
    print("❌ 数据库为空，退出")
    sys.exit(1)

# 使用app.py的切割函数
from app import split_into_chars
print(f"\n✂️ 切割字符...")
chars = split_into_chars(img_path)
print(f"   切出 {len(chars)} 个字符")
if len(chars) == 0:
    print("❌ 未切割出任何字符")
    sys.exit(1)

# 显示每个字符的小预览
for i, c in enumerate(chars):
    hc, wc = c.shape
    print(f"   字符{i+1}: {wc}x{hc}")

# === 匹配 ===
print(f"\n🔍 匹配结果:")
print("-" * 80)
print(f"{'#':>3s} {'大小':>12s} {'→字符':>6s} {'置信度':>9s} {'HOG':>6s} {'Prf':>6s} {'Edge':>6s} {'Asp':>5s} {'NCC':>6s} {'pHash':>5s}")
print("-" * 80)

results = []
all_candidates = []  # 收集每个位置的所有候选，用于词级纠错
for i, char_img in enumerate(chars):
    char, conf, details = matcher.match_single_char(char_img, threshold=0.0)
    # 收集该位置的候选
    if hasattr(matcher, '_last_candidates') and matcher._last_candidates:
        all_candidates.append(matcher._last_candidates[:10])
    char_display = char if char else '?'
    hc, wc = char_img.shape
    hog_s = f"{details.get('hog',0)*100:.1f}%" if details else '-'
    hp = details.get('hproj', 0)
    vp = details.get('vproj', 0)
    prof_s = f"{(hp+vp)/2*100:.1f}%" if details else '-'
    edge_s = f"{details.get('edge',0)*100:.1f}%" if details else '-'
    asp_s = f"{details.get('aspect',0)*100:.1f}%" if details else '-'
    ncc_s = f"{details.get('ncc',0)*100:.1f}%" if details else '-'
    ph_s = f"{details.get('phash',0)*100:.1f}%" if details else '-'
    print(f"{i+1:>3d}  {wc}x{hc:>5d}px  {char_display:>4s}  {conf:>7.1%}  {hog_s:>6s} {prof_s:>6s} {edge_s:>6s} {asp_s:>5s} {ncc_s:>6s} {ph_s:>5s}")

    # 显示top3候选做调试
    try:
        if hasattr(matcher, '_last_top3') and matcher._last_top3:
            top3 = matcher._last_top3[:3]
            print(f"      └→ ", end="")
            for rank, (ch, sc) in enumerate(top3):
                print(f"#{rank+1}='{ch}'({sc:.1%})  ", end="")
            print()
    except:
        pass

    results.append(char_display)

print("-" * 70)

# === 最终结果 ===
raw_text = ''.join(results)
print(f"\n📝 识别结果: {raw_text}")

# === 词典纠错（带逐位置候选）===
from matcher import spell_correct_word
if len(all_candidates) == len(results):
    corrected = spell_correct_word(raw_text, candidates_per_pos=all_candidates)
else:
    corrected = spell_correct_word(raw_text)
if corrected != raw_text:
    print(f"📖 纠错后: {corrected}")
    raw_text = corrected

# === 翻译 ===
print(f"\n🌏 翻译中...")
try:
    params = {"client": "gtx", "sl": "en", "tl": "zh", "dt": "t", "q": raw_text}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params=params, headers=headers, timeout=10
    )
    if resp.ok:
        result = resp.json()
        translated = "".join(part[0] for part in result[0] if part[0])
        print(f"   ✅ {translated}")
    else:
        print(f"   ❌ HTTP {resp.status_code}")
except Exception as e:
    print(f"   ❌ {e}")

print("=" * 70)
