"""深度诊断 - 检查数据库加载和匹配全过程"""
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "char_db")

print("=" * 60)
print("🔍 Sora 翻译器 - 深度诊断")
print("=" * 60)

sys.path.insert(0, BASE_DIR)
import char_mapping

# 1. 加载映射
cm = char_mapping.get_char_map()
print(f"\n📋 映射表: {len(cm)} 个条目")

# 2. 用 PIL 读取每张图片，测试 ORB 特征提取
from PIL import Image
import cv2
import numpy as np

orb = cv2.ORB_create(nfeatures=100)
images_loaded = 0
orb_ok = 0
orb_fail = []
pixel_info = []

files = [f for f in os.listdir(DB_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

print(f"\n📸 逐一检查 {len(files)} 张图片:")
print("-" * 70)
print(f"{'文件名':20s} {'大小':>6s} {'像素':>10s} {'ORB特征':>8s} {'状态'}")
print("-" * 70)

for fname in sorted(files):
    key = os.path.splitext(fname)[0]
    path = os.path.join(DB_PATH, fname)
    size = os.path.getsize(path)

    try:
        pil_img = Image.open(path).convert('L')
        img = np.array(pil_img, dtype=np.uint8)
        h, w = img.shape

        if key not in cm:
            print(f"{fname:20s} {size:>6}B {w}x{h:>6s} {'-':>8s} ⚠ 未在映射中")
            continue

        images_loaded += 1
        kp, des = orb.detectAndCompute(img, None)
        if des is not None and len(kp) >= 3:
            orb_ok += 1
            status = "✅"
        else:
            orb_fail.append((fname, key))
            status = "⚠ 无特征点"
        print(f"{fname:20s} {size:>6}B {w}x{h:>6s} {len(kp) if kp is not None else 0:>4d}个  {status}")

    except Exception as e:
        print(f"{fname:20s} {size:>6}B {'-':>10s} {'-':>8s} ❌ {str(e)[:30]}")

print("-" * 70)

# 3. 总结
print(f"\n📊 总结:")
print(f"   映射表条目: {len(cm)}")
print(f"   有对应图片: {images_loaded}")
print(f"   ORB特征正常: {orb_ok}/{images_loaded}")

if orb_fail:
    print(f"\n⚠ 以下 {len(orb_fail)} 张图片无ORB特征点(可能太小或太简单):")
    for fname, key in orb_fail:
        mapped_to = cm.get(key, '?')
        path = os.path.join(DB_PATH, fname)
        pil_img = Image.open(path).convert('L')
        img = np.array(pil_img)
        print(f"   {fname} → '{mapped_to}'  ({img.shape[1]}x{img.shape[0]}像素)")

print(f"\n💡 ORB 失败的原因: 图片太小(<20x20)或内容太简单(纯黑白块)")
print(f"   如果全是 '无特征点', 说明标准图本身质量有问题")

# 4. 测试自检：拿 char_db 里的图自己匹配自己
print(f"\n🔄 自检测试: 用标准图匹配自身 (理想情况应全部匹配)")
print("-" * 70)

# 加载成功加载的图
matcher_success = 0
matcher_fail = []
for fname in sorted(files):
    key = os.path.splitext(fname)[0]
    if key not in cm:
        continue
    path = os.path.join(DB_PATH, fname)
    try:
        pil_img = Image.open(path).convert('L')
    except:
        continue
    matcher_success += 1

if matcher_success > 5:
    # 只挑几个测试
    from matcher import GameCharMatcher
    matcher_test = GameCharMatcher(db_path=DB_PATH)
    matcher_test.load_database()
    print(f"\n   实际数据库加载: {len(matcher_test.std_images)}/{len(cm)} 个")
    print(f"   被跳过的: {len(cm) - len(matcher_test.std_images)} 个")

    if len(matcher_test.std_images) > 0:
        print(f"\n   自检匹配结果:")
        match_ok = 0
        match_fail = []
        for key, label in cm.items():
            if key not in matcher_test.std_images:
                continue
            path = os.path.join(DB_PATH, fname)
            try:
                pil_img = Image.open(path).convert('L')
                img = np.array(pil_img)
                # 用标准图匹配自己
                char_result, conf, details = matcher_test.match_single_char(img, threshold=0.0)
                if char_result == label:
                    match_ok += 1
                else:
                    match_fail.append((key, label, char_result, conf))
            except:
                pass
        print(f"   匹配正确: {match_ok} 个")
        if match_fail:
            print(f"   匹配错误:")
            for k, lbl, got, c in match_fail[:10]:
                print(f"     {k} → 期望'{lbl}' 得到'{got}' (置信度:{c:.3f})")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
