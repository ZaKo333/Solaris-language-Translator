"""诊断工具 - 检查数据库加载问题"""
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "char_db")
MAP_PATH = os.path.join(BASE_DIR, "char_mapping.py")

print("=" * 50)
print("🔍 Sora 翻译器 - 诊断工具")
print("=" * 50)

# 1. 检查目录是否存在
print(f"\n📁 项目目录: {BASE_DIR}")
print(f"📁 char_db路径: {DB_PATH}")
print(f"  存在? {'✅' if os.path.exists(DB_PATH) else '❌'}")

if not os.path.exists(DB_PATH):
    print("  → char_db 目录不存在！请手动建一个空文件夹")
    sys.exit(1)

# 2. 列出所有文件
files = os.listdir(DB_PATH)
print(f"\n📄 char_db 中的文件 ({len(files)}个):")
for f in sorted(files):
    fpath = os.path.join(DB_PATH, f)
    size = os.path.getsize(fpath)
    ext = os.path.splitext(f)[1].lower()
    key = os.path.splitext(f)[0]
    is_img = ext in ('.png', '.jpg', '.jpeg', '.bmp')
    print(f"   {f:20s}  {size:>6}B  后缀:{ext:5s}  键值:{key:10s}  {'🖼' if is_img else '⚠ 非图片'}")

img_files = [f for f in files if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
print(f"\n🖼 图片文件: {len(img_files)}/{len(files)}")
print(f"⚠ 非图片文件: {len(files) - len(img_files)}/{len(files)}")

# 3. 检查映射
print(f"\n📋 char_mapping.py 是否存在: {'✅' if os.path.exists(MAP_PATH) else '❌'}")
try:
    sys.path.insert(0, BASE_DIR)
    import char_mapping
    cm = char_mapping.get_char_map()
    print(f"   映射表条目数: {len(cm)}")
    print(f"   映射的英文字符: {''.join(sorted(set(cm.values())))}")

    # 4. 交叉检查
    print(f"\n🔗 交叉检查:")
    ok = 0
    missing_img = []
    for k, v in cm.items():
        found = False
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            if (k + ext) in files or k in files:
                found = True
                break
        if found:
            ok += 1
        else:
            missing_img.append((k, v))

    print(f"   映射有对应图片: {ok}/{len(cm)}")
    if missing_img:
        print(f"   以下映射缺图片:")
        for k, v in missing_img:
            print(f"     - '{k}' → '{v}'  的文件不存在于 char_db/")

    extra = []
    for f in img_files:
        k = os.path.splitext(f)[0]
        if k not in cm:
            extra.append(f)
    if extra:
        print(f"\n   以下图片未在映射表中:")
        for f in extra:
            print(f"     - {f}")

except Exception as e:
    print(f"   ❌ 导入映射表失败: {e}")

# 5. 尝试直接读一张图片
print(f"\n📸 尝试读取第一张图片...")
if img_files:
    try:
        from PIL import Image
        import cv2
        import numpy as np

        first = img_files[0]
        path = os.path.join(DB_PATH, first)

        # 用 PIL 读取（支持中文路径）
        pil_img = Image.open(path).convert('L')
        img = np.array(pil_img, dtype=np.uint8)
        h, w = img.shape
        print(f"   ✅ 成功! {first} = {w}x{h} 像素")
        print(f"   (使用 PIL 读取，解决中文路径问题)")

        # 尝试读取所有文件
        print(f"\n📸 尝试读取所有图片...")
        success = 0
        fail = []
        for f in img_files:
            try:
                path = os.path.join(DB_PATH, f)
                pil_img = Image.open(path).convert('L')
                img = np.array(pil_img, dtype=np.uint8)
                if img is not None and img.size > 0:
                    success += 1
                else:
                    fail.append(f)
            except:
                fail.append(f)
        print(f"   成功: {success}/{len(img_files)}")
        if fail:
            print(f"   失败:")
            for f in fail:
                print(f"     - {f}")

    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 读取图片时出错: {e}")
else:
    print(f"   ⚠ 没有图片文件可测试")

print("\n" + "=" * 50)
print("诊断完成")
print("=" * 50)
