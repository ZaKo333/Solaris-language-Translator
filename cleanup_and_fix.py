"""
清理字体渲染的垃圾模板，保留原始字体图和游戏提取图
"""
import os, glob

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_db")

# 删除 _font_ 命名的模板
deleted = 0
for f in glob.glob(os.path.join(db_path, "*_font_*")):
    os.remove(f)
    deleted += 1

print(f"已删除 {deleted} 个字体渲染模板")

# 列出保留的模板
remaining = [f for f in os.listdir(db_path) if f.lower().endswith('.png')]
print(f"剩余 {len(remaining)} 个模板:")
for f in sorted(remaining):
    print(f"  {f}")
