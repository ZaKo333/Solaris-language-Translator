"""
一键清理 char_db，只保留 _game_ 命名的真实游戏字符
用法: python clean_db.py
"""
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_db")
if not os.path.exists(BASE):
    print("❌ char_db 不存在")
    exit(1)

files = os.listdir(BASE)
keep = [f for f in files if '_game_' in f]
delete = [f for f in files if '_game_' not in f and not f.endswith('.gitkeep')]

print(f"📁 char_db 中共 {len(files)} 个文件")
print(f"✅ 保留: {len(keep)} 个 (从游戏提取的)")
print(f"🗑️  删除: {len(delete)} 个 (旧的不匹配的)")

if delete:
    print("\n即将删除:")
    for f in sorted(delete):
        print(f"  - {f}")
    ans = input("\n确认删除? (y/n): ")
    if ans.lower() == 'y':
        for f in delete:
            os.remove(os.path.join(BASE, f))
        print(f"✅ 已删除 {len(delete)} 个文件")
    else:
        print("已取消")
else:
    print("没有需要删除的文件")

print(f"\n📊 当前 char_db: {len(os.listdir(BASE))} 个文件")
