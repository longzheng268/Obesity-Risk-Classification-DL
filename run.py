"""
肥胖等级分类 - 一键运行脚本
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    steps = [
        ("数据预处理与探索性分析", "src/preprocess.py"),
        ("模型训练与评估", "src/train.py"),
    ]
    for desc, script in steps:
        print(f"\n{'=' * 60}")
        print(f"  {desc}")
        print(f"{'=' * 60}\n")
        ret = subprocess.run([sys.executable, script])
        if ret.returncode != 0:
            print(f"\n[错误] {desc} 失败，终止运行。")
            return
    print(f"\n{'=' * 60}")
    print("  全部流程完成！输出文件在 output/ 目录下。")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
