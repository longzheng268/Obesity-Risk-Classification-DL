"""
通用环境检查脚本 - 跨平台支持 (Windows / Linux / macOS)
检查当前 conda 环境是否安装了项目所需的依赖包

使用方法:
    conda activate obesity-risk
    python src/check_envs.py
"""
import sys
import platform
import subprocess
import shutil

REQUIRED = ['torch', 'torchvision', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
OPTIONAL = ['xgboost', 'lightgbm']


def check_pkg(name):
    try:
        r = subprocess.run(
            [sys.executable, '-c', f'import {name}; print({name}.__version__)'],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def main():
    print('=' * 50)
    print('  环境检查')
    print('=' * 50)
    print(f'  Python : {sys.version.split()[0]}')
    print(f'  平台   : {platform.system()} {platform.release()}')
    print(f'  解释器 : {sys.executable}')
    print('=' * 50)

    # 检查 conda
    conda = shutil.which('conda')
    if conda:
        print(f'\n  conda  : {conda}')
    else:
        print('\n  conda  : 未找到 (可能未加入 PATH)')

    # 检查 GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f'  GPU    : {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})')
        else:
            print('  GPU    : 不可用，将使用 CPU')
    except ImportError:
        print('  GPU    : 未安装 torch，无法检测')

    # 必需包
    print(f'\n{"=" * 50}')
    print('  必需依赖')
    print(f'{"=" * 50}')
    all_ok = True
    for pkg in REQUIRED:
        ver = check_pkg(pkg)
        status = f'OK ({ver})' if ver else 'MISSING'
        symbol = '  [OK]' if ver else '  [!!]'
        print(f'{symbol} {pkg:<15} {status}')
        if not ver:
            all_ok = False

    # 可选包
    print(f'\n{"=" * 50}')
    print('  可选依赖')
    print(f'{"=" * 50}')
    for pkg in OPTIONAL:
        ver = check_pkg(pkg)
        status = f'OK ({ver})' if ver else '未安装'
        symbol = '  [OK]' if ver else '  [--]'
        print(f'{symbol} {pkg:<15} {status}')

    print(f'\n{"=" * 50}')
    if all_ok:
        print('  所有必需依赖均已安装，可以运行项目！')
    else:
        print('  存在缺失的必需依赖，请先安装。')
        print('  pip install -r requirements.txt')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
