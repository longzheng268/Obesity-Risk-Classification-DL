import sys, os, subprocess

envs = ['ARPV','Cybersecurity','Ingram','XiaoYuanKouSuan','agv_gui','autoglm','campus_analysis',
        'common','fc-clip','game','handwritten','imis-bench','intelligent_system','nlp',
        'pose-estimation','ros_data_process','star-office-ui','x-account','youtube']

base = r'D:\env\miniconda3\envs'

for env in envs:
    py = os.path.join(base, env, 'python.exe')
    if not os.path.exists(py):
        print(f"{env}: python not found")
        continue
    try:
        r = subprocess.run([py, '-c', 'import torch; import sklearn; import pandas; import numpy; import matplotlib; import seaborn; print("ALL_OK")'],
                           capture_output=True, text=True, timeout=30)
        output = r.stdout.strip() + r.stderr.strip()
        if 'ALL_OK' in output:
            print(f"{env}: ALL OK")
        else:
            # check individually
            pkgs = []
            for pkg in ['torch','sklearn','pandas','numpy','matplotlib','seaborn']:
                r2 = subprocess.run([py, '-c', f'import {pkg}'], capture_output=True, text=True, timeout=10)
                if r2.returncode == 0:
                    pkgs.append(pkg)
            print(f"{env}: has {pkgs}")
    except Exception as e:
        print(f"{env}: error - {e}")
