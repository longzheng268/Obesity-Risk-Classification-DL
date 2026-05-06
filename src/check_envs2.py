import os, subprocess

envs = ['ARPV','Cybersecurity','Ingram','XiaoYuanKouSuan','agv_gui','autoglm','campus_analysis',
        'common','fc-clip','game','handwritten','imis-bench','intelligent_system','nlp',
        'pose-estimation','ros_data_process','star-office-ui','x-account','youtube']

base = r'D:\env\miniconda3\envs'
pkgs = ['torch','sklearn','pandas','numpy','matplotlib','seaborn','xgboost','lightgbm']

for env in envs:
    py = os.path.join(base, env, 'python.exe')
    if not os.path.exists(py):
        continue
    has = []
    for pkg in pkgs:
        try:
            r = subprocess.run([py, '-c', f'import {pkg}'], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                has.append(pkg)
        except:
            pass
    if len(has) >= 3:
        print(f"{env}: {has}")
