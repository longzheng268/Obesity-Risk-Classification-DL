$envs = @('ARPV','Cybersecurity','Ingram','XiaoYuanKouSuan','agv_gui','autoglm','campus_analysis','common','fc-clip','game','handwritten','imis-bench','intelligent_system','nlp','pose-estimation','r_heatmap','ros_data_process','star-office-ui','x-account','youtube')
foreach ($env in $envs) {
    Write-Host "--- $env ---"
    $python = "D:\env\miniconda3\envs\$env\python.exe"
    if (Test-Path $python) {
        $result = & $python -c "import torch; import sklearn; import pandas; import numpy; import matplotlib; import seaborn; print('ALL OK')" 2>&1
        Write-Host "  $result"
    } else {
        Write-Host "  Python not found"
    }
}
