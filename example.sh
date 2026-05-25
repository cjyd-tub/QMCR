echo "=== 训练HGA数据集 ==="
python3 script/run.py -c /cjy/QMCR/config/HGA.yaml --gpus [1]
echo "=== 训练PWA数据集 ==="
python3 script/run.py -c /cjy/QMCR/config/PWA.yaml --gpus [1]