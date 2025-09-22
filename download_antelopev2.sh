#!/bin/bash
set -e

# (可選) 指定 InsightFace 快取目錄
export INSIGHTFACE_HOME=/home/inferno/.insightface

# 呼叫 Python 自動下載模型
python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='antelopev2')   # 可加 providers，例如 ['CUDAExecutionProvider','CPUExecutionProvider']
app.prepare(ctx_id=0)                   # 0 = 用 GPU，-1 = CPU
print("antelopev2 downloaded & ready")
PY

# 確認下載結果
ls -lah "${INSIGHTFACE_HOME:-$HOME/.insightface}/models/antelopev2"
