from flask import Flask, render_template
import pynvml
import psutil
import json
from threading import Thread
import time
import os

app = Flask(__name__)

# Initialize NVIDIA management library
pynvml.nvmlInit()

# Store training metrics (updated by training script)
metrics_file = "training_metrics.json"
if not os.path.exists(metrics_file):
    with open(metrics_file, 'w') as f:
        json.dump({"epoch": 0, "train_loss": 0, "val_loss": 0, "train_acc": 0, "val_acc": 0}, f)

def get_gpu_stats():
    try:
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(device)
        temp = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(device) / 1000.0  # mW to W
        return {
            "gpu_util": utilization.gpu,
            "mem_used": mem_info.used / 1024**2,  # Bytes to MiB
            "mem_total": mem_info.total / 1024**2,
            "temp": temp,
            "power": power
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def dashboard():
    gpu_stats = get_gpu_stats()
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return render_template('dashboard.html', gpu_stats=gpu_stats, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)