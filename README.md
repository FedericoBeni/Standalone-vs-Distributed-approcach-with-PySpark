# Standalone vs Distributed Training with PySpark and PyTorch

Performance analysis of distributed deep learning on multi-core CPU using PySpark (TorchDistributor) and PyTorch.

##  Key Results

- Speedup up to **2.60x** using 4 CPU cores  
- No loss in accuracy (MAPE 3.23% vs 3.54% baseline)  
- Additional **+14.45% I/O performance** using HDFS  

## Objective

Compare training performance of an LSTM model in two configurations:

- **Standalone** → single-process PyTorch (baseline)  
- **Distributed** → PySpark + TorchDistributor (multi-core CPU)  

## Tech Stack

- Python, PyTorch  
- Apache Spark (PySpark)  
- HDFS  
- Pandas, NumPy, Scikit-learn  

## Results

### CPU Scalability

| Cores | Time (s) | Speedup |
|------|--------|--------|
| 1 | 2053.34 | 1.00x |
| 2 | 1050.34 | 1.95x |
| 3 | 894.32 | 2.29x |
| 4 | 789.16 | **2.60x** |

✔ Distributed training significantly reduces training time

---

### Model Accuracy

| Cores | MAPE (%) | RMSE |
|------|---------|------|
| 1 | 3.54 | 8.58 |
| 4 | 3.23 | 5.98 |

✔ Accuracy remains statistically equivalent across configurations

---

### HDFS Benchmark

| Storage | Time (s) | Improvement |
|--------|---------|------------|
| Local | 789.16 | — |
| HDFS | 675.12 | **+14.45%** |

✔ Distributed storage improves I/O performance

## 📂 Dataset

- S\&P 500 historical stock dataset (Kaggle)
