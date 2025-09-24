# Standalone vs Distributed approach in PySpark and PyTorch

This repository contains the source code and results for a project focused on the performance analysis of distributed deep learning in a multi-core CPU environment.

## Objective
The main goal is to compare the training time and accuracy of a Recurrent Neural Network (LSTM) in two distinct configurations:

1.  **Standalone:** A standard PyTorch implementation running on a single process (serving as the baseline).
2.  **Distributed:** A parallelized implementation across multiple CPU cores using PySpark's `TorchDistributor`.

The project aims to quantify the performance gain (speedup) and evaluate the impact of distribution on the model's accuracy.

##  Tech Stack
- **Python 3.9**
- **Apache Spark (PySpark):** For data parallelization and training orchestration.
- **PyTorch:** For defining and training the LSTM model.
- **Apache Hadoop (HDFS):** Used for tests involving distributed storage.
- **Pandas & NumPy:** For data manipulation and preparation.
- **Scikit-learn:** For evaluating performance metrics.

## Dataset
The model was trained and tested using the **S&P 500 Stock Data** dataset, which contains historical stock information. It is available for download on [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500).

## Key Results: CPU Scalability
The following tables present the results obtained by comparing the execution on 1, 2, 3, and 4 CPU cores.

#### Table 1: Average Training Time and Speedup Comparison

| # Processes (Cores) | Average Time (s) | Std. Dev. (s) | Speedup (vs 1 Core) |
| :------------------ | :--------------: | :-----------: | :-------------------: |
| 1 (Standalone)      |     2053.34      |   ± 89.55     |    1.00x (Baseline)   |
| 2 (Distributed)     |     1050.34      |   ± 55.60     |         1.95x         |
| 3 (Distributed)     |      894.32      |   ± 43.46     |         2.29x         |
| 4 (Distributed)     |      789.16      |   ± 61.12     |         2.60x         |

**Analysis:** Using PySpark with TorchDistributor led to a significant reduction in training times. Scaling from 1 to 4 cores resulted in a **2.60x speedup**, demonstrating the effectiveness of the distributed approach even on a single multi-core machine.

#### Table 2: Average Accuracy Metrics Comparison

| # Processes (Cores) | Avg. MAPE (%) | MAPE Std. Dev. (%) | Avg. RMSE |
| :------------------ | :-----------: | :----------------: | :-------: |
| 1 (Standalone)      |     3.54%     |       0.44%        |   8.58    |
| 2 (Distributed)     |     2.87%     |       0.36%        |   5.68    |
| 3 (Distributed)     |     3.52%     |       0.41%        |   5.58    |
| 4 (Distributed)     |     3.23%     |       0.52%        |   5.98    |

**Analysis:** The acceleration achieved did not compromise model quality. The accuracy metrics remained statistically equivalent across all tests, with minor fluctuations attributed to the stochastic nature of the training process.

### Bonus: Test on Distributed Storage (HDFS)
An additional test was conducted to evaluate the impact of reading data from HDFS versus the local file system, simulating a more realistic Big Data scenario.

| Data Source (4 Cores) | Average Time (s) | Difference vs. Local |
| :-------------------- | :--------------: | :-------------------: |
| Local File System     |      789.16      |       Baseline        |
| HDFS (Warm Avg)       |      675.12      |   -14.45% (faster)      |

**Note:** Reading from HDFS was on average **14.45% faster**.  
This result — likely due to the optimized caching mechanisms of Hadoop/Java — confirms both the full compatibility and the efficiency of the solution in a true end-to-end Big Data environment.

**Conclusions**  
The integration of PySpark and PyTorch through TorchDistributor has proven to be a solid and effective solution for accelerating deep learning model training on multi-core architectures. The results highlight not only a significant improvement in computational performance, but also the preservation of model quality, paving the way for further exploration on multi-node clusters.

