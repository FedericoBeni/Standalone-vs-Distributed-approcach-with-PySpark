# ==============================================================================
# train_distributed.py
#
# DESCRIZIONE:
# Questo script esegue l'addestramento e la valutazione di un modello LSTM 
# per la previsione dei prezzi delle azioni in un ambiente distribuito.
# Utilizza PySpark e TorchDistributor per parallelizzare l'addestramento su 
# più processi (CPU).
#
# FUNZIONAMENTO:
# 1. Inizializza una SparkSession.
# 2. Definisce una funzione di addestramento (`train_loop_fn`) che verrà 
#    eseguita su ogni worker.
# 3. Ogni worker:
#    a. Inizializza il gruppo di processi per la comunicazione distribuita.
#    b. Carica, preprocessa e partiziona i dati usando un DistributedSampler.
#    c. Incapsula il modello con DistributedDataParallel (DDP).
#    d. Esegue il ciclo di addestramento e validazione, sincronizzando i gradienti.
# 4. Il worker con rank 0 si occupa di salvare il modello, i risultati e i grafici.
# 5. TorchDistributor orchestra l'esecuzione della `train_loop_fn` sui worker.
# ==============================================================================

# SEZIONE 1: SETUP E IMPORT
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import json
from datetime import datetime
import math
import matplotlib.pyplot as plt

# Import per PySpark e TorchDistributor
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.ml.torch.distributor import TorchDistributor

# SEZIONE 2: CONFIGURAZIONE
class Config:
    # --- Percorsi di I/O ---
    DATA_DIR = "./"       # Directory sorgente per i dati di input.
    OUTPUT_DIR = "./"     # Directory di destinazione per i risultati (modelli, grafici, etc.).

    # --- Configurazione del Dataset ---
    STOCKS_FILE = "sp500_stocks.csv" # Nome del file contenente i dati azionari.
    
    # --- Azioni da analizzare ---
    STOCKS_TO_ANALYZE = ['AES', 'ALL', 'CCL', 'GIS']
    
    # --- Feature e Target ---
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Feature usate per la previsione.
    TARGET = 'Close' # Feature che il modello deve prevedere.
    SEQUENCE_LENGTH = 60  # Lunghezza della sequenza temporale usata come input.
    
    # --- Iperparametri di Addestramento ---
    TRAIN_SPLIT_RATIO = 0.8
    VALIDATION_SPLIT_RATIO = 0.1
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # --- Architettura del Modello LSTM ---
    INPUT_SIZE = len(FEATURES) # Corrisponde al numero di feature.
    HIDDEN_SIZE = 50           # Numero di neuroni nell'hidden layer LSTM.
    NUM_LAYERS = 2             # Numero di layer LSTM.
    OUTPUT_SIZE = 1

# SEZIONE 3: CLASSE DATASET
class StockDataset(Dataset):
    """
    Classe Dataset personalizzata per PyTorch.
    Permette di incapsulare i tensori di input (X) e target (y) e di accedervi 
    in modo efficiente tramite il DataLoader.
    """
    def __init__(self, X, y):
        """Inizializza il dataset con i dati di input e target."""
        self.X = X
        self.y = y
    
    def __len__(self):
        """Restituisce la dimensione totale del dataset."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Restituisce un singolo campione (sequenza e target) all'indice specificato."""
        return self.X[idx], self.y[idx]

# SEZIONE 4: MODELLO LSTM
class LSTMModel(nn.Module):
    """
    Architettura del modello LSTM.
    Composta da uno strato LSTM seguito da uno strato fully-connected (lineare) 
    per produrre l'output finale.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Strato LSTM: processa le sequenze di input.
        # batch_first=True indica che l'input ha dimensione (batch, seq_len, features).
        # dropout aggiunge regolarizzazione per prevenire l'overfitting.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Strato Lineare: mappa l'output dell'LSTM alla dimensione di output desiderata.
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Definisce il forward pass del modello."""
        # Inizializza gli stati nascosti (h0) e di cella (c0) a zero.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Passa l'input e gli stati iniziali attraverso lo strato LSTM.
        out, _ = self.lstm(x, (h0, c0))
        # Estrae solo l'ultimo timestep della sequenza.
        out = out[:, -1, :]

        out = self.fc(out)
        return out

# SEZIONE 5: METRICHE DI VALUTAZIONE
def calculate_metrics(y_true, y_pred):
    """
    Calcola un set di metriche di regressione per valutare le performance del modello.

    Args:
        y_true (np.array): Valori reali.
        y_pred (np.array): Valori predetti dal modello.

    Returns:
        dict: Un dizionario contenente MSE, RMSE, MAE e MAPE.
    """
    # Maschera per evitare la divisione per zero nel calcolo del MAPE.
    non_zero_mask = y_true != 0
    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]
    
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else 0
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# SEZIONE 6: FUNZIONE DI ADDESTRAMENTO DISTRIBUITO
def train_loop_fn(config_dict):
    """
    Funzione eseguita da ogni processo worker gestito da TorchDistributor.

    Questa funzione orchestra l'intero ciclo di vita dell'addestramento su un 
    singolo worker, includendo l'inizializzazione del processo, la preparazione
    dei dati, l'addestramento e la validazione del modello distribuito.
    """
    # --- Inizializzazione del processo distribuito ---
    # Ogni processo worker riceve il proprio RANK (identificativo univoco) e WORLD_SIZE (numero totale di processi)
    # da TorchDistributor attraverso le variabili d'ambiente.
    # - RANK: Identifica univocamente ogni worker (0, 1, 2, ...)
    # - WORLD_SIZE: Numero totale di processi worker avviati
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Imposta il device (GPU se disponibile, altrimenti CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # Inizializzazione del gruppo di processi per la comunicazione distribuita
    # - backend="gloo": Protocollo di comunicazione ottimizzato per CPU
    #   (usare "nccl" se si utilizzano GPU NVIDIA)
    # - rank: Identificativo univoco del processo corrente
    # - world_size: Numero totale di processi nel gruppo
    # Questo è un punto di sincronizzazione: tutti i processi devono raggiungere questa linea per proseguire
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"Worker {rank}/{world_size} - Using device: {device}")

    # --- CARICAMENTO E PREPROCESSING DATI (OGNI WORKER CARICA IL FILE) ---
    stocks_path = os.path.join(config.DATA_DIR, config.STOCKS_FILE) # per file system locale
	#stocks_path = "hdfs://user/user/input/sp500_stocks.csv" # per HDFS
    full_stocks_df = pd.read_csv(stocks_path)
    data = full_stocks_df[full_stocks_df['Symbol'].isin(config.STOCKS_TO_ANALYZE)].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Symbol', 'Date'])
    data.set_index('Date', inplace=True)
    data = data[config.FEATURES + ['Symbol']].dropna()

    # --- SPLIT DEI DATI ---
    unique_dates = data.index.unique().sort_values()
    train_end_idx = int(len(unique_dates) * config.TRAIN_SPLIT_RATIO)
    val_end_idx = train_end_idx + int(len(unique_dates) * config.VALIDATION_SPLIT_RATIO)
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]

    train_df = data[data.index <= train_end_date]
    val_df = data[(data.index > train_end_date) & (data.index <= val_end_date)]
    test_df = data[data.index > val_end_date]

    # --- SCALING DELLE FEATURE ---
    # Lo scaling è cruciale per le reti neurali: normalizza i range delle diverse 
    # feature, aiutando il modello a convergere più velocemente e stabilmente.    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[config.FEATURES])

    train_df_scaled = pd.DataFrame(scaler.transform(train_df[config.FEATURES]), columns=config.FEATURES, index=train_df.index)
    train_df_scaled['Symbol'] = train_df['Symbol']
    val_df_scaled = pd.DataFrame(scaler.transform(val_df[config.FEATURES]), columns=config.FEATURES, index=val_df.index)
    val_df_scaled['Symbol'] = val_df['Symbol']
    test_df_scaled = pd.DataFrame(scaler.transform(test_df[config.FEATURES]), columns=config.FEATURES, index=test_df.index)
    test_df_scaled['Symbol'] = test_df['Symbol']

    def create_sequences_from_scaled(df, seq_length, target_idx):
        all_x, all_y = [], []
        for _, group in df.groupby('Symbol'):
            group_values = group[config.FEATURES].values
            if len(group_values) > seq_length:
                for i in range(len(group_values) - seq_length):
                    all_x.append(group_values[i:(i + seq_length)])
                    all_y.append(group_values[i + seq_length, target_idx])
        return np.array(all_x), np.array(all_y).reshape(-1, 1)

    # --- CREAZIONE DELLE SEQUENZE TEMPORALI ---
    target_idx = config.FEATURES.index(config.TARGET)
    X_train, y_train = create_sequences_from_scaled(train_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    X_val, y_val = create_sequences_from_scaled(val_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    X_test, y_test = create_sequences_from_scaled(test_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    # --- CREAZIONE DEI DATASET E LOADER ---
    train_dataset = StockDataset(X_train, y_train)
	# DistributedSampler partiziona i dati in modo che ogni worker riceva un sottoinsieme diverso
    # - num_replicas: Numero totale di processi (deve corrispondere a world_size)
    # - rank: Identifica quale partizione dei dati deve essere gestita da questo worker
    # Ogni epoca, i dati vengono ridistribuiti per garantire che il modello veda tutti i dati
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
	 # Il DataLoader utilizza il DistributedSampler per caricare solo la partizione di dati assegnata a questo worker
    # shuffle=False perché lo shuffling è gestito dal DistributedSampler
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
    
    val_dataset = StockDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- CREAZIONE DEL MODELLO E INIZIALIZZAZIONE ---
    model = LSTMModel(
        input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE
    ).to(device)

	# Incapsula il modello con DistributedDataParallel (DDP) per l'addestramento distribuito
    # DDP si occupa automaticamente di:
    # 1. Sincronizzare i gradienti tra i processi durante la backward pass
    # 2. Ridistribuire il modello aggiornato a tutti i processi
    # device_ids=None perché stiamo usando la CPU (per GPU, specificare gli ID delle GPU)
    model = nn.parallel.DistributedDataParallel(model, device_ids=None)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- ADDESTRAMENTO DEL MODELLO ---
    if rank == 0:
        print(f"Inizio addestramento su {world_size} processi...")
    
    worker_start_time = time.time()
    stopped_epoch = config.NUM_EPOCHS
    train_losses, val_losses = [], []

    for epoch in range(config.NUM_EPOCHS):
	    # Imposta l'epoca corrente per il DistributedSampler
        # Questo garantisce che i dati vengano mescolati in modo diverso ad ogni epoca
        # e che ogni worker riceva una diversa permutazione dei dati
        # È fondamentale per evitare che tutti i processi elaborino gli stessi batch in ogni epoca
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward() # DDP calcola e sincronizza i gradienti
            optimizer.step()
            train_loss += loss.item()

        # --- Sincronizzazione e Validazione (solo su rank 0) ---
		# Calcola la loss media su tutti i batch di questo worker
        avg_train_loss_tensor = torch.tensor(train_loss / len(train_loader)).to(device)

	    # all_reduce somma il valore del tensore su tutti i processi
        # ReduceOp.SUM: somma i valori da tutti i processi
        # Ogni worker invia la sua loss media e riceve la somma di tutte le loss
        torch.distributed.all_reduce(avg_train_loss_tensor, op=torch.distributed.ReduceOp.SUM)

		#Calcola la los media su tutti i processi
        avg_train_loss = avg_train_loss_tensor.item() / world_size

        stop_training = False
        if rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # print(f'Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            # Log ogni 10 epoche (più la prima)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}] - "
	            f"Train Loss: {avg_train_loss:.6f} - "
	            f"Val Loss: {avg_val_loss:.6f}")            

        # --- Sincronizzazione tra processi ---
		# Crea un tensore per comunicare se l'addestramento deve essere interrotto
        # Viene inviato dal rank 0 a tutti gli altri processi
        stop_training_tensor = torch.tensor(int(stop_training), dtype=torch.int).to(device)
	    # Broadcast del segnale di stop a tutti i processi
        # Questo garantisce che tutti i processi si fermino contemporaneamente
        torch.distributed.broadcast(stop_training_tensor, src=0)

		# Se il segnale di stop è attivo, interrompi il ciclo di addestramento
        # Questo è importante per sincronizzare l'arresto su tutti i processi
        if bool(stop_training_tensor.item()):
            break

    worker_training_time = time.time() - worker_start_time
    print(f"Worker {rank}/{world_size} - Addestramento completato in {worker_training_time:.2f} secondi")

    # --- Il worker con rank 0 salva i risultati finali ---
    if rank == 0:
        print("\nInizio valutazione sul test set aggregato (Rank 0)...")
        model.eval()
        predictions_scaled = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                # Accede al modello originale
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions_scaled.append(outputs.cpu())
        predictions_scaled = torch.cat(predictions_scaled).numpy()

        # Inversione dello scaling per riportare le predizioni e i valori reali alla scala originale.
        # Creazione di array fittizi con la stessa struttura delle feature originali
        # Questo è necessario perché lo scaler si aspetta lo stesso numero di colonne delle feature originali
        dummy_array_preds = np.zeros((len(predictions_scaled), len(config.FEATURES)))
		# Inserimento delle predizioni scalate nella colonna corretta (quella del target)
        dummy_array_preds[:, target_idx] = predictions_scaled.flatten()
		# Applicazione della trasformazione inversa e estrazione della colonna di interesse
        predictions_actual = scaler.inverse_transform(dummy_array_preds)[:, target_idx]

	    # Stesso procedimento per i valori reali del test set
        dummy_array_y = np.zeros((len(y_test), len(config.FEATURES)))
        dummy_array_y[:, target_idx] = y_test.numpy().flatten()
        y_test_actual = scaler.inverse_transform(dummy_array_y)[:, target_idx]

        # Calcolo delle metriche finali sul test set
		# Le metriche vengono calcolate solo sul rank 0 per evitare duplicazioni
        metrics = calculate_metrics(y_test_actual, predictions_actual)
        print("\nMetriche di Performance sul Test Set:")
		# Stampa formattata delle metriche di valutazione
        # Le metriche includono MSE, RMSE, MAE e MAPE
        for key, value in metrics.items(): print(f"- {key}: {value:.4f}")
		 # Nota: in un contesto distribuito, queste metriche si riferiscono solo al test set
         # elaborato dal rank 0, che è sufficiente dato che il modello è stato addestrato
         # su tutti i dati attraverso la distribuzione del training


        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        model_name_tag = f"Distributed_MultiStock_{world_size}proc"
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        MODEL_PATH = os.path.join(config.OUTPUT_DIR, f"modello_{model_name_tag}_{timestamp}.pth")
        torch.save(model.module.state_dict(), MODEL_PATH)
        print(f"\nModello salvato in: {MODEL_PATH}")

        results_summary = {
            'run_type': 'distributed', 'num_processes': world_size,
            'stocks_analyzed': config.STOCKS_TO_ANALYZE,
            'total_training_samples': len(X_train),
            'training_time_seconds': round(worker_training_time, 2),
            'stopped_at_epoch': stopped_epoch,
            'aggregate_performance_metrics': {k: round(v, 4) for k, v in metrics.items()},
        }
        RESULTS_PATH = os.path.join(config.OUTPUT_DIR, f"risultati_{model_name_tag}_{timestamp}.json")
        with open(RESULTS_PATH, 'w') as f: json.dump(results_summary, f, indent=4)
        print(f"Riepilogo risultati salvato in: {RESULTS_PATH}")

        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Curve di Apprendimento ({world_size} processi)')
        plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
        PLOT_PATH_LOSS = os.path.join(config.OUTPUT_DIR, f"grafico_loss_{model_name_tag}_{timestamp}.png")
        plt.savefig(PLOT_PATH_LOSS)
        print(f"Grafico Loss salvato in: {PLOT_PATH_LOSS}")
        plt.close()

    # Rilascia le risorse del gruppo di processi.
	# Importante per pulire le risorse di comunicazione e prevenire memory leak
    # Deve essere chiamato da tutti i processi prima della terminazione
    torch.distributed.destroy_process_group()

# SEZIONE 7: ESECUZIONE TRAMITE SPARK-SUBMIT
def main_distributed():
    config = Config() # Crea un'istanza della configurazione

    print(f"Inizio esecuzione distribuita per {len(config.STOCKS_TO_ANALYZE)} azioni con {Config.NUM_PROCESSES} processi...")

    # Inizializza SparkSession. 
    # master="local[*]" usa tutti i core disponibili.
    # local[NUM_PROCESSES] per specificare i core.
    # spark.executor.cores=1 è importante per la modalità local[N] per PySpark
    spark = SparkSession.builder \
        .appName("DistributedLSTMStockPrediction") \
        .master(f"local[{Config.NUM_PROCESSES}]") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print("SparkSession inizializzata.")

    # Inizializza TorchDistributor
    distributor = TorchDistributor(
        num_processes=Config.NUM_PROCESSES,
        use_gpu=False, # Imposta a False perché non hai GPU
        local_mode=True # Imposta a True per l'esecuzione su una singola VM
    )

    # Lancia la funzione di addestramento distribuita
    results = distributor.run(train_loop_fn, config.__dict__)
    
    spark.stop()
    print("SparkSession terminata. Esecuzione distribuita completata.")

# ESECUZIONE DELLO SCRIPT TRAMITE SPARK-SUBMIT

# --- PASSO 1: Inizializza la SparkSession ---
spark = SparkSession.builder.appName("DistributedLSTMStockPrediction").getOrCreate()

print("SparkSession recuperata o creata con successo.")
print(f"Versione di Spark: {spark.version}")

# Leggi il numero di processi da una variabile d'ambiente o usa un default.
num_processes = int(os.getenv("NUM_PROCESSES", "2")) 

config = Config()

# --- PASSO 2: Inizializza TorchDistributor ---
distributor = TorchDistributor(
    num_processes=num_processes, 
    local_mode=True, 
    use_gpu=False
)

# --- PASSO 3: Lancia l'addestramento ---
distributor.run(train_loop_fn, config.__dict__)

# --- PASSO 4: Ferma la SparkSession ---
spark.stop()

print(f"\nEsecuzione distribuita con {num_processes} processi completata.")
