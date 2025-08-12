# =============================================================================
# train_standalone.py
#
# DESCRIZIONE:
# Questo script esegue l'addestramento e la valutazione di un modello LSTM 
# per la previsione dei prezzi di chiusura delle azioni. L'intero processo 
# viene eseguito in un ambiente a processo singolo (standalone).
#
# FUNZIONAMENTO:
# 1. Carica i dati storici delle azioni da un file CSV.
# 2. Preprocessa i dati: filtraggio per ticker, scaling delle feature.
# 3. Trasforma i dati in sequenze temporali per l'addestramento del modello.
# 4. Suddivide i dati in set di training, validazione e test.
# 5. Definisce e addestra un modello LSTM per un numero fisso di epoche.
# 6. Valuta le performance del modello sul set di test.
# 7. Salva il modello addestrato, un riepilogo dei risultati e un grafico 
#    delle curve di apprendimento.
# ==============================================================================

# SEZIONE 1: SETUP E IMPORT
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
import math

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

# FUNZIONE PRINCIPALE
def main():
    """Funzione principale che orchestra l'intero processo."""
    config = Config()
    
    # Imposta il device (GPU se disponibile, altrimenti CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Analisi su {len(config.STOCKS_TO_ANALYZE)} azioni: {config.STOCKS_TO_ANALYZE}")

    # --- 1. CARICAMENTO E PREPROCESSING DEI DATI ---
    print(f"\nCaricamento del dataset '{config.STOCKS_FILE}'...")
    stocks_path = os.path.join(config.DATA_DIR, config.STOCKS_FILE)
    try:
        full_stocks_df = pd.read_csv(stocks_path)
        print("Dataset caricato con successo.")
    except FileNotFoundError:
        print(f"ERRORE: File non trovato in '{stocks_path}'.")
        return

    # Filtra i dati per mantenere solo i ticker di interesse.
    print(f"Filtraggio per tickers: {config.STOCKS_TO_ANALYZE}")
    data = full_stocks_df[full_stocks_df['Symbol'].isin(config.STOCKS_TO_ANALYZE)].copy()

    # Converte la colonna 'Date' e ordina i dati per garantire la coerenza temporale.
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Symbol', 'Date'])
    data.set_index('Date', inplace=True)
    data = data[config.FEATURES + ['Symbol']].dropna()

    # --- 1. SUDDIVISIONE DEL DATASET ---
    # Suddivide il dataset in train, validation e test.
    unique_dates = data.index.unique().sort_values()
    train_end_idx = int(len(unique_dates) * config.TRAIN_SPLIT_RATIO)
    val_end_idx = train_end_idx + int(len(unique_dates) * config.VALIDATION_SPLIT_RATIO)
    # Estrae le date di fine train e validation.
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]

    # Estrae i dati di train, validation e test.
    train_df = data[data.index <= train_end_date]
    val_df = data[(data.index > train_end_date) & (data.index <= val_end_date)]
    test_df = data[data.index > val_end_date]

    # --- 2. SCALING DELLE FEATURE ---
    # Lo scaling è cruciale per le reti neurali: normalizza i range delle diverse 
    # feature, aiutando il modello a convergere più velocemente e stabilmente.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[config.FEATURES])
    
    # Funzione per applicare lo scaling a un DataFrame.
    def scale_df(df):
        scaled = pd.DataFrame(scaler.transform(df[config.FEATURES]), columns=config.FEATURES, index=df.index)
        scaled['Symbol'] = df['Symbol']
        return scaled

    train_df_scaled = scale_df(train_df)
    val_df_scaled = scale_df(val_df)
    test_df_scaled = scale_df(test_df)

    # --- 3. CREAZIONE DELLE SEQUENZE TEMPORALI ---
    # Funzione per creare le sequenze temporali a partire di un DataFrame.
    def create_sequences_from_scaled(df, seq_length, target_idx):
        all_x, all_y = [], []
        for _, group in df.groupby('Symbol'):
            group_values = group[config.FEATURES].values
            if len(group_values) > seq_length:
                for i in range(len(group_values) - seq_length):
                    all_x.append(group_values[i:(i + seq_length)])
                    all_y.append(group_values[i + seq_length, target_idx])
        return np.array(all_x), np.array(all_y).reshape(-1, 1)

    # Estrae le sequenze di train, validation e test.
    target_idx = config.FEATURES.index(config.TARGET)
    X_train, y_train = create_sequences_from_scaled(train_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    X_val, y_val = create_sequences_from_scaled(val_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    X_test, y_test = create_sequences_from_scaled(test_df_scaled, config.SEQUENCE_LENGTH, target_idx)

    # --- 4. CONVERSIONE IN TENSORI ---
    # Converte le sequenze in tensori PyTorch.
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    # --- 5. CREAZIONE DEI DATALOADER ---
    # Crea i DataLoader per train, validation e test.
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 6. INIZIALIZZAZIONE MODELLO, CRITERION E OTTIMIZZATORE ---
    # Inizializza il modello, il criterio e l'ottimizzatore.
    model = LSTMModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE
    ).to(device)

    print(f"\nInfo modello: {sum(p.numel() for p in model.parameters()):,} parametri totali.")
    
    criterion = nn.MSELoss() # Funzione di costo: Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- 7. ADDESTRAMENTO DEL MODELLO ---
    print("\nInizio addestramento...")
    start_time = time.time()
    train_losses, val_losses = [], []

    for epoch in range(config.NUM_EPOCHS):
        # Fase di Training
        model.train() # Imposta il modello in modalità training
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)

        # Fase di Validazione
        model.eval() # Imposta il modello in modalità valutazione
        val_loss = 0.0
        with torch.no_grad(): # Disabilita il calcolo dei gradienti
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        # Calcola e salva le loss medie per l'epoca corrente
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # print(f"Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        # Log ogni 10 epoche (più la prima)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}] - "
                  f"Train Loss: {avg_train_loss:.6f} - "
		              f"Val Loss: {avg_val_loss:.6f}")

    training_time = time.time() - start_time
    print(f"\nAddestramento completato in {training_time:.2f} secondi")

    # --- 8. VALUTAZIONE SUL TEST SET ---
    print("\nValutazione sul test set...")
    model.eval()
    predictions_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions_scaled.append(outputs.cpu())
    predictions_scaled = torch.cat(predictions_scaled).numpy()

    # Inversione dello scaling per riportare le predizioni e i valori reali alla scala originale.
    dummy_array_preds = np.zeros((len(predictions_scaled), len(config.FEATURES)))
    dummy_array_preds[:, target_idx] = predictions_scaled.flatten()
    predictions_actual = scaler.inverse_transform(dummy_array_preds)[:, target_idx]

    dummy_array_y = np.zeros((len(y_test), len(config.FEATURES)))
    dummy_array_y[:, target_idx] = y_test.numpy().flatten()
    y_test_actual = scaler.inverse_transform(dummy_array_y)[:, target_idx]

    # Calcolo delle metriche finali
    metrics = calculate_metrics(y_test_actual, predictions_actual)
    print("\nMetriche di Performance sul Test Set:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")

    # --- 9. SALVATAGGIO DEI RISULTATI ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name_tag = f"Standalone_MultiStock_{len(config.STOCKS_TO_ANALYZE)}tickers"
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Salva i pesi del modello addestrato
    MODEL_PATH = os.path.join(config.OUTPUT_DIR, f"modello_{model_name_tag}_{timestamp}.pth")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModello salvato in: {MODEL_PATH}")

    # Salva un riepilogo dei risultati in formato JSON
    results_summary = {
        'run_type': 'standalone',
        'stocks_analyzed': config.STOCKS_TO_ANALYZE,
        'total_training_samples': len(X_train),
        'training_time_seconds': round(training_time, 2),
        'aggregate_performance_metrics': {k: round(v, 4) for k, v in metrics.items()},
    }
    RESULTS_PATH = os.path.join(config.OUTPUT_DIR, f"risultati_{model_name_tag}_{timestamp}.json")
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"Riepilogo risultati salvato in: {RESULTS_PATH}")

    # --- 10. PLOT DELLE CURVE DI APPRENDIMENTO ---
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Curve di Apprendimento (Training vs Validation Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    PLOT_PATH_LOSS = os.path.join(config.OUTPUT_DIR, f"grafico_loss_{model_name_tag}_{timestamp}.png")
    plt.savefig(PLOT_PATH_LOSS)
    print(f"Grafico Loss salvato in: {PLOT_PATH_LOSS}")
    plt.close()

# --- ESECUZIONE SCRIPT ---
if __name__ == "__main__":
    main()
