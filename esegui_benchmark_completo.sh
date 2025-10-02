# =============================================================================
# Script MASTER per eseguire l'intero benchmark HDFS per N volte,
# svuotando la cache prima di ogni run e salvando l'output su un file di log.
#
# USO:
#   ./esegui_benchmark_completo.sh      (esegue 10 run)
#   ./esegui_benchmark_completo.sh 5    (esegue 5 run)
# =============================================================================

NUM_RUNS=${1:-10} # Numero di esecuzioni, 10 di default
LOG_FILE="benchmark_hdfs_results.log" # Nome del file dove salvare i log

# Pulisce il file di log precedente all'inizio del benchmark
echo "========================================================" > $LOG_FILE
echo "INIZIO BENCHMARK COMPLETO SU HDFS" >> $LOG_FILE
echo "Verranno eseguiti $NUM_RUNS run." >> $LOG_FILE
echo "Data: $(date)" >> $LOG_FILE
echo "========================================================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Stampa un messaggio anche a console per sapere che è partito
echo "Benchmark avviato. L'output verrà salvato in '$LOG_FILE'."
echo "Puoi monitorare il progresso con il comando: tail -f $LOG_FILE"

# Ciclo for da 1 a NUM_RUNS
for (( i=1; i<=$NUM_RUNS; i++ ))
do
  # Aggiunge un separatore al file di log
  echo "---------------------------------------------------------" >> $LOG_FILE
  echo "INIZIO RUN $i / $NUM_RUNS" >> $LOG_FILE
  echo "---------------------------------------------------------" >> $LOG_FILE
  
  # PASSO 1: Svuota la cache di sistema
  echo "Svuotamento cache di sistema (richiede password sudo)..."
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  sleep 5
  echo "Cache svuotata." >> $LOG_FILE
  
  # PASSO 2: Esegui lo script di test e accoda TUTTO l'output al file di log
  # '2>&1' reindirizza anche gli errori (stderr) nello stesso file, così non perdi nulla.
  ./esegui_test_hdfs.sh 4 >> $LOG_FILE 2>&1
  
  echo "---------------------------------------------------------" >> $LOG_FILE
  echo "FINE RUN $i / $NUM_RUNS" >> $LOG_FILE
  echo "---------------------------------------------------------" >> $LOG_FILE
  echo "" >> $LOG_FILE
  
  if [ $i -lt $NUM_RUNS ]; then
      echo "Pausa di 15 secondi prima del prossimo run..." >> $LOG_FILE
      sleep 15
  fi
done

echo "========================================================" >> $LOG_FILE
echo "BENCHMARK COMPLETO TERMINATO." >> $LOG_FILE
echo "========================================================" >> $LOG_FILE

# Messaggio finale a console
echo "Benchmark terminato. Risultati completi salvati in '$LOG_FILE'."
