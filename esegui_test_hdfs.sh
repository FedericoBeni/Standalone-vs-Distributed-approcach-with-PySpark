# =============================================================================
# Script per eseguire il benchmark distribuito con HDFS
#
# USO:
#   ./esegui_test_hdfs.sh      (esegue con 4 processi di default)
#   ./esegui_test_hdfs.sh 2    (esegue con 2 processi)
# =============================================================================

# Legge il numero di processi dal primo argomento passato allo script.
# Se non viene passato nessun argomento, usa 4 come valore di default.
NUM_PROCESSES=${1:-4}

echo "---------------------------------------------------------"
echo "Avvio del benchmark HDFS con $NUM_PROCESSES processi..."
echo "---------------------------------------------------------"

# Attiva l'ambiente virtuale
echo "Attivazione ambiente virtuale..."
source venv/bin/activate

# Esporta le variabili d'ambiente necessarie per Hadoop e Spark
echo "Configurazione delle variabili d'ambiente..."
export HADOOP_HOME=/opt/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)
export PYSPARK_PYTHON=$(which python)
export NUM_PROCESSES=$NUM_PROCESSES

# Lancia il comando spark-submit
echo "Avvio di spark-submit..."
spark-submit --master local[$NUM_PROCESSES] train_distributed_hdfs.py

echo "---------------------------------------------------------"
echo "Esecuzione completata."
echo "---------------------------------------------------------"
