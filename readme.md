# MOBD
Progetto per il corso di Metodi di Ottimizzazione per Big Data dell'università di Roma Tor Vergata.
Questo readme contiene le istruzioni sia per l'ambiente locale sia per l'ambiente google colab.

# google colab
1. Creare nella home di google drive (/drive/Mydrive) la cartella mobd e copiare all'interno il contenuto della cartella colab.
2. mettere il file di test rinominato 'test.csv' all'interno della cartella.
2. aprire il file load_test_and_score.ipynb e farlo partire.

# locale
1. installare i pacchetti usati in locale utilizzando il file requirements.txt con il seguente comando
``` sh
pip install -r requirements.txt
```
2. mettere nella cartella dove è presente sia il codice che il file model.pkl il file `test.csv`.
2. far partire lo script `load_and_score.py`