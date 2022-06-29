# MOBD
Progetto per il corso di Metodi di Ottimizzazione per Big Data dell'università di Roma Tor Vergata.
Questo readme contiene le istruzioni sia per l'ambiente locale sia per l'ambiente google colab.
## nota
sia nella home del progetto, sia nella cartella colab sono presenti due distinti file model.pkl, i due file sono lo stesso modello, ma creato con due versioni differenti di sklearn, quello in colab è stato creato sull'ambiente google colab, quello nella home è stato creato in locale con le versioni presenti nel file requirements.txt,
per motivi di compatibilità e preferibile che non vengano scambiati, usare il modello giusto per la versione giusta.

# google colab
1. Creare nella home di google drive (/drive/Mydrive) la cartella mobd e copiare all'interno il contenuto della cartella colab.
2. mettere il file di test rinominato 'test.csv' all'interno della cartella.
3. aprire il file load_test_and_score.ipynb e farlo partire.
## nota
il notebook `train_model.ipynb` ricrera il file model.pkl

# locale
1. installare i pacchetti usati in locale utilizzando il file requirements.txt con il seguente comando
``` sh
pip install -r requirements.txt
```
2. mettere nella cartella dove è presente sia il codice che il file model.pkl il file `test_set.csv`.
3. far partire lo script `load_and_score.py`
## nota
lo script `train_model.py` ricrera il file model.pkl

