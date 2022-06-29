# MOBD
Progetto per il corso di Metodi di Ottimizzazione per Big Data dell'università di Roma Tor Vergata.
Questo readme contiene le istruzioni sia per l'ambiente locale.
# Istruzioni
1. assicurarsi di avere installato python è di averlo nel PATH
2. eseguire il clone del progetto con git e copiare il file `test_set.csv` nella cartella clonata.
```
git clone https://github.com/christiansantapaola/mobd
```
3. nella cartella del progetto aprire una shell ed creiamo un ambiente virtuale con il seguente comando:
``` ps
python -m venv mobd
```
4. attiviamo il nuovo ambiente di sviluppo python con il seguente comando:
- su windows:
``` ps
./mobd/Scripts/activate
```
- su linux/unix:
``` sh
source ./mobd/bin/activate
```
5. installare i pacchetti richiesti per far partire l'applicazione utilizzando il file requirements.txt con il seguente comando
``` sh
pip install -r requirements.txt
```
6. far partire lo script `load_and_score.py`
``` sh
python load_and_score.py
```

