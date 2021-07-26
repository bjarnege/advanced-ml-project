# advanced-ml-project

# Hardwarevoraussetzungen

Das folgende Projekt stellt hohe Anforderungen an die Hardware, auf der es ausgeführt wird.  
Abhängig davon, ob ein Neutraining angestrebt wird, sind so also mindestens 30 GB Speicherplatz vorzuhalten.  
Des weiteren werden mindestens 32 GB RAM für die Vorhersagegenerierung empfohlen. Auf Systemen mit geringeren Spezifikationen ist das Projekt nicht getestet worden und kann ggf. zu Fehlfunktionen führen.  

Um das Neutrainining anzuregen ist außerdem eine performante GPU + CUDA empfehlenswert, da so der Prozess deutlich verkürzt wird.  


# Vorbereitung

Im den vorhanden Code auszuführen zu können, sind folgende Schritte auszuführen  

```
pip install -r requirements.txt  
```

# Vortrainierte Bereitstellung des Projektes

Um bereits vorverarbeitete Daten beziehen zu können muss der folgende Python-Code, auf der obersten ebene des Repositories ausgeführt werden.  

```
import gdown   
import zipfile  

# Beziehen der pickle-files (Modelle, Vectoren, Metadaten)  
gdown.download('https://drive.google.com/uc?id=1FuTYss2WElJuA1zkU-ldXg9n-_g3bzby')  

# Extrahieren des Archivs
with zipfile.ZipFile("resource.zip", 'r') as zip_ref:  
    zip_ref.extractall("./resource")  
```

# Retraining mit neuen/anderen Daten

Falls cuda aktuell nicht installiert ist, sollte es installiert werden!
Ansonsten wird der Schritt "python ./create-knn-datasets.py" ewig dauern.

```
pip install -r requirements.txt  
cd ./src/arxiv-data-access  
python ./download-files.py  
# Der folgende Schritt wird manuell Aausgeführt:  
# Downloaden der arXiv-Metadaten von:  
# https://www.kaggle.com/Cornell-University/arxiv?select=arxiv-metadata-oai-snapshot.json  
# Entpacken in den Ordner resource  
# Nun weiter in der CMD mit (im Ordner src/arxiv-data-access):  
python ./metadata-collector.py  
cd ../KNNRecommends/  
python ./create-knn-datasets.py  

```

# Ausführen des Codes

Um nun die API muss folgendes getan werden:

```
cd ./src/  
python ./api.py  
```

Nachdem die Schritte durchgeführt worden, ist die API aktiv.  
Demnach können nun fast alle arXiv-Paper, mit Ausnahme der aktuellesten Paper, da diese noch nicht in den Metadaten enthalten sind verarbeitet werden.  
Anfragen an die API müssen den folgenden Format folgen:  
0.0.0.0:12345/api?url=**Link zum PDF**&pipeline=**Schritte der Empfehlungspipeline**

Ein Beispiel dafür ist der folgende Code:  
0.0.0.0:12345/api?url=https://arxiv.org/pdf/gr-qc/9411004&pipeline=titles,abstracts