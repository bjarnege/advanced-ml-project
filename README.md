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
cd ./src/arxiv-data-access # navigate to the arxiv-data-access folder  
python ./download-files.py # please modify the file if you want to download any other topics  
  
######################## MANUAL STEP ##########################  
# The arXiv-metadata must be downloaded manually from kaggle.com by following these steps:  
https://www.kaggle.com/Cornell-University/arxiv?select=arxiv-metadata-oai-snapshot.json    
# Unpack the archive, s.t. the file /resource/arxiv-metadata-oai-snapshop.json will be created  
###############################################################   

cd ./src/arxiv-data-access # navigate again into the arxiv-data-access folder  
python ./metadata-collector.py # creates the co-author-mapping and DataFrames for the metadata  
cd ../KNNRecommends # navigate to the KNN-folder  
python ./create-knn_sciBERT-datasets.py # Vectorize the titles and abstracts of the downloaded papers  

######### MISSING PART ###############  
# tbd by Johannes  
######################################  
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