# advanced-ml-project

# How to setup

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
cd ../
python ./api.py
```

Nachdem die Schritte durchgeführt worden, ist die API aktiv.
Demnach können nun fast alle arXiv-Paper, mit Ausnahme der aktuellesten Paper, da diese noch nicht in den Metadaten enthalten sind verarbeitet werden.
Dies geschieht wie folgt:

1. Ihr sucht euch ein Paper aus
2. Ihr kopiert extrahiert die Kategorie und die ID des Papers aus dessen URL
> https://arxiv.org/abs/quant-ph/0110064  
> quant-ph/0110064
3. Ihr gebt dieses Paper an die API durch:
>> http://localhost:12345/api/quant-ph/0110064
4. Ihr bekommt folgende Rückgabe
```
{"abstract":"  We study a wide class of solvable PT symmetric potentials in order to\nidentify conditions under which these potentials have regular solutions with\ncomplex energy. Besides confirming previous findings for two potentials, most\nof our results are new. We demonstrate that the occurrence of conjugate energy\npairs is a natural phenomenon for these potentials. We demonstrate that the\npresent method can readily be extended to further potential classes.\n",  
"author_ids":[474918,1242915],  
"authors":"G. Levai and M. Znojil",  
"authors_parsed":[["Levai","G.",""],["Znojil","M.",""]],  
"categories":"quant-ph","doi":"10.1142/S0217732301005321",  
"title":"Conditions for complex spectra in a class of PT symmetric potentials",  
"top_n_abstracts":{"physics/0002008":0.7434443235397339,"physics/0003034":0.7316288948059082,"physics/0003043":0.7258390784263611,"physics/0003081":0.7594684362411499,"physics/0004017":0.7294284701347351,"physics/0004076":0.7397626042366028,"physics/0005027":0.7477297782897949,"physics/0005036":0.7498555779457092,"physics/0006009":0.7265245914459229,"physics/0006018":0.7285990715026855},  
"top_n_titles":{"physics/0001008":0.8143051862716675,"physics/0002011":0.8109896779060364,"physics/0003039":0.8128641247749329,"physics/0003081":0.8091407418251038,"physics/0003085":0.8055613040924072,"physics/0003097":0.8076887130737305,"physics/0004014":0.8167643547058105,"physics/0006013":0.815031886100769,"physics/0006017":0.8083177804946899,"physics/0006053":0.8160438537597656}}
```
