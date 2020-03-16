# MLPIPE

## Projektbeschreibung

TODO

## Entwicklungssprache

Dieses Projekt verwendet die Entwicklungssprache Python v3.7. 
Die erforderlichen Libraries sind im `requirements.txt` zu finden.
Eine Installationsanleitung für die virtuelle 
Umgebung von Python ist [hier](https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b) zufinden.

Nach der bereitstellung der virutellen Umgebung, 
können die erforderlichen Pakete mit folgendem Befehl installiert werden:

    pip install -r requirements.txt


## Ornderstruktur: Stammverzeichnis

Dies ist die Beschreibung der Ordner im Stammverzeichnis:

| Ordner    | Beschreibung                                                  |
|-----------|---------------------------------------------------------------|
| docker    | Enthält Dateien für die Erstellung der Docker Images          |
| mlpipe    | Enthält Sourcecode für die Softwarelösung MLPIPE              |
| schemas   | Enthält JSON-Schemas für die Beschreibung der DSL Instanzen   |
| templates | Enthält HTML-Template für die Generierung des Analyse Bericht |



## Ordnerstruktur: mlpipe

Der Sourcecode von MLPIPE ist im Ordner `mlpipe` zu finden und ist wie folgt strukturiert:

| Ordner          | Beschreibung                                                                                 |
|-----------------|----------------------------------------------------------------------------------------------|
| admin           | Skripte für administrative Zwecke                                                            |
| aggregators     | Klasse `AbstractAggregator` und deren Subklassen                                             |
| cli             | Implementation der Konsolenapplikation                                                       |
| config          | Klassen für die Verwaltung der Daten und Konfiguration                                       |
| datasources     | Klasse `AbstractDatasourceAdapter` und deren Subklassen                                      |
| dsl             | Export der Implementationen für DSL. Wird von `../scripts/generate_dsl_exports.sh` erstellt. |
| dsl_interpreter | Implementation des DSL Interpreter                                                           |
| exceptions      | Verwendete Exceptions in MLPIPE                                                              |
| groupers        | Klasse `AbstractGrouper` und deren Subklassen                                                |
| outputs         | Klasse `AbstractOutput` und deren Subklassen                                                 |
| pipeline        | Implementation der Pipes und Filter-Architektur                                              |
| processors      | Implementation der Transformatoren (modifizieren den Datenstromﬂ)                            |
| test            | Tests                                                                                        |
| utils           | Hilfsklassen                                                                                 |
| workflows       | Implementation der Workflows                                                                 |



Bemerkung:

  * Dateien im Ordner `dsl` werden lediglich verwendet, 
    um die implementierten Klassen für die DSL hervorzuheben. 
    Das erlaubt uns in der DSL den Präfix `mlpipe.dsl.*` 
    für den Auswahl der Klassen zu verwenden.

  
## Tests

Wir haben den Framework [Nose](https://nose.readthedocs.io/en/latest/index.html) verwendet. Die Tests können wie folgt ausgeführt werden:

    nosetests -i test_ mlpipe/test

## Docker Image

Wir haben die Softwarelösung für einfache 
Installation als Docker Image 
auf https://hub.docker.com bereit gestellt.
Nach der erfolgreichen 
[Installation von Docker Engine](https://docs.docker.com/install/)
können wir ein Skript für den einfachen 
Aufruf des *MLPIPE*-Containers erstellen.
Nachfolgend sehen wir ein Beispiel:

```bash
# File: mlpipe-cli-docker.sh
args="$@"
DSL_INSTANCES=$(pwd)
MLPIPE_TRAIN_DATA=/tmp/mlpipe_train_data

docker run --rm -it --name=mlpipe \
    -v$MLPIPE_TRAIN_DATA:/tmp/mlpipe/training \
    -v$DSL_INSTANCES:/data \
    robie2011/mlpipe $args
```

Beim Starten von Container wird der Pfad `$MLPIPE_TRAIN_DATA` 
auf dem Hostsystem mit dem Pfad `/tmp/mlpipe/training` im Container verbunden.
Dieser Ordner enthält Trainings-Daten wie Gewichte für das trainierte Modell.
Ebenso wird der aktueller Arbeitsverzeichnis mit 
dem Pfad `/data` im Container verbunden. 
Der zweite Ordner ist das Arbeitsverzeichnis im Container. 
Das Arbeitsverzeichnis wird zum Einlesen der Dateien
verwendet und auch zum Schreiben 
von Analyse Berichte aus *WF1 Daten analyse*.


Für den Schnellzugriff auf dem Skript haben wir den Alias 
`mlpipe` mit einer Verlinkung bereitgestellt. 
Hier ist ein Beispiel für die Verlinkung:


    ln -s /path/to/script/mlpipe-cli-docker.sh /usr/bin/mlpipe



Bemerkung: Werden Dateien als Datenquelle verwendet, 
so wird diese am besten in einem Unterordner von `$DSL_INSTANCES` abgelegt
und in der DSL Instanz als relativer Pfad angegeben.

