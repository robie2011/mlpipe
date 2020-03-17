# MLPIPE

Dieses Dokument beschreibt den MLPIPE Projekt, 
die verwendeten Softwareversionen, 
die Ordnerstruktur dieser Repository,
die Verwendung von Docker Image 
sowie Beispiel DSL Instanzen.

## Projektbeschreibung

Die Entwicklung von Machine-Learning-Modellen besteht aus mehreren Prozessschritten. 
Viele dieser Prozessschritte werden von einem Framework bzw. einer Library bereits implementiert. 
Andere müssen selbst programmiert werden. Auswahl, Initialisierung und 
Zusammenführung dieser Prozessschritte werden in der Regel 
mit einer General Purpose Language wie Python, Java oder einer anderen beschrieben. 
Infolge der vielen Ausdrucksmöglichkeiten in General Purpose Languages 
haben die entwickelten Skripte bzw. Programme keine oder unterschiedliche Strukturen. 
Deshalb müssen für das Verständnis und die Weiterentwicklung 
nicht selbstentwickelter Modelle z. T. komplexe Quellcodes gelesen werden.

In dieser Masterarbeit wurden die Prozessschritte eines Deep-Learning-Projekts 
generalisiert und es wurde eine domänenspezifische Sprache (DSL) 
für die Beschreibung der Workflows entwickelt. Die entwickelte DSL 
definiert klare Strukturen und bietet eine vereinfachte, 
aussagekräftige und prägnante Sprache zur Beschreibung der 
einzelnen Prozessschritte im Workflow. Als Gegenstück zur DSL 
wurde eine Softwarelösung für die Ausführung der DSL-Instanzen entwickelt. 
Die Gesamtlösung beschleunigt die Entwicklung von Machine-Learning-Modellen 
und ermöglicht es, die einzelnen Prozessschritte im bestehenden Workflow besser zu verstehen.


Keywords: Machine Learning, Deep Learning, ANN, KNN, Keras, DSL, Domain Specific Language, Python

## Softwareversionen

Dieses Projekt verwendet die Programmiersprache Python v3.7. Für die Datenmanipulation werden die Libraries
NumPy, Pandas und Scikit-Learn eingesetzt. 
Als Machine Learning Framework verwenden wir 
Keras mit Tensorflow Backend.
Die erforderlichen Libraries sind im [requirements.txt](./requirements.txt) definiert.
Eine Installationsanleitung für die virtuelle 
Umgebung von Python ist [hier](https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b) zufinden. 
Nach Bereitstellung der virtuellen Umgebung, 
können die erforderlichen Pakete mit folgendem Befehl installiert werden:

    pip install -r requirements.txt


## Ordnerstruktur: Stammverzeichnis

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




Dateien im Ordner `dsl` werden lediglich verwendet, 
    um die implementierten Klassen für die DSL hervorzuheben. 
    Das erlaubt uns in der DSL den Präfix `mlpipe.dsl.*` 
    für den Auswahl der Klassen zu verwenden.

  
## Tests

Wir haben den Framework [Nose](https://nose.readthedocs.io/en/latest/index.html) verwendet. Die Tests können wie folgt ausgeführt werden:

    nosetests -i test_ mlpipe/test

## Docker Image

Die Softwarelösung wurde für einfache 
Installation als Docker Image 
auf der [öffentlichen Docker Registry](https://hub.docker.com/robie2011/mlpipe) bereit gestellt.
Nach der erfolgreichen 
[Installation von Docker Engine](https://docs.docker.com/install/)
kann ein Skript für den einfachen 
Aufruf des *MLPIPE*-Containers erstellt werden.
Nachfolgend sehen wir ein Beispiel:

```bash
# File: mlpipe-cli-docker.sh
args="$@"
DSL_INSTANCES=$(pwd)
FALLBACK="/tmp/mlpipe/training"
MLPIPE_TRAIN_DATA=${MLPIPE_TRAIN_DATA:-$FALLBACK}

docker run --rm -it \
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

## Beispiel DSL Instanzen

Im Unterordner [dsl-examples](dsl-examples) können Beispiel DSL Instanzen gefunden werden.
