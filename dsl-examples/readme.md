# DSL Examples

Dieses Dokument beschreibt die verwendeten Trainingsdaten von EMPA 
und beschreibt die Verwendung von DSL Instanzen mit MLPIPE CLI.

Wir gehen davon aus, dass MLPIPE bereits auf dem System [installiert](../readme.md) ist 
und über den Befehl `mlpipe` aufgerufen werden kann. 
Als Arbeitsverzeichnis soll der Ordner `dsl-examples` verwendet werden.
Dieser Ordner enthält Beispiele für DSL Instanzen (`*.yml`-Dateien) inkl. Sensordaten von EMPA.


## Daten

Die CSV-Datei `meeting_room_sensors_201807_201907.csv` enthält Sensoren, welche im Sitzungszimmer 12 und Sitzungszimmer 22 aufgezeichnet worden sind. In EMPA werden sogenannte *NumericId* für Sensoridentifikation verwendet. 
Die nachfolgende Tabelle zeigt das Mapping von *NumericId* zu Name des Sensors.

| Sensor           | Zimmer 012 | Zimmer 022 |
|------------------|------------|------------|
| Präsenz          | 40210148   | 40210149   |
| CO2              | 40210013   | 40210033   |
| Abluft           | 40210005   | 40210025   |
| Zuluft           | 40210002   | 40210022   |
| Innentemperatur  | 40210012   | 40210032   |
| Aussentemperatur | 3200000    | 3200000    |


## Workflows

### WF1 Daten analysieren

Beispiel für WF1:

    mlpipe analyze example.analyze.yml


### WF2 Modell entwickeln 

Beispiel für WF2:

    mlpipe train mlp_simple.train.yml

Die Konsolenausgaben enthält den Modellnamen sowie Session-Id.
Diese beiden Informationen müssen für WF3 bzw. WF4 angegeben werden.

### WF3 Modell evaluieren

Beispiel für WF3:

    mlpipe evaluate mlp_simple.evaluate.yml

Bemerkung: 

  * In dieser Modellevaluation werden Daten von zweiten Sitzungszimmer verwendet.
  * Die Variable `session` muss gemäss Output von WF2 angepasst werden.


### WF4 Modell integrieren

Beispiel für WF4:

    mlpipe integrate mlp_simple.integrate.yml

Bemerkung: 

  * Es muss hier vorher in der Datei `mlp_simple.integrate.yml` Benutzername und Password für den Zugriff auf EMPA Visualizer Schnittstelle eingegeben werden. 
  * Die Variable `session` muss gemäss Output von WF2 angepasst werden.
  * Im Fall von EMPA ist die Sequenzlänge und Zeitfenstergrösse äquivalent, 
weil die Sensoren einmal pro Minute abgetastet werden. Im WF2 haben wir für Aggregationen die Sequenzlänge 15 verwendet, deshalb setzen wir in der Datei `mlp_simple.integrate.yml` die Variable `duration_minutes` auf 15 Minuten.

