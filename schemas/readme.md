# Schemas

Wir verwenden für jeden Workflow eine eigene Schema.
Die nachfolgende Tabelle listet den Schema für jeden Workflow auf.

| Workflow                | Datei                                                |
|-------------------------|------------------------------------------------------|
| WF 1 Daten analysieren  | [wf-analyze.schema.json](wf-analyze.schema.json)     |
| WF 2 Modell entwickeln  | [wf-train.schema.json](wf-train.schema.json)         |
| WF 3 Modell evaluieren  | [wf-evaluate.schema.json](wf-evaluate.schema.json)   |
| WF 4 Modell integrieren | [wf-integrate.schema.json](wf-integrate.schema.json) |

Die Liste der möglichen Implementationen für jede Schnittstelle
wurde als eine eigene Schema abgelegt und im Schema der Workflows referenziert.
Die nachfolgende Tabelle listet den Schema für jede Schnittstelle auf.

Im Workflow *WF1 Daten analysieren* verwenden 
wir Aggregatoren für die Generierung der Metriken.
In diesem Kontext benötigen wir keine Sequenzlänge 
und der `fields`-Parameter für die Aggregatoren ist auch irrelevant.
Deshalb haben wir die Datei `aggregators.schema.json` 
zusätzlich in einer reduzierten Variante als `metrics.schema.json` publiziert.
Für die Generierung dieser reduzierten Variante verwenden wir 
den Skript [generate_metrics_schema.js](generate_metrics_schema.js).


| Schnittstelle               | Datei                                              |
|-----------------------------|----------------------------------------------------|
| `AbstractDatasourceAdapter` | [datasource.schema.json](datasource.schema.json)   |
| `AbstractAggregator`        | [aggregators.schema.json](aggregators.schema.json) |
| `AbstractAggregator`        | [metrics.schema.json](metrics.schema.json) |
| `AbstractOutput`            | [outputs.schema.json](outputs.schema.json)         |
| `AbstractProcessor`         | [processors.schema.json](processors.schema.json)   |


Für die Verwendung der Schemen muss das IDE eingerichtet werden. 
Eine Beispielkonfiguration für Visual Studio Code ist bereits für die Beispiele im Ordner [dsl-examples](../dsl-examples) erstellt worden. [Die Konfigurationsdatei](../dsl-examples/.vscode/settings.json) beschreibt die Schemen für DSL Instanzen in YAML-Format. 
Für die Verwendung der Konfiguration muss Visual Studio Code den Ordner `dsl-examples` als Stammordner öffnen und die Namenskonvention gemäss nachfolgender Tabelle einhalten.

| Workflow | Muster für Dateinamen |
|----------|-----------------------|
| WF1      | `*.analyze.yml`       |
| WF2      | `*.train.yml`         |
| WF3      | `*.evaluate.yml`      |
| WF4      | `*.integrate.yml`     |



Bemerkungen: Falls das Schema geändert wird, muss Visual Studio neugestartet werden.


## Links

  * [Schema Visualizer](https://navneethg.github.io/jsonschemaviewer/)
  * [Schema Validation for YAML](https://json-schema-everywhere.github.io/yaml)
  * [Visual Studio Code YAML Intelisense konfigurieren](https://joshuaavalon.io/intellisense-json-yaml-vs-code)

