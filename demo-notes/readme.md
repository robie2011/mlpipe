# Demo Setup

Für das Demo verwenden wir die DSL-Instanz `mlp_simple.integrate.yml`, 
in dem versucht wird den Präsenzsensor (NumericId: 40210148) vorherzusagen.
Die DSL-Instanz verwendet ein Output-Adapter, welches die Vorhersagen in ein CSV-Datei umleitet.

Für das Demo haben wir zwei Skripte erstellt; 
Eines um die aktuelle Präsenzmeldung von Visualizer API in eine CSV-Datei zu speichern (1) und
ein anderes, um die Präsenzmeldungen aus beiden CSV-Datei herauszulesen und zu vergleichen (2). 


(1): `/scripts/start_livemonitoring.sh`

(2): `/scripts/show_csv_results.sh`  
    

Nachfolgend sehen wir die Befehle für das Demo. 
Jedes Befehl wird in ein separates Terminal ausgeführt.
Je nach Umgebung muss das virtuelle Python Umgebung zuerst aktiviert werden (`venv/bin/activate`).
Für die Befehle wird das Arbeitsverzeichnis `/dsl-examples` verwendet.

```bash
../mlpipe.cli.sh integrate credentials_mlp_simple.integrate.yml
../scripts/start_livemonitoring.sh
../scripts/show_csv_results.sh
```


Für die Konfiguration der Skripte wird die YAML-Datei 
`.live_monitoring` im Arbeitsverzeichnis verwendet.
Diese enthält Attribute für den Zugriff auf 
den Visualizer API, Auswahl von Metrik für das Monitoring 
sowie den Ausgabepfad für die CSV-Datei.
Nachfolgend sehen wir eine YAML-Beispielkonfiguration:
  
```yaml
username: 'NEST\????'
password: '?????'
csv_file_out: /tmp/mlpipe_live_40210148
metric: 40210148
```