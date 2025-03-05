# Wasserqualitätsanalyse in deutschen Bundesländern

Dieses Projekt analysiert Wasserqualitätsdaten aus verschiedenen deutschen Bundesländern, um Muster und Zusammenhänge zwischen chemischen Eigenschaften und der Trinkbarkeit des Wassers zu identifizieren.

## Projektübersicht

Die Analyse umfasst folgende Schritte:
- Einlesen und Aufbereitung der Wasserpotabilitätsdaten
- Zuweisung von Messstellen zu deutschen Bundesländern
- Analyse der Wasserqualität nach Bundesländern
- Vergleich von trinkbarem und nicht-trinkbarem Wasser
- Visualisierung der Ergebnisse

## Datensatz

Der verwendete Datensatz `water_potability.csv` enthält folgende Kennwerte:
- pH-Wert
- Härte
- Feststoffe
- Chloramine
- Sulfate
- Leitfähigkeit
- Organischer Kohlenstoff
- Trihalomethane
- Trübung
- Trinkbarkeit (0 = nicht trinkbar, 1 = trinkbar)

## Voraussetzungen

```
pandas
numpy
matplotlib
seaborn
```

## Installation

```bash
git clone https://github.com/yasinfaridzad/wasserqualitaetsanalyse.git
cd wasserqualitaetsanalyse
pip install -r requirements.txt
```

## Nutzung

```python
python main.py
```

## Ergebnisse

Die Analyse zeigt Unterschiede in der Wasserqualität zwischen verschiedenen Bundesländern. Besonders auffällig ist, dass [Bundesland] die höchste Anzahl an Messstellen mit trinkbarem Wasser aufweist.

## Lizenz

MIT

## Autor

Yasin Faridzad
