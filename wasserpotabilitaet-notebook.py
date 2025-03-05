{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Wasserqualitätsanalyse in deutschen Bundesländern\n",
    "\n",
    "Dieses Notebook analysiert Daten zur Wasserpotabilität und weist Messstellen den verschiedenen deutschen Bundesländern zu, um regionale Unterschiede zu untersuchen."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Bibliotheken importieren und Daten laden"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Einstellungen für bessere Visualisierungen\n",
    "plt.style.use('seaborn-whitegrid')\n",
    "sns.set_palette('colorblind')\n",
    "plt.rcParams['figure.figsize'] = (12, 8)\n",
    "plt.rcParams['font.size'] = 12\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Daten laden\n",
    "water = pd.read_csv('water_potability.csv')\n",
    "water.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Daten erkunden"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Informationen über den Datensatz anzeigen\n",
    "water.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistische Zusammenfassung\n",
    "water.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Überprüfen auf fehlende Werte\n",
    "missing_values = water.isnull().sum()\n",
    "print(f\"Fehlende Werte pro Spalte:\\n{missing_values}\")\n",
    "\n",
    "# Prozentsatz der fehlenden Werte\n",
    "print(f\"\\nProzentsatz der fehlenden Werte pro Spalte:\")\n",
    "print((missing_values / len(water) * 100).round(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fehlende Werte mit dem Mittelwert ersetzen\n",
    "water = water.fillna(water.mean())\n",
    "\n",
    "# Überprüfen, ob alle fehlenden Werte ersetzt wurden\n",
    "print(f\"Verbleibende fehlende Werte: {water.isnull().sum().sum()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Indizierung und Bundesländer zuweisen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Messstellen-IDs erstellen\n",
    "anzahl_messstellen = len(water)\n",
    "indizes = [f'M{i+1}' for i in range(anzahl_messstellen)]\n",
    "water.index = indizes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bundesländer definieren\n",
    "bundeslaender = ['Hessen', 'Sachsen', 'Bayern', 'Nordrhein-Westfalen', \n",
    "                'Baden-Württemberg', 'Hamburg', 'Berlin']\n",
    "\n",
    "# Zufallszahlen erstellen\n",
    "np.random.seed(42)  # Für Reproduzierbarkeit\n",
    "zufallszahlen = np.random.choice(range(1, anzahl_messstellen+1), \n",
    "                                size=anzahl_messstellen, replace=False)\n",
    "\n",
    "# In 7 Gruppen aufteilen\n",
    "gruppen = np.array_split(zufallszahlen, 7)\n",
    "\n",
    "# Bundesländer zuweisen\n",
    "laender = []\n",
    "for i in range(7):\n",
    "    laender.extend([bundeslaender[i]] * len(gruppen[i]))\n",
    "\n",
    "# Zum DataFrame hinzufügen\n",
    "water['Bundesland'] = laender\n",
    "\n",
    "# Verteilung anzeigen\n",
    "bundesland_verteilung = water['Bundesland'].value_counts()\n",
    "print(\"Verteilung der Messstellen nach Bundesländern:\")\n",
    "print(bundesland_verteilung)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Daten nach Trinkbarkeit aufteilen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Aufteilen in trinkbar und nicht-trinkbar\n",
    "df_trinkbar = water.loc[water['Potability'] == 1]\n",
    "df_nicht_trinkbar = water.loc[water['Potability'] == 0]\n",
    "\n",
    "print(f\"Trinkbares Wasser: {len(df_trinkbar)} Messstellen ({len(df_trinkbar)/len(water)*100:.1f}%)\")\n",
    "print(f\"Nicht-trinkbares Wasser: {len(df_nicht_trinkbar)} Messstellen ({len(df_nicht_trinkbar)/len(water)*100:.1f}%)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Bundesland mit dem besten Wasser finden"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Zählen der Messstellen pro Bundesland (trinkbar)\n",
    "trinkbar_pro_bundesland = df_trinkbar['Bundesland'].value_counts()\n",
    "\n",
    "# Gesamtzahl der Messstellen pro Bundesland\n",
    "gesamt_pro_bundesland = water['Bundesland'].value_counts()\n",
    "\n",
    "# Anteil berechnen\n",
    "anteil_pro_bundesland = (trinkbar_pro_bundesland / gesamt_pro_bundesland * 100).round(2)\n",
    "\n",
    "# Bundesland mit dem höchsten Anteil an trinkbarem Wasser\n",
    "bestes_bundesland = anteil_pro_bundesland.idxmax()\n",
    "bester_anteil = anteil_pro_bundesland.max()\n",
    "\n",
    "print(f\"Bundesland mit dem höchsten Anteil trinkbaren Wassers: {bestes_bundesland} ({bester_anteil}%)\")\n",
    "\n",
    "# Ergebnisse in DataFrame für bessere Übersicht\n",
    "ergebnisse = pd.DataFrame({\n",
    "    'Bundesland': gesamt_pro_bundesland.index,\n",
    "    'Messstellen_Gesamt': gesamt_pro_bundesland.values,\n",
    "    'Messstellen_Trinkbar': trinkbar_pro_bundesland.reindex(gesamt_pro_bundesland.index).fillna(0).values,\n",
    "    'Anteil_Trinkbar_Prozent': anteil_pro_bundesland.reindex(gesamt_pro_bundesland.index).fillna(0).values\n",
    "})\n",
    "\n",
    "ergebnisse"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Wertebereiche vergleichen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistiken für trinkbares Wasser\n",
    "print(\"Wertebereiche für Trinkbares Wasser:\")\n",
    "stats_trinkbar = df_trinkbar.describe()\n",
    "stats_trinkbar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistiken für nicht-trinkbares Wasser\n",
    "print(\"Wertebereiche für Nicht-Trinkbares Wasser:\")\n",
    "stats_nicht_trinkbar = df_nicht_trinkbar.describe()\n",
    "stats_nicht_trinkbar"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Korrelationsanalyse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Korrelationen zwischen den Eigenschaften berechnen\n",
    "korrelation = water.corr()\n",
    "\n",
    "# Korrelationsmatrix visualisieren\n",
    "plt.figure(figsize=(12, 10))\n",
    "sns.heatmap(korrelation, annot=True, cmap='coolwarm', linewidths=0.5)\n",
    "plt.title('Korrelation zwischen den Wassereigenschaften')\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Visualisierungen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Balkendiagramm: Vergleich der Bundesländer\n",
    "plt.figure(figsize=(14, 7))\n",
    "bar_width = 0.35\n",
    "index = np.arange(len(ergebnisse))\n",
    "\n",
    "plt.bar(index, ergebnisse['Messstellen_Gesamt'], bar_width, \n",
    "        label='Gesamt', color='lightgray')\n",
    "plt.bar(index + bar_width, ergebnisse['Messstellen_Trinkbar'], bar_width,\n",
    "        label='Trinkbar', color='steelblue')\n",
    "\n",
    "plt.xlabel('Bundesland')\n",
    "plt.ylabel('Anzahl Messstellen')\n",
    "plt.title('Verteilung der Wasserqualität nach Bundesländern')\n",
    "plt.xticks(index + bar_width/2, ergebnisse['Bundesland'], rotation=45)\n",
    "plt.legend()\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Kreisdiagramm: Anteil trinkbaren Wassers nach Bundesland\n",
    "plt.figure(figsize=(10, 8))\n",
    "plt.pie(ergebnisse['Anteil_Trinkbar_Prozent'], \n",
    "        labels=ergebnisse['Bundesland'],\n",
    "        autopct='%1.1f%%',\n",
    "        startangle=90,\n",
    "        shadow=True)\n",
    "plt.axis('equal')\n",
    "plt.title('Anteil trinkbaren Wassers nach Bundesland')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Boxplots für verschiedene Eigenschaften nach Trinkbarkeit\n",
    "eigenschaften = [col for col in water.columns if col not in ['Potability', 'Bundesland']]\n",
    "\n",
    "fig, axes = plt.subplots(3, 3, figsize=(18, 14))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, eigenschaft in enumerate(eigenschaften):\n",
    "    if i < len(axes):\n",
    "        # Erstellen eines neuen DataFrames für das Plotting\n",
    "        plot_data = pd.DataFrame({\n",
    "            'Wert': pd.concat([df_trinkbar[eigenschaft], df