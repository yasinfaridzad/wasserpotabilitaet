import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class WasserQualitaetsAnalyse:
    """
    Eine Klasse zur Analyse von Wasserqualitätsdaten in deutschen Bundesländern.
    """

    def __init__(self, csv_pfad):
        """
        Initialisiert die Analyseklasse mit dem Datensatz.

        Args:
            csv_pfad: Pfad zur CSV-Datei mit den Wasserqualitätsdaten
        """
        self.original_data = pd.read_csv(r'C:\Users\Administrator\Documents\Kursmaterial\Selbs_übung\Github_wasserpotabilitaets\wasserpotabilitaet\data\water_potability.csv')
        self.data = self.original_data.copy()
        self.bundeslaender = ['Hessen', 'Sachsen', 'Bayern', 'Nordrhein-Westfalen',
                              'Baden-Württemberg', 'Hamburg', 'Berlin']
        self.df_trinkbar = None
        self.df_nicht_trinkbar = None

        # Ordner für Ergebnisse erstellen
        if not os.path.exists('ergebnisse'):
            os.makedirs('ergebnisse')
        if not os.path.exists('plots'):
            os.makedirs('plots')

    def daten_vorbereiten(self):
        """
        Bereitet die Daten vor, indem fehlende Werte behandelt werden und
        eine eindeutige Indexierung erstellt wird.
        """
        # Überprüfen auf fehlende Werte
        fehlende_werte = self.data.isnull().sum()
        print("Fehlende Werte pro Spalte:")
        print(fehlende_werte)

        # Behandeln fehlender Werte durch Mittelwert-Ersetzung
        self.data = self.data.fillna(self.data.mean())

        # Eindeutige Indizes erstellen
        anzahl_messstellen = len(self.data)
        indizes = [f'M{i + 1}' for i in range(anzahl_messstellen)]
        self.data.index = indizes

        print(f"Daten mit {anzahl_messstellen} Messstellen vorbereitet.")

    def bundeslaender_zuweisen(self):
        """
        Weist den Messstellen zufällig ein Bundesland zu.
        """
        anzahl_messstellen = len(self.data)

        # Zufällige Indizes erstellen
        zufallszahlen = np.random.choice(range(1, anzahl_messstellen + 1),
                                         size=anzahl_messstellen, replace=False)

        # In 7 Gruppen aufteilen (6 Hauptgruppen + Rest für Berlin)
        haupt_gruppen = np.array_split(zufallszahlen[:6 * (anzahl_messstellen // 7)], 6)
        rest = zufallszahlen[6 * (anzahl_messstellen // 7):]

        # Bundesländer den Gruppen zuweisen
        laender = []
        for i in range(6):
            laender.extend([self.bundeslaender[i]] * len(haupt_gruppen[i]))
        laender.extend([self.bundeslaender[6]] * len(rest))

        # Bundesland-Spalte zum DataFrame hinzufügen
        self.data['Bundesland'] = laender

        bundesland_verteilung = self.data['Bundesland'].value_counts()
        print("Verteilung der Messstellen nach Bundesländern:")
        print(bundesland_verteilung)

    def daten_aufteilen(self):
        """
        Teilt die Daten in trinkbar und nicht-trinkbar auf.
        """
        self.df_trinkbar = self.data[self.data['Potability'] == 1]
        self.df_nicht_trinkbar = self.data[self.data['Potability'] == 0]

        print(f"Trinkbares Wasser: {len(self.df_trinkbar)} Messstellen")
        print(f"Nicht-trinkbares Wasser: {len(self.df_nicht_trinkbar)} Messstellen")

    def bundesland_mit_bestem_wasser_finden(self):
        """
        Findet das Bundesland mit dem höchsten Anteil an trinkbarem Wasser.
        """
        # Zählen der Messstellen pro Bundesland (trinkbar)
        trinkbar_pro_bundesland = self.df_trinkbar['Bundesland'].value_counts()

        # Gesamtzahl der Messstellen pro Bundesland
        gesamt_pro_bundesland = self.data['Bundesland'].value_counts()

        # Anteil berechnen
        anteil_pro_bundesland = (trinkbar_pro_bundesland / gesamt_pro_bundesland * 100).round(2)

        # Bundesland mit dem höchsten Anteil an trinkbarem Wasser
        bestes_bundesland = anteil_pro_bundesland.idxmax()
        bester_anteil = anteil_pro_bundesland.max()

        print(f"\nBundesland mit dem höchsten Anteil trinkbaren Wassers: {bestes_bundesland} ({bester_anteil}%)")

        # Ergebnisse speichern
        ergebnisse = pd.DataFrame({
            'Bundesland': gesamt_pro_bundesland.index,
            'Messstellen_Gesamt': gesamt_pro_bundesland.values,
            'Messstellen_Trinkbar': trinkbar_pro_bundesland.reindex(gesamt_pro_bundesland.index).fillna(0).values,
            'Anteil_Trinkbar_Prozent': anteil_pro_bundesland.reindex(gesamt_pro_bundesland.index).fillna(0).values
        })

        ergebnisse.to_csv('ergebnisse/bundeslaender_analyse.csv', index=False)
        return ergebnisse

    def wertebereiche_vergleichen(self):
        """
        Vergleicht die Wertebereiche von trinkbarem und nicht-trinkbarem Wasser.
        """
        # Statistiken für trinkbares Wasser
        stats_trinkbar = self.df_trinkbar.describe()
        stats_nicht_trinkbar = self.df_nicht_trinkbar.describe()

        # Statistiken speichern
        stats_trinkbar.to_csv('ergebnisse/statistik_trinkbar.csv')
        stats_nicht_trinkbar.to_csv('ergebnisse/statistik_nicht_trinkbar.csv')

        return stats_trinkbar, stats_nicht_trinkbar

    def korrelationsanalyse(self):
        """
        Führt eine Korrelationsanalyse der Wassereigenschaften durch.
        """
        # Entferne die nicht-numerischen Spalten für die Korrelationsanalyse
        daten_fuer_korrelation = self.data.drop(columns=['Bundesland'])

        korrelation = daten_fuer_korrelation.corr()

        # Korrelationsmatrix speichern
        korrelation.to_csv('ergebnisse/korrelationsmatrix.csv')

        # Korrelationsmatrix visualisieren
        plt.figure(figsize=(12, 10))
        sns.heatmap(korrelation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Korrelation zwischen den Wassereigenschaften')
        plt.tight_layout()
        plt.savefig('plots/korrelationsmatrix.png')
        plt.close()

        return korrelation

    def visualisiere_bundeslaender_vergleich(self):
        """
        Visualisiert den Vergleich der Wasserqualität nach Bundesländern.
        """
        ergebnisse = self.bundesland_mit_bestem_wasser_finden()

        # Balkendiagramm erstellen
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(ergebnisse))

        plt.bar(index, ergebnisse['Messstellen_Gesamt'], bar_width,
                label='Gesamt', color='lightgray')
        plt.bar(index + bar_width, ergebnisse['Messstellen_Trinkbar'], bar_width,
                label='Trinkbar', color='steelblue')

        plt.xlabel('Bundesland')
        plt.ylabel('Anzahl Messstellen')
        plt.title('Verteilung der Wasserqualität nach Bundesländern')
        plt.xticks(index + bar_width / 2, ergebnisse['Bundesland'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/bundeslaender_vergleich.png')
        plt.close()

        # Kreisdiagramm für den Anteil trinkbaren Wassers
        plt.figure(figsize=(10, 8))
        plt.pie(ergebnisse['Anteil_Trinkbar_Prozent'],
                labels=ergebnisse['Bundesland'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True)
        plt.axis('equal')
        plt.title('Anteil trinkbaren Wassers nach Bundesland')
        plt.tight_layout()
        plt.savefig('plots/anteil_trinkbar_kreisdiagramm.png')
        plt.close()

    def visualisiere_eigenschaften_boxplots(self):
        """
        Erstellt Boxplots für die verschiedenen Wassereigenschaften, aufgeteilt nach Trinkbarkeit.
        """
        # Eigenschaften im Datensatz (ohne Potability und Bundesland)
        eigenschaften = [col for col in self.data.columns
                         if col not in ['Potability', 'Bundesland']]

        # Für jede Eigenschaft einen Boxplot erstellen
        for eigenschaft in eigenschaften:
            plt.figure(figsize=(10, 6))

            # Erstellen eines neuen DataFrames für das Plotting
            plot_data = pd.DataFrame({
                'Wert': pd.concat([self.df_trinkbar[eigenschaft], self.df_nicht_trinkbar[eigenschaft]]),
                'Kategorie': ['Trinkbar'] * len(self.df_trinkbar) + ['Nicht Trinkbar'] * len(self.df_nicht_trinkbar)
            })

            # Boxplot erstellen
            sns.boxplot(x='Kategorie', y='Wert', data=plot_data)
            plt.title(f'Verteilung von {eigenschaft} nach Trinkbarkeit')
            plt.ylabel(eigenschaft)
            plt.tight_layout()
            plt.savefig(f'plots/boxplot_{eigenschaft}.png')
            plt.close()

    def visualisiere_bundesland_eigenschaften(self):
        """
        Visualisiert die durchschnittlichen Eigenschaften je Bundesland.
        """
        # Eigenschaften im Datensatz (ohne Potability und Bundesland)
        eigenschaften = [col for col in self.data.columns
                         if col not in ['Potability', 'Bundesland']]

        # Für jede Eigenschaft einen Barplot erstellen
        for eigenschaft in eigenschaften:
            plt.figure(figsize=(12, 6))

            # Durchschnittswerte pro Bundesland berechnen
            durchschnitt = self.data.groupby('Bundesland')[eigenschaft].mean().sort_values(ascending=False)

            # Barplot erstellen
            ax = durchschnitt.plot(kind='bar', color='steelblue')
            plt.title(f'Durchschnittlicher {eigenschaft} nach Bundesland')
            plt.ylabel(eigenschaft)
            plt.xlabel('Bundesland')
            plt.xticks(rotation=45)

            # Werte über den Balken anzeigen
            for i, v in enumerate(durchschnitt):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(f'plots/bundesland_{eigenschaft}.png')
            plt.close()

    def fuehre_gesamtanalyse_durch(self):
        """
        Führt die vollständige Analyse durch und erstellt alle Visualisierungen.
        """
        print("Start der Wasserqualitätsanalyse")
        self.daten_vorbereiten()
        self.bundeslaender_zuweisen()
        self.daten_aufteilen()
        self.bundesland_mit_bestem_wasser_finden()
        self.wertebereiche_vergleichen()
        self.korrelationsanalyse()
        self.visualisiere_bundeslaender_vergleich()
        self.visualisiere_eigenschaften_boxplots()
        self.visualisiere_bundesland_eigenschaften()
        print("\nAnalyse abgeschlossen. Ergebnisse wurden im Ordner 'ergebnisse' und 'plots' gespeichert.")


# Hauptausführung, wenn die Datei direkt ausgeführt wird
if __name__ == "__main__":
    analyse = WasserQualitaetsAnalyse('water_potability.csv')
    analyse.fuehre_gesamtanalyse_durch()
