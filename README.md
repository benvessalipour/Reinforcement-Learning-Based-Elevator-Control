# Aufzugsteuerung mittels Reinforcement Learning

Dieses Projekt implementiert eine Aufzugsteuerung mit Hilfe von Q-Learning. Es simuliert eine realitätsnahe Umgebung mit mehreren Etagen, zufälligen Fahrgästen und einer handgebauten Referenzstrategie. Ziel ist es, durch RL eine bessere Steuerungsstrategie zu lernen.

## Projektstruktur

```
project/
│
├── learning.py                # Q-Learning mit g1 und g2
├── reference.py               # Referenzstrategie (klassisch heuristisch)
├── Environment/               # Simulierte Aufzugsumgebung
│   ├── environment.py         # Zustände, Aktionen, Step-Funktion
│   └── constants.py           # Definition von Richtungen, Aktionen, etc.
│   └── policy.py              # Definition und Auswahl von Strategien
├── comparison_learning_curve.png     # Lernkurvenvergleich g1 vs. g2
├── reference_learning_curve.png      # Lernkurve der Referenzstrategie
└── README.md                 # Diese Datei
```

## Methodik

Das Projekt nutzt tabellarisches Q-Learning. Die Zustände wurden stark vereinfacht, um die Größe des Zustandsraums zu reduzieren. Zwei Ein-Schritt-Kostenfunktionen (`g1`, `g2`) wurden definiert, um Belohnungen für verschiedene Verhaltensweisen zu vergeben:

- `g1`: einfache Belohnung für Auslieferung, Schrittstrafe
- `g2`: differenzierte Belohnung (Tür öffnen, Richtungswahl, Vermeidung unnötiger Aktionen)

## Referenzstrategie

Die Referenzstrategie (`reference_policy`) modelliert das Verhalten eines klassischen Aufzugs:

- fährt in Richtung nächstem Ziel (Cabin- oder Call-Button)
- öffnet Türen nur wenn nötig
- stoppt bei Bedarf unterwegs

Diese Strategie ist deterministisch und nicht lernbasiert. Sie dient als Vergleich für die Performance der gelernten Strategien.

## Training

- 3.000 Episoden mit je 400 Zeitschritten
- Trainingsparameter: `α = 0.1`, `γ = 0.95`, `ε` beginnend bei 0.3 mit linearer Reduktion
- Evaluation über Gesamtreward pro Episode sowie gleitender Durchschnitt

## Ergebnisse

Die Lernkurven zeigen:
- `g2` lernt differenzierteres Verhalten, erfordert aber mehr Episoden
- `g1` ist einfacher zu lernen, aber weniger effektiv
- Die RL-Strategien übertreffen die Referenzstrategie langfristig

## Visualisierung

- **`comparison_learning_curve.png`** zeigt die Belohnungsverläufe für `g1` vs. `g2`
- **`reference_learning_curve.png`** zeigt die Performance der Referenzstrategie

## Starten des Trainings

```bash
python learning.py
```

## Evaluierung der Referenzstrategie

```bash
python reference.py
```

## Demonstration

Um eine gelernte Policy in einer Episode zu demonstrieren, verwenden Sie:

```bash
python demonstration.py
```

## 📄 Bericht / Dokumentation

Der vollständige Projektbericht mit Methodik, Versuchsaufbau, Lernkurven und Ergebnisanalyse ist hier verfügbar:

👉 [Q-Learning Elevator control (PDF)](https://drive.google.com/file/d/1YxnPScZop35mV5LH4GEtGzOqSD5p0v47/view?usp=share_link)

Der Bericht enthält:
- mathematische Beschreibung des Q-Learning-Algorithmus
- die vereinfachte Zustandsrepräsentation
- die Definition der Kostenfunktionen `g1` und `g2`
- experimentelle Evaluation mit Hyperparameter-Variation
- Vergleich mit Referenzstrategie

Das Skript verwendet die beste bekannte Policy basierend auf dem Q-Table aus `learning.py` und zeigt exemplarisch den Ablauf einer Episode.
