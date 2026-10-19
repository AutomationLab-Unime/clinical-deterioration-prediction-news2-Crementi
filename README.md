# Predizione del deterioramento clinico con NEWS-2 e Delta-NEWS

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189FDD)](https://xgboost.ai)
[![Dati](https://img.shields.io/badge/dati-PhysioNet-8A2BE2)](https://physionet.org)
[![HL7 FHIR](https://img.shields.io/badge/HL7-FHIR-E34F26)](https://hl7.org/fhir/)

Un modello di machine learning che stima il rischio di **deterioramento clinico nelle 24 ore successive** a partire
dai parametri vitali di routine sintetizzati nel punteggio **NEWS-2** e dalla sua evoluzione nel tempo (**Delta-NEWS**).
Il modello è validato su tre database di terapia intensiva e su dati in formato HL7 FHIR.

Codice della tesi triennale in Ingegneria Biomedica
**"Progettazione di un modulo di interoperabilità per la sorveglianza automatizzata in terapia sub-intensiva"**
Autrice: Mara Crementi, a.a. 2025/2026
Relatore: Luca Patané

---

## Indice
- [Panoramica del progetto](#panoramica-del-progetto)
- [Dataset](#dataset)
- [Struttura della cartella](#struttura-della-cartella)
- [Requisiti e installazione](#requisiti-e-installazione)
- [Come eseguire il progetto](#come-eseguire-il-progetto)
- [Risultati e performance](#risultati-e-performance)
- [Citazioni](#citazioni)

---

## Panoramica del progetto
L'obiettivo è un modello di sorveglianza automatizzata che sia **interoperabile**, cioè applicabile con la stessa
metodologia a basi dati cliniche con schemi, codifiche e unità di misura diverse. Le variabili, l'endpoint e i modelli
sono gli stessi per tutti e tre i database.

* **Tipo di problema:** classificazione binaria (deterioramento entro 24 ore sì / no), valutata su blocchi di 10 minuti.
* **Endpoint composito:** insorgenza di almeno uno di questi eventi nelle 24 ore successive: morte in terapia intensiva,
  inizio di un vasopressore, intubazione.
* **Feature engineering:** 20 variabili ricavate dai parametri NEWS-2:
  * frequenza respiratoria, SpO2, pressione sistolica, frequenza cardiaca, temperatura, ossigeno supplementare;
  * NEWS-2, con la coscienza ottenuta dal GCS convertito in AVPU;
  * statistiche mobili su 6 ore (massimo, minimo, media, deviazione standard);
  * Delta-NEWS istantaneo e su 6 ore;
  * numero di parametri mancanti e sei indicatori di dato mancante.
* **Modelli testati:** Regressione Logistica, Random Forest e XGBoost.
* **Validazione:**
  * split per paziente 60/20/20 (seed 42), così che nessun paziente compaia in più di un insieme;
  * scelta del modello sul validation set e soglia di allarme con l'indice di Youden;
  * ricalibrazione isotonica;
  * intervallo di confidenza al 95% con bootstrap per paziente (1.000 ripetizioni).
* **Analisi aggiuntive:** SHAP, decision curve analysis, confronto con il NEWS-2 da solo, tempo di anticipo degli allarmi,
  consensus dei tre modelli (ensemble).
* **Interoperabilità HL7 FHIR:** nel notebook `04`, le stesse predizioni vengono ottenute partendo dai dati in formato FHIR,
  con i codici dei parametri vitali tradotti in LOINC.

---

## Dataset

| Database | Ruolo nella tesi | Usati | Totale nel database | Quota |
| :--- | :--- | :--- | :--- | :--- |
| [HiRID 1.1.1](https://physionet.org/content/hirid/1.1.1/) (Berna, CH) | database di riferimento | 33.897 pazienti (tutti quelli con parametri vitali) | 33.905 pazienti | 99,9% |
| [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) (Boston, USA) | confronto principale | 20.000 pazienti (campione casuale, seed 42) | 65.366 pazienti con almeno un ricovero in terapia intensiva | 30,6% |
| [eICU-CRD 2.0](https://physionet.org/content/eicu-crd/2.0/) (multicentrico, USA) | confronto secondario | 20.000 ricoveri (campione casuale, seed 42) di 19.046 persone | 192.831 ricoveri con parametri vitali in `vitalPeriodic` (200.859 ricoveri in totale, di 139.367 persone) | 10,4% |

I dati non sono inclusi: MIMIC-IV, eICU-CRD e HiRID sono ad accesso credenziato su PhysioNet e vanno scaricati
dai link qui sopra, previa autorizzazione. Vanno messi nella cartella principale del progetto così:

```text
mimic-iv-3.1/                                    # contiene hosp/ e icu/
eicu-collaborative-research-database-2.0/        # contiene patient.csv.gz, vitalPeriodic.csv.gz, ...
hirid-a-high-time-resolution-icu-dataset-1.1.1/  # contiene raw_stage/, reference_data/ (sottocartelle trovate in automatico)
```

La prova FHIR usa invece due demo ad **accesso aperto** (licenza ODbL, 100 pazienti):
- [MIMIC-IV Clinical Database Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/), da mettere in `mimic-iv-demo-csv/mimic-iv-clinical-database-demo-2.2/`
- [MIMIC-IV Clinical Database Demo on FHIR 2.1.0](https://physionet.org/content/mimic-iv-demo-fhir/2.1.0/), da mettere in `mimic-iv-fhir-demo/mimic-iv-clinical-database-demo-on-fhir-2.1.0/`

---

## Struttura della cartella
```text
news2-deltanews-icu/
│
├── 01_HiRID.ipynb                  # Fase 1 (preprocessing) + Fase 2 (modello) su HiRID
├── 02_MIMIC-IV.ipynb               # Fase 1 + Fase 2 su MIMIC-IV
├── 03_eICU-CRD.ipynb               # Fase 1 + Fase 2 su eICU-CRD
├── 04_prova_FHIR_MIMIC-IV.ipynb    # Prova di interoperabilità HL7 FHIR (demo MIMIC-IV)
├── gestione_errori_tesi.py         # Controlli (caricamento, validazione, data leakage) e logging
│
├── models/                         # Modelli addestrati (.pkl): classificatore, scaler, ricalibrazione isotonica
├── outputs/                        # Figure (ROC, PR, calibrazione, DCA, SHAP, diagnostica) e tabelle consensus
├── data_hirid/                     # Dataset train/validation/test scritti dalla Fase 1 di HiRID (non pubblicati)
├── data_mimic/  data_eicu/         # Cartelle create dai notebook (MIMIC-IV ed eICU-CRD lavorano in memoria)
├── logs/                           # Log della pipeline
│
├── requirements.txt                # Librerie Python necessarie
├── .gitignore                      # Esclude dati e file temporanei
└── README.md                       # Documentazione del progetto
```

I notebook contengono gli output dell'ultima esecuzione: grafici, metriche e tabelle.

---

## Requisiti e installazione

Il progetto è stato sviluppato con **Python 3.14** su Windows 11. Dopo aver scaricato il repository si installano le dipendenze:

```bash
# Crea e attiva un ambiente virtuale (consigliato)
python -m venv .venv
.venv\Scripts\activate      # Su macOS/Linux: source .venv/bin/activate

# Installa i pacchetti richiesti
pip install -r requirements.txt
```

**Nota sulle versioni:**
* **MIMIC-IV ed eICU-CRD:** eseguiti con pandas 2.3.3. Con pandas 3 lo split per persona di eICU-CRD si interrompe
  con un `TypeError`, perché gli identificativi diventano array Arrow.
* **HiRID:** eseguito con pandas 3.0.2, gira anche con la 2.3.3.

---

## Come eseguire il progetto

1. **Addestrare e valutare il modello su un database.**
   Si apre il notebook del database (`01`, `02` o `03`) e lo si esegue dall'inizio alla fine (`Kernel > Restart & Run All`),
   con la cartella del progetto come cartella di lavoro.
   * I notebook sono indipendenti tra loro.
   * La Fase 1 legge i dati grezzi e costruisce le 20 variabili e l'etichetta.
   * La Fase 2 addestra i modelli e salva modelli e figure in `models/` e `outputs/`.
   * La Fase 1 legge tabelle molto grandi (`chartevents` di MIMIC-IV, le partizioni parquet di HiRID): servono molta RAM
     e diverse ore per HiRID.

2. **Prova di interoperabilità FHIR.**
   Si esegue `04_prova_FHIR_MIMIC-IV.ipynb`.
   * Bastano le due demo aperte e i modelli MIMIC-IV già inclusi in `models/`.
   * Non serve rieseguire il notebook `02`.

3. **Usare un modello addestrato (inferenza).**
   ```python
   import joblib

   FEATURES = [
       "freq_respiratoria", "spo2", "pressione_sistolica", "freq_cardiaca", "temperatura",
       "ossigeno_supplementare", "news2",
       "news2_max_6h", "news2_min_6h", "news2_media_6h", "news2_std_6h",
       "delta_news", "delta_news_6h", "n_parametri_mancanti",
       "freq_respiratoria_mancante", "spo2_mancante", "pressione_sistolica_mancante",
       "freq_cardiaca_mancante", "temperatura_mancante", "coscienza_gcs_mancante",
   ]

   modello = joblib.load("models/modello_news2_mimic.pkl")      # XGBoost scelto su MIMIC-IV
   calibratore = joblib.load("models/isotonic_mimic.pkl")       # ricalibrazione isotonica

   # X: DataFrame con le 20 feature, calcolate come nella Fase 1 (una riga = un blocco di 10 minuti)
   prob = modello.predict_proba(X[FEATURES])[:, 1]
   rischio = calibratore.predict(prob)                          # probabilità di deterioramento entro 24 ore
   ```
   Il notebook `04` mostra come calcolare le 20 feature a partire da dati grezzi (CSV o FHIR).

---

## Risultati e performance

I modelli sono stati valutati sul test set (20% dei pazienti, mai visti in addestramento):

| Database | Modello scelto | AUC [IC 95%] | AUPRC | Brier (dopo isotonica) | NEWS-2 da solo (AUC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HiRID | Random Forest | **0,722** [0,711–0,733] | 0,339 | 0,135 | 0,688 |
| MIMIC-IV | XGBoost | **0,740** [0,729–0,750] | 0,344 | 0,122 | 0,665 |
| eICU-CRD | XGBoost | **0,743** [0,729–0,760] | 0,357 | 0,119 | 0,650 |

* In tutti e tre i database il modello supera il NEWS-2 usato da solo.
* **Prova FHIR:** sui 100 pazienti della demo, le 20 variabili e le probabilità predette ottenute dai dati FHIR sono
  **identiche** a quelle ottenute dai CSV originali (differenza 0).
* Le tabelle complete del consensus sono in `outputs/consensus_*.csv`.

---

## Citazioni
Se si usano i dati, vanno citati i dataset originali secondo le indicazioni di PhysioNet:
- Johnson A. et al., *MIMIC-IV* (versione 3.1), PhysioNet.
- Pollard T. et al., *The eICU Collaborative Research Database*, Scientific Data, 2018.
- Hyland S. et al., *Early prediction of circulatory failure in the intensive care unit using machine learning*, Nature Medicine, 2020 (HiRID).
- Goldberger A. et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation, 2000.
