"""
Modulo di gestione degli errori per la pipeline della tesi
(predizione del deterioramento clinico in ICU tramite ML).

Da importare nei notebook con:
    from gestione_errori_tesi import *

Le funzioni sono organizzate per fase della pipeline:
    1. Caricamento file
    2. Validazione dati
    3. Controllo data leakage
    4. Training e valutazione modelli

Adatta i valori (range fisiologici, nomi colonne) alla tua pipeline reale:
sono impostati con valori plausibili ma generici.
"""

import logging
from pathlib import Path
from functools import wraps


# ============================================================
# 1. LOGGING - configurazione centralizzata
# ============================================================
# Usiamo il logging invece dei print: i messaggi restano su file
# anche se il notebook viene chiuso, e si distinguono per gravita
# (INFO, WARNING, ERROR) invece di essere tutti uguali.

def configura_logging(nome_log="pipeline_tesi", livello=logging.INFO, cartella_log="logs"):
    """
    Configura un logger che scrive sia su file (logs/pipeline_tesi.log)
    sia in console. Da chiamare una sola volta all'inizio del notebook.
    """
    Path(cartella_log).mkdir(exist_ok=True)  # crea la cartella se non esiste
    logger = logging.getLogger(nome_log)
    logger.setLevel(livello)

    if logger.handlers:
        # evita di aggiungere handler duplicati se la cella viene rieseguita
        return logger

    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    handler_file = logging.FileHandler(Path(cartella_log) / f"{nome_log}.log", encoding="utf-8")
    handler_file.setFormatter(formato)
    logger.addHandler(handler_file)

    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formato)
    logger.addHandler(handler_console)

    return logger


logger = configura_logging()


# ============================================================
# 2. ECCEZIONI PERSONALIZZATE - una gerarchia per fase
# ============================================================
# Usare eccezioni specifiche (invece di Exception generica) permette
# di scrivere: except ErroreCaricamentoFile: ... senza catturare
# per sbaglio anche errori di battitura nel codice (NameError, TypeError...).

class ErrorePipelineTesi(Exception):
    """Classe base: tutte le eccezioni della pipeline ereditano da questa."""
    pass


class ErroreCaricamentoFile(ErrorePipelineTesi):
    """File o cartella non trovati, formato non leggibile."""
    pass


class ErroreValidazioneDati(ErrorePipelineTesi):
    """Colonne mancanti, valori fuori range fisiologico, tipi errati."""
    pass


class ErroreDataLeakage(ErrorePipelineTesi):
    """Sovrapposizione di pazienti tra split train/val/test."""
    pass


class ErroreModello(ErrorePipelineTesi):
    """Errori durante training o valutazione dei modelli ML."""
    pass


# ============================================================
# 3. CARICAMENTO FILE - fase 1 (bloccante)
# ============================================================

def trova_cartella_dati(cartella_radice, nome_target, max_profondita=4):
    """
    Cerca ricorsivamente una cartella con un certo nome sotto cartella_radice.
    Utile per dataset scaricati con struttura a cartelle annidate duplicate
    (es. HiRID: la cartella reale e' dentro un'altra con lo stesso nome).

    Solleva ErroreCaricamentoFile se non trova nulla entro max_profondita livelli.
    """
    radice = Path(cartella_radice)
    if not radice.exists():
        raise ErroreCaricamentoFile(f"La cartella radice non esiste: {radice}")

    for profondita in range(max_profondita + 1):
        pattern = "*/" * profondita + nome_target
        trovate = list(radice.glob(pattern))
        if trovate:
            logger.info(f"Cartella '{nome_target}' trovata a profondita {profondita}: {trovate[0]}")
            return trovate[0]

    raise ErroreCaricamentoFile(
        f"Cartella '{nome_target}' non trovata sotto {radice} (cercato fino a {max_profondita} livelli)"
    )


def carica_file_sicuro(percorso, funzione_caricamento, **kwargs):
    """
    Wrapper generico per caricare un file (es. pd.read_csv, pd.read_parquet).
    funzione_caricamento e' la funzione da usare per leggerlo.
    Distingue tra file mancante e file presente ma corrotto/malformato,
    cosi' il messaggio d'errore dice subito qual e' il problema.
    """
    percorso = Path(percorso)
    if not percorso.exists():
        raise ErroreCaricamentoFile(f"File non trovato: {percorso}")

    try:
        dati = funzione_caricamento(percorso, **kwargs)
    except (ValueError, UnicodeDecodeError) as e:
        raise ErroreCaricamentoFile(f"File presente ma non leggibile ({percorso}): {e}") from e

    if dati is None or len(dati) == 0:
        raise ErroreCaricamentoFile(f"File caricato ma vuoto: {percorso}")

    logger.info(f"Caricato {percorso.name}: {len(dati)} righe")
    return dati


# ============================================================
# 4. VALIDAZIONE DATI - fase 2 (di solito non bloccante)
# ============================================================

# Range fisiologici plausibili: ADATTA questi valori alle tue feature reali
RANGE_FISIOLOGICI = {
    "frequenza_cardiaca": (20, 300),
    "frequenza_respiratoria": (0, 80),
    "spo2": (50, 100),
    "pressione_sistolica": (30, 300),
    "temperatura": (25, 45),
}


def valida_colonne_richieste(df, colonne_richieste):
    """
    Controlla che tutte le colonne richieste siano presenti nel DataFrame.
    Questo e' bloccante: se manca anche una sola colonna, il codice dopo
    fallirebbe comunque in modo piu' confuso (KeyError sparsi nel notebook).
    """
    mancanti = set(colonne_richieste) - set(df.columns)
    if mancanti:
        raise ErroreValidazioneDati(f"Colonne mancanti nel DataFrame: {sorted(mancanti)}")


def valida_range_fisiologico(df, colonna, azione="log"):
    """
    Controlla che i valori di una colonna rientrino nel range fisiologico plausibile.

    azione="log"      -> logga un warning col conteggio di valori anomali, tiene le righe
    azione="rimuovi"  -> rimuove le righe con valori fuori range, restituisce il df pulito
    azione="solleva"  -> interrompe con ErroreValidazioneDati
    """
    if colonna not in RANGE_FISIOLOGICI:
        logger.warning(f"Nessun range definito per '{colonna}', controllo saltato")
        return df

    minimo, massimo = RANGE_FISIOLOGICI[colonna]
    fuori_range = ~df[colonna].between(minimo, massimo) & df[colonna].notna()
    n_anomali = fuori_range.sum()

    if n_anomali == 0:
        return df

    messaggio = f"{colonna}: {n_anomali} valori fuori range [{minimo}, {massimo}]"

    if azione == "solleva":
        raise ErroreValidazioneDati(messaggio)
    elif azione == "rimuovi":
        logger.warning(messaggio + " -> righe rimosse")
        return df[~fuori_range]
    else:  # "log"
        logger.warning(messaggio + " -> valori mantenuti, controllare manualmente")
        return df


# ============================================================
# 5. CONTROLLO DATA LEAKAGE - fase 3 (bloccante)
# ============================================================

def verifica_split_senza_leakage(id_train, id_val, id_test, colonna_id="patient_id"):
    """
    Verifica che non ci siano pazienti condivisi tra i tre split.
    Questo controllo e' volutamente bloccante: un leakage qui invalida
    tutte le metriche calcolate a valle, quindi va scoperto subito.
    """
    train_val = set(id_train) & set(id_val)
    train_test = set(id_train) & set(id_test)
    val_test = set(id_val) & set(id_test)

    if train_val or train_test or val_test:
        dettagli = f"train∩val={len(train_val)}, train∩test={len(train_test)}, val∩test={len(val_test)}"
        raise ErroreDataLeakage(f"Leakage rilevato su '{colonna_id}': {dettagli}")

    logger.info("Nessun leakage tra i pazienti: split train/val/test verificato")


# ============================================================
# 6. TRAINING E VALUTAZIONE - fase 4 (di solito non bloccante)
# ============================================================

def gestisci_errori(eccezioni=(Exception,), azione="log", valore_default=None):
    """
    Decorator da applicare alle funzioni di training/valutazione.

    azione="log"      -> logga l'errore e restituisce valore_default (la pipeline continua)
    azione="solleva"  -> logga l'errore e lo rilancia come ErroreModello (la pipeline si ferma)

    Esempio:
        @gestisci_errori(eccezioni=(ValueError,), azione="log", valore_default=float("nan"))
        def calcola_auc(y_true, y_score):
            return roc_auc_score(y_true, y_score)
    """
    def decoratore(funzione):
        @wraps(funzione)
        def wrapper(*args, **kwargs):
            try:
                return funzione(*args, **kwargs)
            except eccezioni as e:
                logger.error(f"Errore in '{funzione.__name__}': {e}")
                if azione == "solleva":
                    raise ErroreModello(f"Errore in {funzione.__name__}: {e}") from e
                return valore_default
        return wrapper
    return decoratore


# ============================================================
# ESEMPIO D'USO IN UNA CELLA DEL NOTEBOOK
# ============================================================
#
# from gestione_errori_tesi import *
#
# ### Caricamento dati HiRID: cerchiamo la cartella reale (nidificata)
# try:
#     cartella_raw = trova_cartella_dati("dati/hirid", "raw_stage")
#     df = carica_file_sicuro(cartella_raw / "observation_tables.csv", pd.read_csv)
# except ErroreCaricamentoFile as e:
#     logger.error(f"Impossibile procedere: {e}")
#     raise  # qui e' giusto interrompere: senza dati non si va avanti
#
# ### Validazione: colonne richieste + range fisiologici
# valida_colonne_richieste(df, ["patient_id", "frequenza_cardiaca", "spo2"])
# df = valida_range_fisiologico(df, "frequenza_cardiaca", azione="rimuovi")
#
# ### Controllo leakage dopo lo split train/val/test
# verifica_split_senza_leakage(id_train, id_val, id_test)
#
# ### Training con decorator: se roc_auc_score fallisce su una fold,
# ### logga l'errore e restituisce NaN invece di far crashare il ciclo
# @gestisci_errori(eccezioni=(ValueError,), azione="log", valore_default=float("nan"))
# def calcola_auc(y_true, y_score):
#     return roc_auc_score(y_true, y_score)
