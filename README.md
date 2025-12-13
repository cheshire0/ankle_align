# AnkleAlign – Bokapozíció osztályozás mélytanulással

## Projekt leírása

Az **AnkleAlign** projekt célja bokáról készült képek automatikus osztályozása három kategóriába:

* **Pronáció**
* **Neutralis**
* **Szupináció**

A projekt egy teljes gépi tanulási pipeline-t valósít meg, amely magában foglalja az adatok előfeldolgozását, a modell tanítását, valamint a végső kiértékelést. A megoldás CPU-n is futtatható, Docker környezetben reprodukálható módon.


## Könyvtárstruktúra

```
ankle_align/
├── src/
│   ├── 01_data_processing.py   # Adatok feldolgozása, tisztítása, split
│   ├── 02_train.py             # Alap tanító script
│   ├── 02_train_v2.py          # Javított modell (GAP CNN)
│   ├── 03_evaluation.py        # Tesztkészlet kiértékelése
│   └── api.py                  # (Opcionális) inference API
├── data/
│   ├── raw/                    # Nyers képek (Neptun kódok szerint)
│   ├── processed/              # Feldolgozott képek + metadata.csv
│   └── splits/                 # Train / val / test split információk
├── output/
│   ├── model_best_v2.pt        # Legjobb modell
│   ├── train_history_v2.csv    # Tanítási napló
│   ├── test_metrics.csv        # Teszt metrikák
│   ├── confusion_matrix.csv   # Konfúziós mátrix
│   └── confusion_matrix.png   # Konfúziós mátrix ábra
├── requirements.txt
├── Dockerfile
└── README.md
```


## Környezet és futtatás

A projekt **Docker Desktop** segítségével futtatható Windows alatt is.

### Docker image buildelése

```powershell
docker build -t anklealign:latest .
```


## 1. Adat-előfeldolgozás

A nyers képek feldolgozása, ellenőrzése és felosztása train / validation / test halmazokra.

```powershell
docker run --rm `
  -v C:\anklealign\data:/data `
  -v C:\anklealign\output:/app/output `
  anklealign:latest `
  python -u /app/src/01_data_processing.py --clean
```

**Megjegyzések:**

* A split rétegzett (stratified), osztályarány-megőrző.


## 2. Modell tanítása (v2)

A javított modell egy **Global Average Pooling (GAP)** alapú CNN, amely:

* kevesebb paraméterrel dolgozik,
* jobban generalizál kis adathalmazon,
* osztálysúlyozott veszteségfüggvényt használ,
* korai leállítást (early stopping) alkalmaz.

```powershell
docker run --rm `
  -v C:\anklealign\data:/data `
  -v C:\anklealign\output:/app/output `
  anklealign:latest `
  python -u /app/src/02_train_v2.py --epochs 80 --patience 10
```


## 3. Kiértékelés

A végső modell értékelése **kizárólag a teszthalmazon** történik.

```powershell
docker run --rm `
  -v C:\anklealign\data:/data `
  -v C:\anklealign\output:/app/output `
  anklealign:latest `
  python -u /app/src/03_evaluation.py
```

### Kiértékelési kimenetek

* `test_metrics.csv` – precision, recall, F1 osztályonként
* `confusion_matrix.csv`
* `confusion_matrix.png`


## Módszertani megjegyzések

* Nem alkalmazunk rotációs adataugmentációt.
* Az alacsony teljesítmény fő oka:

  * kis adatmennyiség,
  * osztályegyensúly hiánya,
  * részben heurisztikus címkézés.
* A projekt célja nem a maximális pontosság, hanem egy **helyes, reprodukálható ML pipeline** bemutatása.


## Következtetések

A projekt demonstrálja, hogy:

* a teljes gépi tanulási folyamat korrekt módon megvalósítható,
* az adatok minősége kritikusabb, mint a modell komplexitása,
* a korlátok megfelelő elemzése és dokumentálása kulcsfontosságú.


## Szerző

Projekt: **AnkleAlign**

Készítette: *Sebők Lili*

Környezet: Mélytanulás - BMEVITMMA19 2025/26/1