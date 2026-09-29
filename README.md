# Compression d'ECG par DCT

Compression d'un signal ECG (base PTB-XL) en appliquant une DCT unidimensionnelle et en mettant à zéro les coefficients de plus faible énergie.

## Fichiers du projet

| Fichier | Rôle |
|---|---|
| `ecg_dct.py` | Fonctions communes : DCT, DCT inverse, mise à zéro, tailles de fichiers (ne se lance pas seul) |
| `compresser.py` | ECG d'origine → fichier compressé `.npz` + graphiques + comparaison de taille |
| `decompresser.py` | Fichier compressé → ECG reconstruit `.hea`/`.dat` + graphiques + erreur + comparaison de taille |

## Installation (une seule fois)

```bash
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install wfdb numpy scipy matplotlib
```

ECG de test à placer dans le dossier du projet :

```bash
wget https://physionet.org/files/ptb-xl/1.0.1/records100/00000/00001_lr.hea
wget https://physionet.org/files/ptb-xl/1.0.1/records100/00000/00001_lr.dat
```

## Utilisation

```bash
source venv/bin/activate

# 1. Compresser
python compresser.py 00001_lr

# 2. Reconstruire (optionnel)
python decompresser.py sortie/compresse/00001_lr.npz
```

Sans interface graphique, préfixer par `MPLBACKEND=Agg` (les PNG sont générés dans tous les cas).

## Options

| Option | Script | Effet | Défaut |
|---|---|---|---|
| `--energie 0.995` | compresser | Fraction d'énergie conservée | 0.999 |
| `--pas 0.001` | compresser | Pas de quantification (mV) | 0.001 |
| `--original chemin` | decompresser | ECG d'origine pour comparaison | cherché dans le dossier courant |
| `--rapide` | les deux | DCT rapide (scipy) au lieu de la boucle manuelle | manuelle |
| `--sortie dossier` | les deux | Dossier de sortie | `sortie` |

> Pour les ECG à 500 Hz (`_hr`), utiliser `--rapide` : la boucle manuelle prend plusieurs minutes.

## Résultats

```
sortie/
├── compresse/
│   └── 00001_lr.npz                    ECG compressé
├── reconstruit/
│   ├── 00001_lr_reconstruit.hea        ECG reconstruit (format PTB-XL,
│   └── 00001_lr_reconstruit.dat        lisible par wfdb)
└── graphiques/
    └── 00001_lr/
        ├── compression/                signal d'origine + coefficients DCT
        └── reconstruction/             original vs reconstruit + erreur
```

## Principe du format compressé

1. DCT de chaque dérivation.
2. Mise à zéro des coefficients les plus faibles (critère d'énergie).
3. Quantification : les coefficients gardés sont arrondis au pas choisi et stockés en entiers.
4. Compression sans perte du tableau (`np.savez_compressed`) : les zéros ne prennent presque plus de place.

La reconstruction redonne un fichier de **même taille que l'original** : la compression sert au stockage et à la transmission ; on décompresse uniquement pour relire le signal.

## Qualité

Avec la DCT orthonormée, l'erreur relative vaut : `PRD ≈ 100 × √(1 − énergie conservée)`.

| Énergie conservée | PRD | Qualité |
|---|---|---|
| 99,9 % | 3,2 % | Très bonne |
| 99,5 % | 7,1 % | Bonne |
| 99 % | 10 % | Limite |
