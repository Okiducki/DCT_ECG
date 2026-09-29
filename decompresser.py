"""
decompresser.py : reconstruit un ECG à partir du fichier compressé (.npz)
=========================================================================

Étapes :
    1. Lecture du fichier compressé
    2. DCT inverse de chaque dérivation
    3. Écriture de l'ECG reconstruit au format d'origine (.hea + .dat),
       lisible par wfdb comme n'importe quel ECG PTB-XL
    4. Si l'ECG d'origine est disponible : graphiques + calcul de l'erreur
    5. Comparaison des tailles : origine / compressé / reconstruit

Utilisation :
    python decompresser.py                                   # sortie/compresse/00001_lr.npz
    python decompresser.py sortie/compresse/00001_hr.npz --original ecg/00001_hr --rapide

Résultats dans le dossier sortie/ :
    sortie/reconstruit/00001_lr_reconstruit.hea / .dat   <- l'ECG reconstruit
    sortie/graphiques/00001_lr/reconstruction/*.png       <- un graphique par dérivation
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wfdb

import ecg_dct as ed

################################################################################
# PARAMÈTRES
################################################################################

parser = argparse.ArgumentParser(description="Reconstruction d'un ECG compressé par DCT")
parser.add_argument("compresse", nargs="?", default=os.path.join("sortie", "compresse", "00001_lr.npz"),
                    help="fichier compressé .npz (défaut : sortie/compresse/00001_lr.npz)")
parser.add_argument("--original", default=None,
                    help="chemin de l'ECG d'origine SANS extension, pour la comparaison "
                         "(défaut : cherché dans le dossier courant)")
parser.add_argument("--rapide", action="store_true",
                    help="utiliser la DCT inverse rapide (scipy) au lieu de la boucle manuelle")
parser.add_argument("--sortie", default="sortie", help="dossier de sortie (défaut : sortie)")
args = parser.parse_args()

idct = ed.idct_rapide if args.rapide else ed.idct_manuelle

################################################################################
# ÉTAPE 1 : lecture du fichier compressé
################################################################################

donnees = np.load(args.compresse)
coefficients = donnees["coefficients"] * float(donnees["pas"])   # entiers -> mV
fs = float(donnees["fs"])
sig_name = list(donnees["sig_name"])
units = list(donnees["units"])
adc_gain = list(donnees["adc_gain"])
nom = str(donnees["nom_original"])

nSamples, nSignals = coefficients.shape
t = np.arange(nSamples) / fs

print(f"Fichier compressé : {args.compresse}")
print(f"ECG : {nom}  |  {nSignals} dérivations  |  {nSamples} échantillons  |  {fs:g} Hz")
print(f"Méthode DCT inverse : {'rapide (scipy)' if args.rapide else 'manuelle (boucles)'}\n")

################################################################################
# ÉTAPE 2 : DCT inverse
################################################################################

reconstruit = np.empty([nSamples, nSignals])
for iSignal in range(nSignals):
    reconstruit[:, iSignal] = idct(coefficients[:, iSignal])
    print(f"  {iSignal:2d} {sig_name[iSignal]:>4} : reconstruit")

################################################################################
# ÉTAPE 3 : écriture au format WFDB (.hea + .dat), comme l'original
################################################################################

dossier_reconstruit = os.path.join(args.sortie, "reconstruit")
os.makedirs(dossier_reconstruit, exist_ok=True)
nom_reconstruit = nom + "_reconstruit"

wfdb.wrsamp(
    nom_reconstruit,
    fs=fs,
    units=units,
    sig_name=sig_name,
    p_signal=reconstruit,
    fmt=["16"] * nSignals,              # entiers 16 bits, comme PTB-XL
    adc_gain=adc_gain,                  # même résolution que l'original
    baseline=[0] * nSignals,
    write_dir=dossier_reconstruit,
)
chemin_reconstruit = os.path.join(dossier_reconstruit, nom_reconstruit)

################################################################################
# ÉTAPE 4 : comparaison avec l'original (s'il est disponible)
################################################################################

chemin_original = args.original if args.original else nom
original_dispo = os.path.exists(chemin_original + ".hea") and os.path.exists(chemin_original + ".dat")

if original_dispo:
    original = np.nan_to_num(wfdb.rdrecord(chemin_original).p_signal)
    # on relit le fichier écrit, pour mesurer l'erreur réelle du fichier final
    relu = wfdb.rdrecord(chemin_reconstruit).p_signal

    dossier_graph = os.path.join(args.sortie, "graphiques", nom, "reconstruction")
    os.makedirs(dossier_graph, exist_ok=True)

    print(f"\n{'Dérivation':>10} | {'PRD':>7} | {'Erreur max':>10}")
    for iSignal in range(nSignals):
        x = original[:, iSignal]
        xr = relu[:, iSignal]
        erreur = x - xr
        p = ed.prd(x, xr)
        print(f"{sig_name[iSignal]:>10} | {p:5.2f} % | {np.max(np.abs(erreur)):7.4f} mV")

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"Reconstruction — {nom} — dérivation {sig_name[iSignal]}",
                     fontsize=14, fontweight="bold")

        # --- Graphique 1 : superposition original / reconstruit
        axs[0].plot(t, x, color="tab:blue", lw=1.5, label="ECG d'origine")
        axs[0].plot(t, xr, "--", color="tab:green", lw=1.2, label="ECG reconstruit")
        axs[0].set_title(f"1) Original et reconstruit superposés  (PRD = {p:.2f} %)")
        axs[0].set_xlabel("Temps (s)")
        axs[0].set_ylabel("Amplitude (mV)")
        axs[0].grid(alpha=0.3)
        axs[0].legend(loc="upper right")

        # --- Graphique 2 : erreur
        axs[1].plot(t, erreur, color="tab:purple", lw=0.8,
                    label="Erreur = original − reconstruit")
        axs[1].axhline(0, color="black", lw=0.5)
        axs[1].set_title("2) Erreur de reconstruction (ce qui a été perdu)")
        axs[1].set_xlabel("Temps (s)")
        axs[1].set_ylabel("Erreur (mV)")
        axs[1].grid(alpha=0.3)
        axs[1].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(dossier_graph, f"signal_{iSignal:02d}_{sig_name[iSignal]}.png"),
                    dpi=120)
        plt.close()
else:
    print(f"\nECG d'origine introuvable ({chemin_original}.hea / .dat) : "
          "pas de graphiques ni de calcul d'erreur.")
    print("Utilise --original chemin/vers/00001_lr pour l'indiquer.")

################################################################################
# ÉTAPE 5 : comparaison des tailles
################################################################################

lignes = []
if original_dispo:
    lignes.append(("ECG d'origine (.hea + .dat)",
                   ed.taille(chemin_original + ".hea", chemin_original + ".dat")))
lignes.append(("ECG compressé (.npz)", ed.taille(args.compresse)))
lignes.append(("ECG reconstruit (.hea + .dat)",
               ed.taille(chemin_reconstruit + ".hea", chemin_reconstruit + ".dat")))

ed.afficher_comparaison("COMPARAISON DES TAILLES", lignes)

print(f"\nECG reconstruit : {chemin_reconstruit}.hea / .dat")
if original_dispo:
    print(f"Graphiques      : {dossier_graph}/")
