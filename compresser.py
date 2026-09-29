"""
compresser.py : compresse un ECG PTB-XL par DCT
================================================

Étapes :
    1. Lecture de l'ECG d'origine (.hea + .dat)
    2. DCT de chaque dérivation
    3. Mise à zéro des coefficients avec la plus faible énergie
    4. Quantification (arrondi) des coefficients gardés
    5. Écriture du fichier compressé (.npz)
    6. Graphiques + comparaison de taille avec le fichier d'origine

Utilisation :
    python compresser.py                          # 00001_lr, 99,9 %
    python compresser.py 00001_lr --energie 0.995
    python compresser.py ecg/00001_hr --rapide    # 500 Hz : version rapide conseillée

Résultats dans le dossier sortie/ :
    sortie/compresse/00001_lr.npz                 <- le fichier compressé
    sortie/graphiques/00001_lr/compression/*.png  <- un graphique par dérivation
"""

import argparse
import os
import zlib

import matplotlib.pyplot as plt
import numpy as np
import wfdb

import ecg_dct as ed

################################################################################
# PARAMÈTRES (modifiables en ligne de commande)
################################################################################

parser = argparse.ArgumentParser(description="Compression d'un ECG par DCT")
parser.add_argument("enregistrement", nargs="?", default="00001_lr",
                    help="chemin de l'ECG SANS extension (défaut : 00001_lr)")
parser.add_argument("--energie", type=float, default=0.999,
                    help="fraction d'énergie conservée (défaut : 0.999 = 99,9 %%)")
parser.add_argument("--pas", type=float, default=0.001,
                    help="pas de quantification en mV (défaut : 0.001, la résolution d'origine)")
parser.add_argument("--rapide", action="store_true",
                    help="utiliser la DCT rapide (scipy) au lieu de la boucle manuelle")
parser.add_argument("--sortie", default="sortie", help="dossier de sortie (défaut : sortie)")
args = parser.parse_args()

nom = os.path.basename(args.enregistrement)
dossier_compresse = os.path.join(args.sortie, "compresse")
dossier_graph = os.path.join(args.sortie, "graphiques", nom, "compression")
os.makedirs(dossier_compresse, exist_ok=True)
os.makedirs(dossier_graph, exist_ok=True)

dct = ed.dct_rapide if args.rapide else ed.dct_manuelle

################################################################################
# ÉTAPE 1 : lecture de l'ECG d'origine
################################################################################

record = wfdb.rdrecord(args.enregistrement)

nSignals = record.n_sig
nSamples = record.sig_len
fs = record.fs
t = np.arange(nSamples) / fs

print(f"ECG : {nom}  |  {nSignals} dérivations  |  {nSamples} échantillons  |  "
      f"{fs} Hz  |  {nSamples / fs:.0f} s")
print(f"Énergie conservée : {args.energie * 100} %  |  Pas de quantification : {args.pas} mV")
print(f"Méthode DCT : {'rapide (scipy)' if args.rapide else 'manuelle (boucles)'}\n")

coeffs_quantifies = np.zeros([nSamples, nSignals])
resume = []

for iSignal in range(nSignals):
    x = np.nan_to_num(record.p_signal[:, iSignal])
    nom_derivation = record.sig_name[iSignal]

################################################################################
# ÉTAPES 2 et 3 : DCT puis mise à zéro des coefficients les plus faibles
################################################################################

    c = dct(x)
    c_zero, k, energie_reelle = ed.mise_a_zero(c, args.energie)

################################################################################
# ÉTAPE 4 : quantification
# Les coefficients sont des nombres à virgule (8 octets chacun). On les arrondit
# à un multiple du pas (0.001 mV par défaut, comme le fichier d'origine) pour
# les stocker sous forme d'entiers, beaucoup plus compacts.
################################################################################

    coeffs_quantifies[:, iSignal] = np.round(c_zero / args.pas)

    prd_theorique = 100 * np.sqrt(1 - energie_reelle)
    resume.append([nom_derivation, k, nSamples / k, prd_theorique])
    print(f"  {iSignal:2d} {nom_derivation:>4} : {k:5d} / {nSamples} coefficients gardés")

################################################################################
# GRAPHIQUE de la dérivation
################################################################################

    garde = c_zero != 0
    indices = np.arange(nSamples)

    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(f"Compression DCT — {nom} — dérivation {nom_derivation}",
                 fontsize=14, fontweight="bold")

    # --- Graphique 1 : le signal d'origine (domaine temporel)
    axs[0].plot(t, x, color="tab:blue", lw=1, label="Signal ECG d'origine")
    axs[0].set_title("1) Signal d'origine (domaine temporel)")
    axs[0].set_xlabel("Temps (s)")
    axs[0].set_ylabel("Amplitude (mV)")
    axs[0].grid(alpha=0.3)
    axs[0].legend(loc="upper right")

    # --- Graphique 2 : les coefficients DCT (domaine fréquentiel)
    axs[1].plot(indices[~garde], c[~garde], ".", color="tab:red", ms=3,
                label=f"Coefficients supprimés (mis à 0) : {nSamples - k}")
    axs[1].plot(indices[garde], c[garde], ".", color="black", ms=3,
                label=f"Coefficients conservés : {k}")
    axs[1].set_title("2) Coefficients de la DCT (domaine fréquentiel)", pad=35)
    axs[1].set_xlabel("Indice k du coefficient DCT (0 = moyenne du signal, "
                      f"{nSamples - 1} = fréquence maximale)")
    axs[1].set_ylabel("Valeur du coefficient (mV)")
    axs[1].grid(alpha=0.3)
    axs[1].legend(loc="upper right")
    ed.ajouter_axe_frequence(axs[1], fs, nSamples)

    axs[1].text(0.99, 0.05,
                f"Énergie conservée : {energie_reelle * 100:.2f} %\n"
                f"Taux de coefficients gardés : {100 * k / nSamples:.1f} %",
                transform=axs[1].transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(dossier_graph, f"signal_{iSignal:02d}_{nom_derivation}.png"),
                dpi=120)
    plt.close()

################################################################################
# ÉTAPE 5 : écriture du fichier compressé
# On choisit le plus petit type d'entier suffisant (int16 ou int32), puis
# np.savez_compressed compresse le tout : les nombreux zéros prennent
# alors très peu de place.
################################################################################

if np.max(np.abs(coeffs_quantifies)) < 32767:
    coeffs_quantifies = coeffs_quantifies.astype(np.int16)
else:
    coeffs_quantifies = coeffs_quantifies.astype(np.int32)

fichier_compresse = os.path.join(dossier_compresse, nom + ".npz")
np.savez_compressed(
    fichier_compresse,
    coefficients=coeffs_quantifies,          # coefficients gardés (entiers), zéros ailleurs
    pas=args.pas,                            # pour retrouver les valeurs en mV
    fs=fs,
    sig_name=np.array(record.sig_name),
    units=np.array(record.units),
    adc_gain=np.array(record.adc_gain),
    energie=args.energie,
    nom_original=nom,
)

################################################################################
# ÉTAPE 6 : résumé et comparaison des tailles
################################################################################

print(f"\n{'Dérivation':>10} | {'gardés':>6} | {'CR':>6} | {'PRD estimé':>10}")
for d, k, cr, p in resume:
    print(f"{d:>10} | {k:6d} | {cr:6.2f} | {p:8.2f} %")

hea = args.enregistrement + ".hea"
dat = args.enregistrement + ".dat"
with open(dat, "rb") as fichier:
    dat_zip = len(zlib.compress(fichier.read(), 9))   # référence sans perte

ed.afficher_comparaison(
    "COMPARAISON DES TAILLES",
    [
        ("ECG d'origine (.hea + .dat)", ed.taille(hea, dat)),
        ("Référence : origine compressée sans perte (zip)", ed.taille(hea) + dat_zip),
        ("ECG compressé par DCT (.npz)", ed.taille(fichier_compresse)),
    ],
)

print(f"\nFichier compressé : {fichier_compresse}")
print(f"Graphiques        : {dossier_graph}/")
print(f"Pour reconstruire : python decompresser.py {fichier_compresse}")
