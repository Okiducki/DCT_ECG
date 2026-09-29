"""
ecg_dct.py : fonctions communes aux scripts compresser.py et decompresser.py
==========================================================================

Ce fichier ne se lance pas directement : il est importé par les deux scripts.
Il contient :
    - la DCT et la DCT inverse (version manuelle de 2022 + version rapide)
    - la mise à zéro des coefficients de plus faible énergie
    - les outils de mesure et de comparaison de taille de fichiers
    - l'ajout d'un axe "fréquence (Hz)" sur les graphiques de coefficients
"""

import math
import os
import numpy as np


################################################################################
# DCT et DCT inverse : version MANUELLE (algorithme du projet 2022, corrigé)
################################################################################

def dct_manuelle(x):
    """DCT-II orthonormée, calculée avec la formule (double boucle)."""
    x = [float(v) for v in x]          # liste Python : boucles plus rapides
    nSamples = len(x)
    c = np.empty(nSamples)

    e = math.sqrt(2 / nSamples)
    for i in range(0, nSamples):
        sum = 0
        for n in range(0, nSamples):
            sum += x[n] * math.cos((math.pi * (2*n + 1) * i) / (2 * nSamples))
        if i == 0:
            f = 1 / math.sqrt(2)
        else:
            f = 1
        c[i] = e * f * sum
    return c


def idct_manuelle(c):
    """DCT-III orthonormée (DCT inverse), calculée avec la formule."""
    c = [float(v) for v in c]
    nSamples = len(c)
    x = np.empty(nSamples)

    e = math.sqrt(2 / nSamples)
    for n in range(0, nSamples):
        sum = 0
        for i in range(0, nSamples):
            if c[i] == 0:              # coefficient supprimé : rien à ajouter
                continue
            if i == 0:
                f = 1 / math.sqrt(2)
            else:
                f = 1
            sum += f * c[i] * math.cos((math.pi * (2*n + 1) * i) / (2 * nSamples))
        x[n] = e * sum
    return x


################################################################################
# DCT et DCT inverse : version RAPIDE (scipy, même résultat, ~1000 fois plus vite)
################################################################################

def dct_rapide(x):
    from scipy.fft import dct
    return dct(np.asarray(x, dtype=float), norm="ortho")


def idct_rapide(c):
    from scipy.fft import idct
    return idct(np.asarray(c, dtype=float), norm="ortho")


################################################################################
# Mise à zéro des coefficients avec la plus faible énergie
################################################################################

def mise_a_zero(c, energie_cible):
    """
    Garde les coefficients les plus énergétiques jusqu'à atteindre
    'energie_cible' (ex : 0.999 = 99,9 %) et met tous les autres à zéro.

    Retourne : (coefficients après mise à zéro, nombre k de coefficients gardés,
                fraction d'énergie réellement conservée)
    """
    energie = c ** 2
    energie_totale = np.sum(energie)
    ordre = np.argsort(energie)[::-1]          # du plus fort au plus faible

    energie_gardee = 0
    k = 0
    while energie_gardee / energie_totale < energie_cible and k < len(c):
        energie_gardee += energie[ordre[k]]
        k += 1

    c_zero = np.zeros_like(c)
    c_zero[ordre[:k]] = c[ordre[:k]]
    return c_zero, k, energie_gardee / energie_totale


################################################################################
# Mesure de qualité
################################################################################

def prd(original, reconstruit):
    """PRD (%) : erreur relative entre le signal d'origine et le reconstruit."""
    erreur = original - reconstruit
    return 100 * math.sqrt(np.sum(erreur ** 2) / np.sum(original ** 2))


################################################################################
# Tailles de fichiers
################################################################################

def taille(*chemins):
    """Taille totale (en octets) d'un ou plusieurs fichiers."""
    return sum(os.path.getsize(c) for c in chemins)


def taille_lisible(octets):
    """Convertit un nombre d'octets en texte lisible (o, Ko, Mo)."""
    if octets < 1024:
        return f"{octets} o"
    if octets < 1024 ** 2:
        return f"{octets / 1024:.1f} Ko"
    return f"{octets / 1024 ** 2:.2f} Mo"


def afficher_comparaison(titre, lignes):
    """
    Affiche un tableau de comparaison de tailles.
    lignes = [(description, taille_en_octets), ...]
    La première ligne sert de référence (100 %).
    """
    reference = lignes[0][1]
    largeur = max(len(desc) for desc, _ in lignes)

    print("\n" + "=" * (largeur + 38))
    print(titre)
    print("=" * (largeur + 38))
    print(f"{'Fichier':<{largeur}} | {'Taille':>10} | {'% origine':>9} | {'Gain':>7}")
    print("-" * (largeur + 38))
    for desc, octets in lignes:
        pourcentage = 100 * octets / reference
        gain = reference / octets
        print(f"{desc:<{largeur}} | {taille_lisible(octets):>10} | "
              f"{pourcentage:8.1f} % | x{gain:6.2f}")
    print("=" * (largeur + 38))


################################################################################
# Graphiques : axe secondaire en fréquence
################################################################################

def ajouter_axe_frequence(ax, fs, nSamples):
    """
    Ajoute en haut du graphique un second axe horizontal en Hz.
    Le coefficient d'indice k correspond à la fréquence k * fs / (2 * N).
    """
    vers_hz = lambda k: k * fs / (2 * nSamples)
    vers_k = lambda f: f * 2 * nSamples / fs
    axe_haut = ax.secondary_xaxis("top", functions=(vers_hz, vers_k))
    axe_haut.set_xlabel("Fréquence correspondante (Hz)")
    return axe_haut
