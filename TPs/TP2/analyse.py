import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image

def analyser_image_complete(chemin_image, titre):
    # 1. Charger l'image
    img_pil = Image.open(chemin_image).convert('RGB')
    donnees_rgb = np.array(img_pil)


    # Convertir en niveaux de gris pour l'analyse de luminance
    img_gris = img_pil.convert('L')
    donnees_gris = np.array(img_gris)
    
    # Configuration de la figure (Grille 3 lignes, 2 colonnes)
    fig = plt.figure(figsize=(16, 15))
    fig.suptitle(f"Analyse Scientifique : {titre}", fontsize=20, fontweight='bold')
    
    # --- 1. APERÇU COULEUR ---
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(donnees_rgb)
    ax1.set_title("1. Image Originale (Couleurs)", fontsize=14)
    ax1.axis('off')
    
    # --- 2. HISTOGRAMME RVB (Détail des canaux) ---
    ax2 = plt.subplot(3, 2, 2)
    couleurs = ('red', 'green', 'blue')
    for i, col in enumerate(couleurs):
        hist, bins = np.histogram(donnees_rgb[:, :, i], bins=256, range=(0, 255))
        ax2.plot(bins[:-1], hist, color=col, label=f'Canal {col.capitalize()}', alpha=0.6)
    ax2.set_title("2. Histogramme RVB (Superposé)", fontsize=14)
    ax2.set_xlabel("Intensité")
    ax2.legend()

    # --- 3. HISTOGRAMME GRAYSCALE (Luminance globale) ---
    ax3 = plt.subplot(3, 2, 3)
    hist_g, bins_g = np.histogram(donnees_gris, bins=256, range=(0, 255))
    ax3.bar(bins_g[:-1], hist_g, color='gray', width=1, edgecolor='black', alpha=0.8)
    ax3.set_title("3. Histogramme Niveaux de Gris (Luminance)", fontsize=14)
    ax3.set_xlabel("Intensité (0=Noir, 255=Blanc)")
    ax3.set_ylabel("Nombre de pixels")

    # --- 4. ANALYSE DE REDONDANCE SPATIALE ---
    # Calcul de la différence entre pixels voisins
    diff_horiz = np.diff(donnees_gris.astype(float), axis=1).flatten()
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(diff_horiz, bins=100, color='purple', alpha=0.7, log=True)
    ax4.set_title("4. Redondance Spatiale (Différences)", fontsize=14)
    ax4.set_xlabel("Delta (Pixel_n - Pixel_n-1)")
    ax4.set_ylabel("Fréquence (Log)")

    # --- 5. STATISTIQUES POUR TON TRAVAIL ---
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('off')
    # Calcul simple de l'entropie de Shannon
    probs = hist_g / hist_g.sum()
    entropie = -np.sum(probs * np.log2(probs + 1e-10))
    
    texte_stats = (
        f"MESURES QUANTITATIVES :\n\n"
        f"• Résolution : {donnees_rgb.shape[1]} x {donnees_rgb.shape[0]}\n"
        f"• Couleurs uniques : {len(np.unique(donnees_rgb.reshape(-1, 3), axis=0)):,}\n"
        f"• Entropie (Gris) : {entropie:.2f} bits/pixel\n"
        f"• Écart-type (Gris) : {np.std(donnees_gris):.2f}\n"
        f"• Delta moyen : {np.mean(np.abs(diff_horiz)):.2f}"
    )
    ax5.text(0.1, 0.5, texte_stats, fontsize=13, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='azure'))

    # --- 6. RÉSUMÉ POUR L'ANALYSE ---
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    hypothese = "HYPOTHÈSE DE COMPRESSION :\n\n"
    if entropie < 2:
        hypothese += "=> Très haute redondance.\nCompression sans perte majeure (ex: RLE, PNG)."
    elif entropie < 6:
        hypothese += "=> Complexité modérée.\nLes filtres prédictifs seront efficaces."
    else:
        hypothese += "=> Haute complexité.\nLe ratio de compression sans perte sera faible."
    
    ax6.text(0.1, 0.5, hypothese, fontsize=13, fontweight='bold', verticalalignment='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def evaluation_compression_d_image(chemin_image, titre):
    
    # --- 1. Chargement image ---
    img_pil = Image.open(chemin_image).convert("RGB")
    donnees_rgb = np.array(img_pil)

    codes = 1000
    donnees_decodees = [[10,10,10]]

    # --- 3. Tailles mémoire ---
    taille_originale = sys.getsizeof(donnees_rgb)
    taille_compressee = sys.getsizeof(codes)
    taille_reconstruite = sys.getsizeof(donnees_decodees)

    # --- 4. Mesures de compression ---
    taux = (1 - taille_compressee / taille_originale) * 100
    ratio = taille_originale / taille_compressee

    # --- 5. Vérification sans perte ---
    sans_perte = np.array_equal(donnees_rgb, donnees_decodees)

    # --- 6. Affichage ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Évaluation de la compression LZW\n{titre}", fontsize=18)

    # Image originale
    axes[0].imshow(donnees_rgb)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    # Image compressée puis décompressée
    axes[1].imshow(donnees_decodees)
    axes[1].set_title("Image compressée → décompressée")
    axes[1].axis("off")

    # --- 7. Bloc texte synthèse ---
    texte = (
        f"Taille image originale : {taille_originale:,} octets\n"
        f"Taille données compressées : {taille_compressee:,} octets\n"
        f"Taille image reconstruite : {taille_reconstruite:,} octets\n\n"
        f"Taux de compression : {taux:.2f} %\n"
        f"Ratio de compression : {ratio:.2f}:1\n\n"
        f"Compression sans perte : {'OUI' if sans_perte else 'NON'}"
    )

    plt.figtext(
        0.5, 0.02, texte,
        ha="center",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="honeydew")
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.9])
    plt.show()

def taille_octets(obj):
    """Taille mémoire en octets"""
    return sys.getsizeof(obj)

def verifier_sans_perte(original, reconstruit):
    """Vérifie l'égalité parfaite pixel par pixel"""
    return np.array_equal(original, reconstruit)

def taux_compression(taille_originale, taille_compressee):
    """En pourcentage"""
    return (1 - taille_compressee / taille_originale) * 100

def ratio_compression(taille_originale, taille_compressee):
    """Ratio classique"""
    return taille_originale / taille_compressee

#analyser_image_complete("image1_natural.png", "Image naturelle")
#evaluation_compression_d_image("image1","La premiere image")