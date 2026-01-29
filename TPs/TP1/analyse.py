import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image

def analyser_image_complete(chemin_image, titre):
    # 1. Charger l'image
    img_pil = Image.open(chemin_image).convert('RGB')
    donnees_rgb = np.array(img_pil)

    # Compression en LZW 
    codes, shape = LZW_encode(img_pil)
    print(sys.getsizeof(codes))
    print(sys.getsizeof(donnees_rgb))
    decoded_img = LZW_decode(codes, shape)
    decoded_img_RGB = np.array(decoded_img)

    # Convertir en niveaux de gris pour l'analyse de luminance
    img_gris = img_pil.convert('L')
    donnees_gris = np.array(img_gris)
    
    # Configuration de la figure (Grille 3 lignes, 2 colonnes)
    fig = plt.figure(figsize=(16, 15))
    fig.suptitle(f"Analyse Scientifique : {titre}", fontsize=20, fontweight='bold')
    
    # --- 1. APERÇU COULEUR ---
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(decoded_img_RGB)
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

def LZW_encode(img):
    arr = np.array(img, dtype=np.uint8)

    height, width, _ = arr.shape
    data = arr.tobytes()

    # Initialize dictionary with single-byte entries
    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256

    w = b""
    encoded = []

    for k in data:
        wk = w + bytes([k])
        if wk in dictionary:
            w = wk
        else:
            encoded.append(dictionary[w])
            dictionary[wk] = next_code
            next_code += 1
            w = bytes([k])

    if w:
        encoded.append(dictionary[w])

    return encoded, (height, width)
    

def LZW_decode(codes, shape):
    """
    LZW decode a list of integer codes.
    Returns the original byte sequence.
    """
    height, width = shape

    # Initialize dictionary
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256

    w = dictionary[codes[0]]
    decoded = bytearray(w)

    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == next_code:
            # Special LZW case
            entry = w + w[:1]
        else:
            raise ValueError("Invalid LZW code")

        decoded.extend(entry)

        # Add new sequence to dictionary
        dictionary[next_code] = w + entry[:1]
        next_code += 1

        w = entry

    arr = np.frombuffer(bytes(decoded), dtype=np.uint8)
    arr = arr.reshape((height, width, 3))

    return Image.fromarray(arr, mode="RGB")


#analyser_image_complete("image1_natural.png", "Image naturelle")
#analyser_image_complete("image2_synthetic.png", "Image synthétique")
analyser_image_complete("image3_binary.png", "Image binary")