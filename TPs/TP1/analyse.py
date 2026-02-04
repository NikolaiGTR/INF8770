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

    # --- 2. Compression / Décompression LZW ---
    codes = LZW_encode(img_pil)
    img_decodee = LZW_decode(codes)
    
    #img_decodee = LZW_decode(code, size)
    donnees_decodees = np.array(img_decodee)


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

    
def LZW_encode(img):
    encoded = []
    arr = np.array(img, dtype=np.uint8)

    height, width, _ = arr.shape
    encoded.append((height,width))

    dictsymb = {bytes([i]): i for i in range(256)}


    size_dict = len(dictsymb)
    data = arr.tobytes()

    i = 0

    while i < len(data):
        w = bytes([data[i]])
        i += 1

        while i < len(data) and (w + bytes([data[i]])) in dictsymb:
            w = w + bytes([data[i]])
            i += 1

        # Emit code
        encoded.append(dictsymb[w])
        # Add new sequence to dictionary
        if i < len(data):
            dictsymb[w + bytes([data[i]])] = size_dict
            size_dict += 1  
    print(size_dict)
    print(np.log2(size_dict))
    return encoded

# Ce code a été écrit initialement en s'inspirant du code fournit pas le professeur il fonctionne, mais extrêmement inefficace.
def LZW_encode2(img):
    encoded = []
    arr = np.array(img, dtype=np.uint8)

    height, width, _ = arr.shape
    encoded.append((height,width))
    dictbin = []
    dictsymb = []
    
    for i in range(256):
        dictbin += ["{:b}".format(i).zfill(int(np.ceil(np.log2(256))))]
        dictsymb += ["{:b}".format(i)]

    size_dict = len(dictsymb)
    data = arr.tobytes()

    i = 0
    longueur = 0
    while i < len(data):
        to_add_dict = "{:b}".format(data[i]) # Caracter to add in dict (+1 char)
        encoded_block = "{:b}".format(data[i]) # Caracter to add to coded block

        while to_add_dict in dictsymb and i < len(data):
            i += 1
            encoded_block = to_add_dict
            if i < len(data):
                to_add_dict += "{:b}".format(data[i])
        binary_code = [dictbin[dictsymb.index(encoded_block)]]
        encoded += binary_code
        longueur += len(binary_code[0])

        if i < len(data):
            dictsymb += [to_add_dict]
            dictbin += ["{:b}".format(size_dict)]
            size_dict +=1

        # Ajout bit pour encoder si necessaire
        if np.ceil(np.log2(size_dict)) > len(encoded[-1]):
            for j in range(size_dict):
                dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(size_dict))))

    
    return encoded

# Code de décodage maison qui fonctionne, mais inefficace
def LZW_decode2(code):
    height, width = code[0]
    decoded = []
    
    dictbin = ["{:b}".format(i).zfill(int(np.ceil(np.log2(256)))) for i in range(256)]
    dictsymb = [[i] for i in range(256)]
  
    size_dict = len(dictsymb)
    i = 1
    while i < len(code):
        block = code[i]
        if block in dictbin :
            decoded_block = dictsymb[dictbin.index(block)]
            decoded += decoded_block

            if i+1 < len(code):
                tmp_dict_entry = []
                for item in decoded_block:
                    tmp_dict_entry += [item]
                dictsymb += [tmp_dict_entry]
                dictbin += ["{:b}".format(size_dict)]
                size_dict +=1

                # Ajout bit pour encoder si necessaire
                if np.ceil(np.log2(size_dict)) > len(dictbin[0]):
                    for j in range(size_dict):
                        dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(size_dict))))

                new_dict_entry = tmp_dict_entry + [dictsymb[dictbin.index(code[i+1])][0]]
                dictsymb[-1] = new_dict_entry
        else:
            print("Symbole inconnu")
            return
        i += 1     
    arr = np.frombuffer(bytes(decoded), dtype=np.uint8)
    arr = arr.reshape((height, width, 3))
    return Image.fromarray(arr, mode="RGB")

def LZW_decode(code):
    height, width = code[0]

    # Initialize dictionary
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256

    # First code
    prev_code = code[1]
    decoded = bytearray(dictionary[prev_code])

    for curr_code in code[2:]:
        if curr_code in dictionary:
            entry = dictionary[curr_code]
        elif curr_code == next_code:
            # KwKwK case
            entry = dictionary[prev_code] + dictionary[prev_code][:1]
        else:
            raise ValueError("Invalid LZW code")

        decoded.extend(entry)

        # Add new dictionary entry
        dictionary[next_code] = dictionary[prev_code] + entry[:1]
        next_code += 1

        prev_code = curr_code

    # Rebuild image
    arr = np.frombuffer(decoded, dtype=np.uint8)
    arr = arr.reshape((height, width, 3))

    return Image.fromarray(arr, mode="RGB")

analyser_image_complete("image1_natural.png", "Image naturelle")
analyser_image_complete("image2_synthetic.png", "Image synthétique")
analyser_image_complete("image3_binary.png", "Image binary")

evaluation_compression_d_image("image1_natural.png", "Image naturelle")
evaluation_compression_d_image("image2_synthetic.png", "Image synthétique")
evaluation_compression_d_image("image3_binary.png", "Image binary")