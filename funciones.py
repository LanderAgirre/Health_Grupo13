
def segmentacion(img_rgb):
    """Método 1: Básico (Escala de grises + Otsu Global)"""
    gris = color.rgb2gray(img_rgb)
    thresh = filters.threshold_otsu(gris)
    return gris < thresh 

def segmentacion_col(img_rgb):
    """
    Método 2: Avanzado (Propuesto en PDF Metodos_segmentacion)
    - Espacio de Color HSV (Canal S)
    - Operaciones morfológicas para limpiar ruido
    """
    img_hsv = color.rgb2hsv(img_rgb)
    s_channel = img_hsv[:, :, 1]
    
    try:
        thresh = filters.threshold_otsu(s_channel)
        mask = s_channel > thresh
    except:
        return np.zeros(s_channel.shape, dtype=bool)
        
    mask = morphology.closing(mask, morphology.disk(3))
    mask = morphology.opening(mask, morphology.disk(3))
    
    labels = measure.label(mask)
    if labels.max() == 0: return mask
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return ndimage.binary_fill_holes(largest)

def calcular_sim(mask1, mask2):
    """Métrica para comparar similitud entre dos segmentaciones"""
    interseccion = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0: return 0
    return np.sum(interseccion) / np.sum(union)

def momento(px):
    if len(px) == 0: return [0, 0, 0, 0]
    m, s = np.mean(px), np.std(px)
    sk = skew(px) if s > 0 else 0
    ku = kurtosis(px) if s > 0 else 0
    return [m, s, sk, ku]

def variables(row):
    path = row['path']
    img = plt.imread(path)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.resize(img, (128, 128))
    if img.max() > 1.0:
        img_norm = img / 255.0
    else:
        img_norm = img
    try:
        hsv = color.rgb2hsv(img_norm)
        s = hsv[:, :, 1]
        mask = s > filters.threshold_otsu(s)
        mask = morphology.closing(mask, morphology.disk(3))
        mask = morphology.opening(mask, morphology.disk(3))
    except:
        mask = np.zeros((128,128), dtype=bool)
    if np.sum(mask) < 50:
        gray = color.rgb2gray(img_norm)
        mask = gray < filters.threshold_otsu(gray)
        mask = morphology.opening(mask, morphology.disk(2))
    if np.sum(mask) < 10:
        mask = np.zeros((128,128), dtype=bool)
        mask[32:96, 32:96] = True 
    props = measure.regionprops(mask.astype(int))[0]
    eccentricity = props.eccentricity
    asym = props.eccentricity 
    compactness = (props.perimeter**2)/(4*np.pi*props.area) if props.area > 0 else 0
    solidity = props.solidity
    diameter = props.equivalent_diameter
    area = props.area
    
    feat_col = []
    img_lab = color.rgb2lab(img_norm)
    img_hsv = color.rgb2hsv(img_norm)
    for space in [img_norm, img_hsv, img_lab]:
        masked_px = space[mask]
        for ch in range(3):
            feat_col.extend(momento(masked_px[:, ch]))
            
    hist = cv2.calcHist([img_norm.astype('float32')], [0,1,2], mask.astype(np.uint8), [8]*3, [0,1]*3)
    n_colors = np.sum((hist/np.sum(hist)) > 0.005)
    
    return np.concatenate([
        [eccentricity, asym, compactness, solidity, diameter, area],
        feat_col, 
        [n_colors]
    ])

import sklearn.metrics as skm
def resultados(model, X_set, y_true, name):
    print(f"\nResultados: {name}")
    y_pred = model.predict(X_set)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusión - {name}')
    plt.show()
def graficar_roc_seguro(model, X_val, label_name):
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)
        else:
            y_prob = model.decision_function(X_val)
    except:
        print(f"Advertencia: No se puede calcular ROC para {label_name}")
        return
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = skm.roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    auc_macro = skm.auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, label=f'{label_name} (Macro AUC={auc_macro:.2f})', lw=2)

def roc_clase_seguro(model, X_set, y_test, model_name):
    """
    Genera un gráfico ROC con una curva por cada clase para un modelo específico.
    Versión segura contra conflictos de nombres.
    """
    try:
        y_score = model.predict_proba(X_set)
    except AttributeError:
        y_score = model.decision_function(X_set)
    n_classes = len(le.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown'])
    print(f"Generando ROC detallado para {model_name}...")
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = skm.roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = skm.auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5) 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title(f'Detalle por Clase: {model_name}')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.show()