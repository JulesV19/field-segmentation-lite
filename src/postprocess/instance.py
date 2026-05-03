import numpy as np
from scipy.ndimage import label as scipy_label, gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def predict_instances(logits: np.ndarray, dist_map: np.ndarray = None,
                      min_distance: int = 15) -> np.ndarray:
    """
    Extrait des instances individuelles de parcelles depuis les logits du U-Net.

    logits     : (3, H, W) — canaux 0=bg, 1=intérieur, 2=bordure
    dist_map   : ignoré (conservé pour compatibilité API)
    min_distance : espacement minimal en pixels entre deux seeds watershed
    """
    probs = _softmax(logits)
    p_bg  = probs[0]
    p_int = probs[1]

    # Masque champ = tout ce qui n'est pas background
    field_mask = p_bg < 0.5

    # EDT calculée depuis les pixels intérieurs confiants → lisse par nature,
    # contrairement à la dist_map prédite par dist_head qui est bruitée.
    interior_mask = p_int > 0.5
    distance = distance_transform_edt(interior_mask)
    # Lissage léger pour fusionner les maxima voisins dans un même champ
    distance = gaussian_filter(distance, sigma=2)

    coords = peak_local_max(distance, min_distance=min_distance, labels=field_mask)
    seed_mask = np.zeros_like(distance, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    seeds, _ = scipy_label(seed_mask)

    instances = watershed(-distance, markers=seeds, mask=field_mask)
    return instances.astype(np.int32)
