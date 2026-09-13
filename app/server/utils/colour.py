"""Conservative dominant-colour estimate for centred product photographs."""
import numpy as np
from PIL import Image


FAMILIES = {
    'Green': {'Green', 'Olive', 'Sea Green', 'Lime Green', 'Fluorescent Green'},
    'Blue': {'Blue', 'Navy Blue', 'Turquoise Blue', 'Teal'},
    'Red': {'Red', 'Maroon', 'Burgundy'},
    'Orange': {'Orange', 'Rust', 'Copper'},
    'Yellow': {'Yellow', 'Mustard', 'Gold'},
    'Purple': {'Purple', 'Lavender', 'Mauve'},
    'Pink': {'Pink', 'Magenta', 'Rose'},
    'Black': {'Black', 'Charcoal'},
    'Grey': {'Grey', 'Grey Melange', 'Silver', 'Steel'},
    'White': {'White', 'Off White', 'Cream'},
}


def estimate_colour(image: Image.Image) -> str | None:
    """Ignore near-white background; abstain when no colour dominates.

    This is a product-photo heuristic, not garment segmentation. A centre crop
    limits background influence; mixed colours fall back to embedding ranking.
    """
    rgb = image.convert('RGB').resize((64, 64))
    hsv = np.asarray(rgb.convert('HSV'), dtype=float)[10:54, 10:54] / 255
    h, s, v = hsv.reshape(-1, 3).T
    foreground = ~((s < .12) & (v > .90))
    if foreground.mean() < .15:
        return None
    h, s, v = h[foreground] * 360, s[foreground], v[foreground]
    labels = np.full(len(h), 'Grey', dtype=object)
    labels[v < .20] = 'Black'
    labels[(s < .15) & (v > .80)] = 'White'
    chromatic = (s >= .20) & (v >= .20)
    for name, low, high in [('Red', 0, 20), ('Orange', 20, 45),
                            ('Yellow', 45, 70), ('Green', 70, 170),
                            ('Blue', 170, 260), ('Purple', 260, 295),
                            ('Pink', 295, 345), ('Red', 345, 361)]:
        labels[chromatic & (h >= low) & (h < high)] = name
    # Dark warm colours are often brown; avoid confidently calling them orange.
    labels[chromatic & (h >= 20) & (h < 70) & (v < .65)] = 'Unknown'
    names, counts = np.unique(labels, return_counts=True)
    winner = int(counts.argmax())
    return str(names[winner]) if counts[winner] / len(labels) >= .55 and names[winner] != 'Unknown' else None


def rank_preferences(indices, metadata, article=None, colours=()):
    """Keep cosine order within type/colour groups; never discard candidates."""
    articles = metadata['articleType'].to_numpy()
    palette = metadata['baseColour'].to_numpy()
    return np.asarray(sorted(indices, key=lambda i: (
        bool(article) and articles[i] != article,
        bool(colours) and palette[i] not in colours,
    )), dtype=int)
