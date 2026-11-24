"""Apply researcher's image-specific corrections at catalog and embedding time."""
import json
from pathlib import Path

import numpy as np

import sticker_ranker

REVIEWS = Path(__file__).parent / 'v6/catalog-reviews.json'


def reviewed_rows(rows, reviews=None):
    if reviews is None:
        reviews = json.loads(REVIEWS.read_text())
    present = set()
    catalog = []
    for row in rows:
        review = reviews.get(row['id'])
        if review is None:
            catalog.append(row)
            continue
        if (row.get('image_sha256') != review['image_sha256']
                or row['meaning'] != review['previous_meaning'] or row['id'] in present):
            raise ValueError('Review does not match original image and annotation: ' + row['id'])
        present.add(row['id'])
        catalog.append(dict(row, meaning=review['meaning'], text=review['meaning'],
                            meaning_review=review['source'], reviewed_intent=review.get('intent','')))
    if present != set(reviews):
        raise ValueError('A reviewed image is missing from the catalog')
    return catalog


def meaning_vectors(rows, historical, encoder=None):
    if len(rows) != len(historical):
        raise ValueError('Mismatched catalog and embedding count')
    vectors = np.array(historical, dtype=np.float32, copy=True)
    indices = [i for i, row in enumerate(rows) if row.get('meaning_review')]
    if indices:
        encoder = encoder or sticker_ranker.local_encoder()
        vectors[indices] = sticker_ranker.unit(encoder.encode([rows[i]['meaning'] for i in indices]))
    return vectors
