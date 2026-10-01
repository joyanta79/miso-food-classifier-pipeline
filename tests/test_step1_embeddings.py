import numpy as np

from processing.step1_extract_embeddings import extract_embeddings


def test_embeddings_are_1280_dimensional_and_cosine_dedup_removes_copies(tmp_path) -> None:
    original = tmp_path / "one.bin"
    duplicate = tmp_path / "two.bin"
    distinct = tmp_path / "three.bin"
    original.write_bytes(b"same local image bytes")
    duplicate.write_bytes(b"same local image bytes")
    distinct.write_bytes(b"different local image bytes")
    manifest = [
        {"id": "one", "label": "fries", "image_path": str(original)},
        {"id": "two", "label": "fries", "image_path": str(duplicate)},
        {"id": "three", "label": "burger", "image_path": str(distinct)},
    ]

    embeddings, mapping = extract_embeddings(manifest, cosine_dedup_threshold=0.999)

    assert embeddings.shape == (2, 1280)
    assert [record["id"] for record in mapping] == ["one", "three"]
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
