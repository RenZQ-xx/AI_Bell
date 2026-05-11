import numpy as np


class BellSampler:
    """Base helpers for Bell-expression coefficient sampling."""

    @staticmethod
    def normalize_vectors(vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    @staticmethod
    def perturb_vector(base_vector, n_samples, noise_level=0.2):
        base_vector = np.asarray(base_vector, dtype=np.float64)
        _, dim = base_vector.shape

        noise = np.random.normal(0, 1, (n_samples, dim))
        noise = BellSampler.normalize_vectors(noise) * noise_level

        indices = np.arange(n_samples) % len(base_vector)
        perturbed_data = noise + base_vector[indices]
        return BellSampler.normalize_vectors(perturbed_data)


class Sampler222(BellSampler):
    """
    Sampler for the 2-party, 2-input, 2-output scenario.

    Vector order:
    [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    """

    def __init__(self):
        self.dim = 8
        self.chsh_vector = self._generate_chsh_symmetries()

    def _generate_chsh_symmetries(self):
        variants = []
        base_patterns = np.array(
            [
                [1, 1, 1, -1],
                [1, 1, -1, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 1],
            ],
            dtype=np.float64,
        )

        for pattern in base_patterns:
            vec_pos = np.concatenate([np.zeros(4), pattern])
            variants.append(vec_pos)
            variants.append(-vec_pos)

        return np.array(variants)

    def generate_data(self, n_samples, ratio_random=0.5, noise_level=0.3):
        n_random = int(n_samples * ratio_random)
        n_perturbed = n_samples - n_random

        random_data = np.random.normal(0, 1, (n_random, self.dim))
        random_data = self.normalize_vectors(random_data)

        if n_perturbed > 0:
            perturbed_data = self.perturb_vector(self.chsh_vector, n_perturbed, noise_level)
            dataset = np.vstack([random_data, perturbed_data])
        else:
            dataset = random_data

        np.random.shuffle(dataset)
        return dataset


def embed_222_to_322(data_222):
    """
    Embed 2-2-2 coefficient vectors into 3-2-2 by zero-padding C terms.

    2-2-2 order:
    [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]

    3-2-2 order:
    [A0, A1, B0, B1, C0, C1, AB..., AC..., BC..., ABC...]
    """
    data_222 = np.asarray(data_222)
    if data_222.ndim == 1:
        data_222 = data_222.reshape(1, -1)
    if data_222.shape[1] != 8:
        raise ValueError(f"Expected 8 columns for 2-2-2 vectors, got {data_222.shape[1]}")

    data_322 = np.zeros((data_222.shape[0], 26), dtype=data_222.dtype)
    data_322[:, 0:4] = data_222[:, 0:4]
    data_322[:, 6:10] = data_222[:, 4:8]
    return data_322


__all__ = ["BellSampler", "Sampler222", "embed_222_to_322"]
