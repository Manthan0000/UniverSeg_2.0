import random
from typing import Optional
import torch


class MultiDomainTaskSampler:

    def __init__(self, datasets, seed: Optional[int] = None):
        self.datasets = datasets
        self.names = list(datasets.keys())
        self.rng = random.Random(seed) if seed is not None else random

    def sample_task(
        self,
        support_size: int = 4,
        variable_n: bool = False,
        aug: bool = False,
    ):
        if variable_n:
            support_size = self.rng.choice([1, 2, 4, 8])

        dataset_name = self.rng.choice(self.names)

        dataset = self.datasets[dataset_name]

        indices = self.rng.sample(
            range(len(dataset)),
            support_size + 1
        )

        support_indices = indices[:support_size]
        query_index = indices[-1]

        support_images = []
        support_labels = []

        for idx in support_indices:

            image, mask = dataset[idx]

            support_images.append(image)
            support_labels.append(mask)

        query_image, query_label = dataset[query_index]

        support_images = torch.stack(
            support_images
        )

        support_labels = torch.stack(
            support_labels
        )

        if aug:
            from training.augmentations import apply_recipe_r_augmentations

            rng_obj = self.rng if isinstance(self.rng, random.Random) else random.Random()
            query_image, query_label, support_images, support_labels = apply_recipe_r_augmentations(
                query_image,
                query_label,
                support_images,
                support_labels,
                rng=rng_obj,
            )

        return {
            "dataset": dataset_name,
            "support_size": support_size,
            "support_images": support_images,
            "support_labels": support_labels,
            "query_image": query_image,
            "query_label": query_label
        }
