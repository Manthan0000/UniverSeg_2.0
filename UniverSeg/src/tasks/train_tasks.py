import random
import torch


class MultiDomainTaskSampler:

    def __init__(self, datasets):
        self.datasets = datasets
        self.names = list(datasets.keys())

    def sample_task(self, support_size=4):

        dataset_name = random.choice(self.names)

        dataset = self.datasets[dataset_name]

        indices = random.sample(
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

        return {
            "dataset": dataset_name,
            "support_images": support_images,
            "support_labels": support_labels,
            "query_image": query_image,
            "query_label": query_label
        }