import torch


def soft_dice_loss(prediction, target, smooth=1e-6):

    prediction = torch.sigmoid(prediction)

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    intersection = (prediction * target).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )

    return 1 - dice