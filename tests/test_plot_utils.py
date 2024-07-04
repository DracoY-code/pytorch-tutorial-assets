from typing import Any, Dict

import matplotlib.pyplot as plt
import pytest
import torch
import torchvision
import torchvision.transforms as transforms

import scripts.plot_utils as plot_utils
import scripts.test_utils as test_utils


@pytest.fixture
def random_image() -> Dict[str, Any]:
    """
    Sets up a random image from the CIFAR10 dataset.

    Returns:
        Dict[str, Any]: 
    """
    # Set the image classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load the CIFAR10 dataset
    cifar10_ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor(),
    )

    # Setup a data loader
    data_loader = torch.utils.data.DataLoader(
        cifar10_ds, batch_size=1, shuffle=True,
    )

    # Get a random image and its label
    image, label = next(iter(data_loader))
    
    return {
        'data': image,
        'target': classes[label],
        'size': image.size(),
    }


def test_plotting(random_image: Dict[str, Any]) -> None:
    """
    Test function to plot an random image.

    Args:
        random_image (Dict[str, Any]): The random image and its metadata.
    """
    # Print header
    print(test_utils.get_func_header(test_plotting))

    # Plot the random image
    print(f"Plotting an image of: {random_image['target']}")
    plot_utils.plot_image(
        random_image['data'], False,
        label=random_image['target'],
    )
    plt.show()

    assert True
