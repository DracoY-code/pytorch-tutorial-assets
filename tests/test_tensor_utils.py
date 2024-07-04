from typing import Any, Dict

import pytest

import scripts.tensor_utils as tensor_utils
import scripts.test_utils as test_utils


@pytest.fixture
def random_tensor() -> Dict[str, Any]:
    """
    Sets up a tensor of size (1, 3, 3) with random values.

    Returns:
        Dict[str, Any]: The random tensor and its metadata.
    """
    return {
        'tensor': tensor_utils.create_random_tensor((1, 3, 3)),
        'size': (1, 3, 3),
    }


def test_normalization(random_tensor: Dict[str, Any]) -> None:
    """
    Tests if the normalization and denormalization functions are working correctly.

    Args:
        random_tensor (Dict[str, Any]): The random tensor and its metadata.
    """
    # Show header
    print(test_utils.get_func_header(test_normalization))
    
    # Normalize the random tensor
    normalized_tensor = tensor_utils.normalize(
        random_tensor['tensor'],
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # Denormalize the normalized tensor
    denormalized_tensor = tensor_utils.denormalize(
        normalized_tensor,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # Print the variables
    print(f"\nRandom Tensor:\n{random_tensor['tensor']}")
    print(f'\nNormalized Tensor:\n{normalized_tensor}')
    print(f'\nDenormalized Tensor:\n{denormalized_tensor}')

    assert denormalized_tensor.size() == normalized_tensor.size()
