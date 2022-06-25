import json
import os
import numpy as np


def tile_to_patches(tile_index):
    return [int((tile_index * 9)) + patch for patch in range(1, 10)]


def create_batches() -> None:
    """
    Divide patches into training, validation, and test batches. Save the patch indices for each batch to disk to
    ensure that we use the same patches for training, validation, and testing across different experiments.
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :returns: Nothing
    """
    # Create Directory For Batches
    batches_directory = "batches"
    if batches_directory not in os.listdir():
        os.mkdir(batches_directory)
    
    # Divide Tiles Into Batches
    tile_indices = np.array(range(0, 400))
    np.random.shuffle(tile_indices)
    train_tiles, val_tiles, test_tiles = tile_indices[0:300], tile_indices[300:335], tile_indices[335:400]
    
    # Generate Patch Indices From Tiles
    train_patches, val_patches, test_patches = [], [], []
    for train_tile in train_tiles:
        train_patches += tile_to_patches(train_tile)
    for val_tile in val_tiles:
        val_patches += tile_to_patches(val_tile)
    for test_tile in test_tiles:
        test_patches += tile_to_patches(test_tile)
    
    # Sort Patches
    train_patches.sort()
    val_patches.sort()
    test_patches.sort()

    # Write Batches To Disk
    with open(f"batches/512.json", 'w') as batch_file:
        batch_file.write(json.dumps({"train": train_patches, "validation": val_patches, "test": test_patches}, indent=2))


if __name__ == "__main__":
    create_batches()
