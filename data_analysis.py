import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from backend.data_loader import DataLoader
from backend.config import get_timestamp


def analyze_dataset(loader: DataLoader) -> None:
    """
    Perform statistical analysis of the dataset
    :param loader: The DataLoader that will be used to read the patches from disk
    :returns: Nothing
    """
    # Create Directory Plotting Images
    if "data_analysis" in os.listdir("images"):
        shutil.rmtree("images/data_analysis")
    os.mkdir("images/data_analysis")

    # Analyze Initial Mask
    mask = np.clip(loader.read_image("data/label.tif"), a_min=0, a_max=1)
    total_pixels = mask.size
    water_pixels = np.sum(mask)
    print("\nMASK\n")
    print(f"Total Pixels: {total_pixels}\nWater Pixels: {water_pixels}\nWater Percentage: {round(water_pixels / total_pixels * 100.0, ndigits=2)}%")

    # Analyze Tiles
    water_pixels_hist = []
    for tile in range(1, 401):
        mask_tile = loader.get_mask(tile)
        total_pixels = mask_tile.size
        water_pixels = np.sum(mask_tile)
        water_percentage = water_pixels / total_pixels * 100.0
        water_pixels_hist.append(water_percentage)
        print(tile, water_percentage)
        
    
    # Generate Histogram For All Tiles
    stats = plt.hist(water_pixels_hist, bins=np.arange(0.0, 25.0, 1.0))
    plt.title("All Tiles")
    plt.xlabel("Water Pixels (%)")
    plt.ylabel("Tiles (Count)")
    plt.savefig("images/data_analysis/histogram_all.png", bbox_inches='tight')
    plt.close()
    
    # Summarize Histogram Statistics
    print("\nSUMMARIZE PATCH HISTOGRAM\n")
    for count, b in zip(stats[0], stats[1]):
        print(f"[{b}, {b+1.0}): {int(count)}")

    # Generate Histogram For Tiles With At Least 5% Water
    stats = plt.hist(list(filter(lambda x: x >= 5.0, water_pixels_hist)), bins=np.arange(0.0, 25.0, 1.0))
    plt.title("Tiles With At Least 5% Water")
    plt.xlabel("Water Pixels (%)")
    plt.ylabel("Tiles (Count)")
    plt.savefig("images/data_analysis/histogram_over5.png", bbox_inches='tight')
    plt.close()

    # Additional Statistics
    print("\nADDITIONAL STATISTICS\n")
    print(f"Tiles With No Water: {len(list(filter(lambda x: x == 0.0, water_pixels_hist)))}")
    print(f"Tiles With Water: {len(list(filter(lambda x: x > 0.0, water_pixels_hist)))}")
    print(f"Tiles With Less Than 5% Water: {len(list(filter(lambda x: x < 5.0, water_pixels_hist)))}")
    print(f"Tiles With Over 5% Water: {len(list(filter(lambda x: x >= 5.0, water_pixels_hist)))}")
    print(f"Tiles With Less Than 10% Water: {len(list(filter(lambda x: x < 10.0, water_pixels_hist)))}")
    print(f"Tiles With Over 10% Water: {len(list(filter(lambda x: x >= 10.0, water_pixels_hist)))}")


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())
    
    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Perform Analysis
    analyze_dataset(loader)


if __name__ == "__main__":
    args = sys.argv
    GPU = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU}"
    main()
