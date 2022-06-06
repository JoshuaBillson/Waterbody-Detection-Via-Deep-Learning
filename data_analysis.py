import os
import sys
import json
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
    # Analyze Initial Mask
    mask = np.clip(loader.read_image("data/label.tif"), a_min=0, a_max=1)
    total_pixels = mask.size
    water_pixels = np.sum(mask)
    print("\nMASK\n")
    print(f"Total Pixels: {total_pixels}\nWater Pixels: {water_pixels}\nWater Percentage: {round(water_pixels / total_pixels * 100.0, ndigits=2)}%")

    # Analyze Patches
    water_pixels_hist = []
    lower_bound, upper_bound = loader.get_bounds()
    for patch in range(lower_bound, upper_bound+1):
        mask_patch = loader.get_mask(patch)
        total_pixels = mask_patch.size
        water_pixels = np.sum(mask_patch)
        water_percentage = water_pixels / total_pixels * 100.0
        water_pixels_hist.append(water_percentage)
    
    # Generate Histogram For All Patches
    stats = plt.hist(water_pixels_hist, bins=np.arange(0.0, 51.0, 2.5))
    plt.title("All Patches")
    plt.xlabel("Water Pixels (%)")
    plt.ylabel("Patches (Count)")
    plt.savefig("images/histogram_all.png", bbox_inches='tight')
    plt.cla()
    
    # Summarize Histogram Statistics
    print("\nSUMMARIZE PATCH HISTOGRAM\n")
    for count, b in zip(stats[0], stats[1]):
        print(f"[{b}, {b+2.5}): {int(count)}")
    

    # Generate Histogram For Patches With At Least 5% Water
    stats = plt.hist(list(filter(lambda x: x >= 5.0, water_pixels_hist)), bins=np.arange(5.0, 51.0, 2.5))
    plt.title("Patches With At Least 5% Water")
    plt.xlabel("Water Pixels (%)")
    plt.ylabel("Patches (Count)")
    plt.savefig("images/histogram_over5.png", bbox_inches='tight')
    plt.cla()

    # Additional Statistics
    print("\nADDITIONAL STATISTICS\n")
    print(f"Patches With No Water: {len(list(filter(lambda x: x == 0.0, water_pixels_hist)))}")
    print(f"Patches With Water: {len(list(filter(lambda x: x > 0.0, water_pixels_hist)))}")
    print(f"Patches With Less Than 5% Water: {len(list(filter(lambda x: x < 5.0, water_pixels_hist)))}")
    print(f"Patches With Over 5% Water: {len(list(filter(lambda x: x >= 5.0, water_pixels_hist)))}")
    print(f"Patches With Less Than 10% Water: {len(list(filter(lambda x: x < 10.0, water_pixels_hist)))}")
    print(f"Patches With Over 10% Water: {len(list(filter(lambda x: x >= 10.0, water_pixels_hist)))}")

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
