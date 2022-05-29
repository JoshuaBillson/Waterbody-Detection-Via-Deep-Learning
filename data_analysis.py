import json
import matplotlib.pyplot as plt
from data_loader import DataLoader, analyze_dataset
from config import get_timestamp


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Perform Analysis
    analyze_dataset(loader)



if __name__ == "__main__":
    main()
