import json
import numpy as np
from backend.data_loader import ImgSequence, DataLoader
from backend.config import get_timestamp, get_bands, get_batch_size


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))
    data = ImgSequence(loader, np.array(range(1, 3601)), bands=get_bands(config), batch_size=get_batch_size(config), augment_data=True)

    # Run Transfer
    data.show_agumentation()


if __name__ == "__main__":
    main()
