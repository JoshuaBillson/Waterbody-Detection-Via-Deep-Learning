import json
import random
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from backend.data_loader import ImgSequence, DataLoader
from backend.config import get_timestamp, get_bands, get_batch_size
from backend.utils import adjust_rgb


class TransferImgSequence(ImgSequence):
    """A class to demonstrate the waterbody transfer method."""
    def show_waterbody_transfer(self):
        """Demonstrate the waterbody tranfer method"""
        for patch in self.indices:
            features = self._get_features(patch)
            self.transfer_waterbody(features, index=patch)

    def transfer_waterbody(self, features: Dict[str, np.ndarray], index: int = 1, threshold: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Given a destination patch, transfers a waterbody to the destination if the destination has a water content below a given threshold.
        :param features: A dictionary of input features (RGB, NIR, SWIR) and the mask to which we want to transfer a waterbody
        :param index: The index of the destination patch; used for plotting the resulting transfer
        :param threshold: The water content threshold below which we apply waterbody transfer
        """
        # while self._water_content(mask) < threshold:
        if self._water_content(features["mask"]) <= threshold:

            # Get Source Mask
            assert len(self.transfer_patches) > 0, "Error: Cannot Augment Dataset Without Transfer Patches!"
            source_index = self.transfer_patches[random.randint(0, len(self.transfer_patches) - 1)]
            source_features = self._get_features(source_index)
            source_mask = source_features["mask"]

            # Apply Waterbody Transfer To Each Feature Map
            for band in self.bands:

                # Get Source Feature
                source_feature = source_features[band]

                # Extract Waterbody From Source Feature 
                waterbody = source_mask * source_feature

                # Plot Source Mask
                _, axs = plt.subplots(1, 6)
                axs[0].imshow(source_mask)
                axs[0].set_title("Src. Mask", fontsize=6)
                axs[0].axis("off")

                # Plot Source Features
                axs[1].imshow(adjust_rgb(source_feature, gamma=0.5) if band == "RGB" else source_feature)
                axs[1].set_title("Src. Features", fontsize=6)
                axs[1].axis("off")

                # Plot Destination Mask
                axs[2].imshow(features["mask"])
                axs[2].set_title("Dest. Mask", fontsize=6)
                axs[2].axis("off")

                # Plot Destination Features
                axs[3].imshow(adjust_rgb(features[band], gamma=0.5) if band == "RGB" else features[band])
                axs[3].set_title("Dest. Features", fontsize=6)
                axs[3].axis("off")

                # Plot Augmented Mask
                axs[4].imshow(np.where((features["mask"] + source_mask) >= 1, 1, 0).astype("uint16"))
                axs[4].set_title("Final Mask", fontsize=6)
                axs[4].axis("off")

                # Remove Waterbody Region From Destination Feature
                features[band] *= np.where(source_mask == 1, 0, 1).astype("uint16")

                # Transfer Waterbody To Destination Feature Map
                features[band] += waterbody
                
                # Plot Augmented Patch
                axs[5].imshow(adjust_rgb(features[band], gamma=0.5) if band == "RGB" else features[band])
                axs[5].set_title("Final Features", fontsize=6)
                axs[5].axis("off")

                # Save Figure
                plt.savefig(f"transfers/transfer_{index}_{band}.png", dpi=1000, bbox_inches='tight')
                plt.close()

            features["mask"] = np.where((features["mask"] + source_mask) >= 1, 1, 0).astype("uint16")

        return features


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))
    data = TransferImgSequence(loader, np.array(range(1, 3601)), bands=get_bands(config), batch_size=get_batch_size(config), augment_data=True)

    # Run Transfer
    data.show_waterbody_transfer()


if __name__ == "__main__":
    main()
