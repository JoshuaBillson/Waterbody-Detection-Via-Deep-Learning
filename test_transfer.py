import json
import random
from typing import Dict, List, Sequence
import numpy as np
import cv2
import matplotlib.pyplot as plt
from backend.data_loader import ImgSequence, DataLoader
from backend.config import get_timestamp, get_bands, get_batch_size
from backend.utils import adjust_rgb
from tensorflow.image import flip_up_down, flip_left_right, rot90


class TransferImgSequence(ImgSequence):
    """A class to demonstrate the waterbody transfer method."""

    def __init__(self, data_loader: DataLoader, patches: List[int], batch_size: int = 32, bands: Sequence[str] = None, augment_data: bool = False, shuffle: bool = True):
        # If We Want To Apply Waterbody Transferrence, Locate All Patches With At Least 10% Water
        super().__init__(data_loader, patches, batch_size, bands, augment_data, shuffle)

        # If We Want To Apply Waterbody Transferrence, Locate All Patches With At Least 10% Water
        self.transfer_patches = []
        if self.augment_data:
            for source_index in self.indices:
                source_mask = self.data_loader.get_mask(source_index)
                if self._water_content(source_mask) > 20:
                    print(source_index, self._water_content(source_mask))
                    self.transfer_patches.append(source_index)

    def show_waterbody_transfer(self):
        """Demonstrate the waterbody tranfer method"""
        for patch in self.indices:
            features = self._get_features(patch)
            self.transfer_waterbody(features, index=patch, threshold=5.0)

    def transfer_waterbody(self, features: Dict[str, np.ndarray], index: int = 1, threshold: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Given a destination patch, transfers a waterbody to the destination if the destination has a water content below a given threshold.
        :param features: A dictionary of input features (RGB, NIR, SWIR) and the mask to which we want to transfer a waterbody
        :param index: The index of the destination patch; used for plotting the resulting transfer
        :param threshold: The water content threshold below which we apply waterbody transfer
        """
        # while self._water_content(mask) < threshold:
        probability = int(100.0 - (self._water_content(features["mask"]) * 20))
        print("PROBABILITY", probability, self._water_content(features["mask"]))
        if random.randint(1, 100) <= probability:

            # Get Random Numbers To Determine Rotation/Flip Of Source Patch
            num_rotations = random.randint(0, 3)
            flip_horizontal = random.randint(1, 100)
            flip_vertical = random.randint(1, 100)

            # Get Source Mask
            assert len(self.transfer_patches) > 0, "Error: Cannot Augment Dataset Without Transfer Patches!"
            source_index = self.transfer_patches[random.randint(0, len(self.transfer_patches) - 1)]
            source_features = self._get_features(source_index)
            source_mask = self._augment_patch(source_features["mask"], num_rotations, flip_horizontal, flip_vertical)

            # Apply Waterbody Transfer To Each Feature Map
            for band in self.bands:

                # Get Source Feature
                source_feature = self._augment_patch(source_features[band], num_rotations, flip_horizontal, flip_vertical)

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
        
    def _augment_patch(self, img, num_rotations, flip_horizontal, flip_vertical):
            channels = img.shape[-1]
            img = self._rotate_90(img, num_rotations)
            img = self._flip_horizontal(img, flip_horizontal)
            img = self._flip_vertical(img, flip_vertical)
            return np.reshape(img, (512, 512, channels))

    def _rotate_90(self, img, num_rotations):
            rotations = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
            return cv2.rotate(img, rotations[num_rotations]) if num_rotations != 0 else img

    def _flip_horizontal(self, img, outcome):
            return cv2.flip(img, 1) if outcome <= 50 else img

    def _flip_vertical(self, img, outcome):
            return cv2.flip(img, 0) if outcome <= 50 else img


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
