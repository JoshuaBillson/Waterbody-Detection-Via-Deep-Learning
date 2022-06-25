
class TransferImgSequence2(ImgSequence):
    """A class to demonstrate the waterbody transfer method."""

    def subsample_patch(self, patch: Dict[str, np.ndarray], sample_random: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        top_left, top_right, bottom_left, bottom_right = dict(), dict(), dict(), dict()
        xs = [0, 256, 0, 256] if not sample_random else [random.randint(0, 256) for _ in range(4)]
        ys = [0, 0, 256, 256] if not sample_random else [random.randint(0, 256) for _ in range(4)]
        for band in patch.keys():
            top_left[band] = patch[band][xs[0]:xs[0]+256, ys[0]:ys[0]+256, :]
            top_right[band] = patch[band][xs[1]:xs[1]+256, ys[1]:ys[1]+256, :]
            bottom_left[band] = patch[band][xs[2]:xs[2]+256, ys[2]:ys[2]+256, :]
            bottom_right[band] = patch[band][xs[3]:xs[3]+256, ys[3]:ys[3]+256, :]
        return top_left, top_right, bottom_left, bottom_right
    
    def generate_composite(self, patch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Generate Random Ordering Of Quarters For Composite Image
        quarter_indices = [0, 1, 2, 3]
        random.shuffle(quarter_indices)

        # Get Sub-Patches
        top_left, top_right, bottom_left, bottom_right = self.subsample_patch(patch, sample_random=True)

        # Apply Rotations To Quarters
        self._rotate_patch(top_left)
        self._rotate_patch(top_right)
        self._rotate_patch(bottom_left)
        self._rotate_patch(bottom_right)

        # Apply Flips To Quarters
        self._flip_patch(top_left)
        self._flip_patch(top_right)
        self._flip_patch(bottom_left)
        self._flip_patch(bottom_right)

        # Generate Composite
        return self._combine_quarters(top_left, top_right, bottom_left, bottom_right)
    
    def _combine_quarters(self, top_left: Dict[str, np.ndarray], top_right: Dict[str, np.ndarray], bottom_left: Dict[str, np.ndarray], bottom_right: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Generate Random Ordering Of Quarters For Composite Image
        quarter_indices = [0, 1, 2, 3]
        random.shuffle(quarter_indices)

        # Assemble Composite Image
        composite = dict()
        for band in top_left.keys():
            if band == "RGB":
                red_quarters = [np.reshape(quarter, (256, 256)) for quarter in [top_left[band][..., 0], top_right[band][..., 0], bottom_left[band][..., 0], bottom_right[band][..., 0]]]
                green_quarters = [np.reshape(quarter, (256, 256)) for quarter in [top_left[band][..., 1], top_right[band][..., 1], bottom_left[band][..., 1], bottom_right[band][..., 1]]]
                blue_quarters = [np.reshape(quarter, (256, 256)) for quarter in [top_left[band][..., 2], top_right[band][..., 2], bottom_left[band][..., 2], bottom_right[band][..., 2]]]

                red_composite = np.reshape(np.array(np.bmat([[red_quarters[0], red_quarters[1]], [red_quarters[2], red_quarters[3]]])), (512, 512, 1))
                green_composite = np.reshape(np.array(np.bmat([[green_quarters[0], green_quarters[1]], [green_quarters[2], green_quarters[3]]])), (512, 512, 1))
                blue_composite = np.reshape(np.array(np.bmat([[blue_quarters[0], blue_quarters[1]], [blue_quarters[2], blue_quarters[3]]])), (512, 512, 1))

                composite[band] = np.concatenate((red_composite, green_composite, blue_composite), axis=-1)
            else:
                quarters = [np.reshape(quarter, (256, 256)) for quarter in [top_left[band], top_right[band], bottom_left[band], bottom_right[band]]]
                composite[band] = np.reshape(np.array(np.bmat([[quarters[0], quarters[1]], [quarters[2], quarters[3]]])), (512, 512, 1))
        
        return composite

    
    def test_composite(self):
        # Create Directory For Composite Images
        if "composites2" in os.listdir("images"):
            shutil.rmtree("images/composites2")
        os.mkdir("images/composites2")

        np.random.shuffle(self.transfer_patches)
        for patch in self.transfer_patches:
            features = self._get_features(patch)
            composites = self.generate_composite(features)
            
            # Create Subplot
            _, axs = plt.subplots(2, len(self.bands) + 1, figsize = (4, 2))

            # Plot Original Features
            for row in range(4):
                for col, band in enumerate(self.bands + ["mask"]):
                    if row == 0:
                        axs[row][col].imshow(adjust_rgb(features[band], gamma=0.5) if band == "RGB" else features[band])
                        axs[row][col].axis("off")

            # Plot Composite Features
            for col, band in enumerate(self.bands + ["mask"]):
                axs[1][col].imshow(adjust_rgb(composites[band], gamma=0.5) if band == "RGB" else composites[band])
                axs[1][col].axis("off")
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
            plt.savefig(f"images/composites2/composite.{random.randint(0, 10000)}.png", dpi=1000, bbox_inches='tight')
            plt.close()