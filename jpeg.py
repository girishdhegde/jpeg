import numpy as np
import cv2
import einops


from utils import to_image


__author__ = "__Girish_Hegde__"


class Encoder:
    """ JPEG Encoder
        author: girish d. hegde
        contact: girish.dhc@gmail.com

    Args:
        image (str/np.ndarray): path to image or [H, W, 3] - image.
    """
    def __init__(self, image, ):
        self.img = image if isinstance(image, np.ndarray) else cv2.imread(image)

    def compress(self):
        ycrcb = self.color_treatment()
        img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        to_image(img, rgb=False)

    def color_treatment(self):
        """ Fucntion to differentiate color and brightness and apply chroma subsampling(4:2:0).
            author: girish d hegde
            contact: girish.dhc@gmail.com

        Ref:
            https://en.wikipedia.org/wiki/Chroma_subsampling

        Returns:
            np.ndarray: [self.img.height, self.img.width, 3] - image.
        """
        h, w, _ = self.img.shape
        self.img = np.pad(self.img, ((0, h%2), (0, w%2), (0, 0)))

        # differenctiating brightness and color
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)

        # 4:2:2 chroma subsampling
        top_left_crcb = ycrcb[::2, ::2, 1:]
        subsampled_crcb = np.repeat(top_left_crcb, 2, axis=0)
        subsampled_crcb = np.repeat(subsampled_crcb, 2, axis=1)
        ycrcb[:, :, 1:] = subsampled_crcb

        return ycrcb


if __name__ == '__main__':
    encoder = Encoder('./data/minion.webp')
    encoder.compress()



