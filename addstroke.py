import typing
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt

# the color type definition from PIL.Image.new()
PIL_Color = typing.Union[float, tuple[float, ...], str, None]
# (top, bottom), (left, right)
NpPadding = tuple[tuple[int, int], tuple[int, int]]
# left, right, top, bottom
ExpandedBy = tuple[int, int, int, int]


def getexpandby(mask: np.ndarray, r: int) -> NpPadding:
    E = []
    for (cropped, axis) in [(mask[:r], 1), (mask[-r:], 1), (mask[:, :r], 0),
                            (mask[:, -r:], 0)]:
        nonzero = np.any(cropped != 0, axis=axis)
        E.append(r - int(np.argmax(nonzero) if np.any(nonzero) else r))
    return ((E[0], E[1]), (E[2], E[3]))


def strokemask(img: Image.Image,
               radius: int,
               auto_expand: bool = False,
               threshold: int = 127) -> tuple[Image.Image, ExpandedBy]:
    """
    adds stroke around the alpha layer of a transparent image

    this function is docstringed separately so you can, for example, use an image for the stroke instead of plain color

    :param PIL.Image.Image img: the image in question
    :param int radius: radius of the stroke
    :param bool auto_expand: whether to expand image if the stroke exceeds its boundaries (default False)
    :param int threshold: what alpha value in the input image is considered opaque (default 127)

    :returns PIL.Image.Image: the alpha mask
    :returns tuple[int, int, int, int]: how much the image was expanded by (if auto_expand is False, this is (0, 0, 0, 0))
    """

    mask = np.array(img)[:, :, 3] > threshold
    nppad = getexpandby(mask, radius) if auto_expand else ((0, 0), (0, 0))
    mask = np.pad(~mask, nppad, constant_values=255)
    expandedby = nppad[1] + nppad[0]

    distance = distance_transform_edt(mask)
    mask = np.clip(radius - distance, 0, 1)
    mask = (mask * 255).astype(np.uint8)
    return (Image.fromarray(mask), expandedby)


def addstroke(img: Image.Image,
              radius: int,
              color: PIL_Color,
              auto_expand: bool = False,
              threshold: int = 127) -> tuple[Image.Image, ExpandedBy]:
    """
    adds stroke around a transparent image

    :param PIL.Image.Image img: the image in question
    :param int radius: radius of the stroke
    :param color: the stroke color (any PIL compatible color type)
    :param bool auto_expand: whether to expand image if the stroke exceeds its boundaries (default False)
    :param int threshold: what alpha value in the input image is considered opaque (default 127)

    :returns PIL.Image.Image: the final image
    :returns tuple[int, int, int, int]: how much the image was expanded by (if auto_expand is False, this is always (0, 0, 0, 0))
    """
    mask, expandedby = strokemask(img, radius, auto_expand, threshold)
    dest = (expandedby[0], expandedby[2])
    outimg = Image.new(mode="RGBA", size=mask.size, color=color)
    outimg.putalpha(mask)
    outimg.alpha_composite(img, dest=dest)
    return (outimg, expandedby)
