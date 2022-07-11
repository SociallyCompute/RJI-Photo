import os
from typing import Optional

from PIL import Image, ImageFilter, ImageEnhance


def blur_image(
        img_path: str,
        blur_type: str = "box",
        radius: int = 2,
        save_path: Optional[str] = None
) -> None:
    org_img = Image.open(img_path)
    if blur_type == "box":
        blur_img = org_img.filter(ImageFilter.BoxBlur(radius))
    elif blur_type == "gauss":
        blur_img = org_img.filter(ImageFilter.GaussianBlur(radius))
    else:
        raise NotImplementedError("Must specify a blur type of box or gauss")

    if save_path:
        blur_img.save(save_path)
    else:
        blur_img.save(os.path.splitext(img_path)[0] + '_{}_{}_blurred.JPG'.format(radius, blur_type))

    org_img.close()


def contrast_image(
        img_path: str,
        level: float = 2.0,
        save_path: Optional[str] = None
) -> None:
    org_img = Image.open(img_path)
    con_img = ImageEnhance.Contrast(org_img).enhance(level)
    if save_path:
        con_img.save(save_path)
    else:
        con_img.save(os.path.splitext(img_path)[0] + '_{}_contrast.JPG'.format(level))

    org_img.close()


def brightness_image(
        img_path: str,
        level: float = 2.0,
        save_path: Optional[str] = None
) -> None:
    org_img = Image.open(img_path)
    bri_img = ImageEnhance.Brightness(org_img).enhance(level)
    if save_path:
        bri_img.save(save_path)
    else:
        bri_img.save(os.path.splitext(img_path)[0] + '_{}_brightness.JPG'.format(level))

    org_img.close()


def coloring_image(
        img_path: str,
        level: float = 2.0,
        save_path: Optional[str] = None
) -> None:
    org_img = Image.open(img_path)
    col_img = ImageEnhance.Color(org_img).enhance(level)
    if save_path:
        col_img.save(save_path)
    else:
        col_img.save(os.path.splitext(img_path)[0] + '_{}_color.JPG'.format(level))

    org_img.close()
