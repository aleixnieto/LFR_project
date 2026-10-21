"""Strong-view augmentation matching the released SemiCLIP implementation."""

import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


PARAMETER_MAX = 10


def _float_parameter(value: int, maximum: float) -> float:
    return float(value) * maximum / PARAMETER_MAX


def _int_parameter(value: int, maximum: int) -> int:
    return int(value * maximum / PARAMETER_MAX)


def _signed(value: float) -> float:
    return -value if random.random() < 0.5 else value


def _autocontrast(image: Image.Image, _value: int, _maximum, _bias) -> Image.Image:
    return ImageOps.autocontrast(image)


def _brightness(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(
        _float_parameter(value, maximum) + bias
    )


def _color(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    return ImageEnhance.Color(image).enhance(_float_parameter(value, maximum) + bias)


def _contrast(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(
        _float_parameter(value, maximum) + bias
    )


def _equalize(image: Image.Image, _value: int, _maximum, _bias) -> Image.Image:
    return ImageOps.equalize(image)


def _identity(image: Image.Image, _value: int, _maximum, _bias) -> Image.Image:
    return image


def _posterize(image: Image.Image, value: int, maximum: int, bias: int) -> Image.Image:
    return ImageOps.posterize(image, _int_parameter(value, maximum) + bias)


def _rotate(image: Image.Image, value: int, maximum: int, bias: int) -> Image.Image:
    angle = _signed(_int_parameter(value, maximum) + bias)
    return image.rotate(angle)


def _sharpness(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(
        _float_parameter(value, maximum) + bias
    )


def _shear_x(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    shear = _signed(_float_parameter(value, maximum) + bias)
    return image.transform(image.size, Image.Transform.AFFINE, (1, shear, 0, 0, 1, 0))


def _shear_y(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    shear = _signed(_float_parameter(value, maximum) + bias)
    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, 0, shear, 1, 0))


def _solarize(image: Image.Image, value: int, maximum: int, bias: int) -> Image.Image:
    return ImageOps.solarize(image, 256 - (_int_parameter(value, maximum) + bias))


def _translate_x(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    offset = int(_signed(_float_parameter(value, maximum) + bias) * image.size[0])
    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, offset, 0, 1, 0))


def _translate_y(image: Image.Image, value: int, maximum: float, bias: float) -> Image.Image:
    offset = int(_signed(_float_parameter(value, maximum) + bias) * image.size[1])
    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, offset))


def _cutout(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    center_x = int(np.random.uniform(0, width))
    center_y = int(np.random.uniform(0, height))
    left = max(0, center_x - size // 2)
    top = max(0, center_y - size // 2)
    right = min(width, left + size)
    bottom = min(height, top + size)
    result = image.copy()
    ImageDraw.Draw(result).rectangle((left, top, right, bottom), fill=(127, 127, 127))
    return result


def _fixmatch_pool() -> tuple[tuple[Callable, float | int | None, float | int | None], ...]:
    return (
        (_autocontrast, None, None),
        (_brightness, 0.9, 0.05),
        (_color, 0.9, 0.05),
        (_contrast, 0.9, 0.05),
        (_equalize, None, None),
        (_identity, None, None),
        (_posterize, 4, 4),
        (_rotate, 30, 0),
        (_sharpness, 0.9, 0.05),
        (_shear_x, 0.3, 0),
        (_shear_y, 0.3, 0),
        (_solarize, 256, 0),
        (_translate_x, 0.3, 0),
        (_translate_y, 0.3, 0),
    )


class RandAugmentMC:
    """FixMatch RandAugment variant used by the SemiCLIP release."""

    def __init__(self, n: int, m: int) -> None:
        if n < 1:
            raise ValueError("n must be at least one")
        if not 1 <= m <= PARAMETER_MAX:
            raise ValueError(f"m must be between 1 and {PARAMETER_MAX}")
        self.n = n
        self.m = m
        self.augment_pool = _fixmatch_pool()

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("RandAugmentMC expects a PIL image")
        for operation, maximum, bias in random.choices(self.augment_pool, k=self.n):
            magnitude = int(np.random.randint(1, self.m)) if self.m > 1 else 1
            if random.random() < 0.5:
                image = operation(image, magnitude, maximum, bias)
        return _cutout(image, 16)
