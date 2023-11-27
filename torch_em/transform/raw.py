import json
import numpy as np

import torch
from torchvision import transforms


#
# normalization functions
#


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def cast(inpt, typestring):
    if torch.is_tensor(inpt):
        assert typestring in TORCH_DTYPES, f"{typestring} not in TORCH_DTYPES"
        return inpt.to(TORCH_DTYPES[typestring])
    return inpt.astype(typestring)


def standardize(raw, mean=None, std=None, axis=None, eps=1e-7):
    raw = cast(raw, "float32")

    mean = raw.mean(axis=axis, keepdims=True) if mean is None else mean
    raw -= mean

    std = raw.std(axis=axis, keepdims=True) if std is None else std
    raw /= (std + eps)

    return raw


def _normalize_torch(tensor, minval, maxval, axis, eps):
    if axis:  # torch returns torch.return_types.min or torch.return_types.max
        minval = torch.amin(tensor, dim=axis, keepdim=True) if minval is None else minval
        tensor -= minval

        maxval = torch.amax(tensor, dim=axis, keepdim=True) if maxval is None else maxval
        tensor /= (maxval + eps)

        return tensor

    # keepdim can only be used in combination with dim
    minval = tensor.min() if minval is None else minval
    tensor -= minval

    maxval = tensor.max() if maxval is None else maxval
    tensor /= (maxval + eps)

    return tensor


def normalize(raw, minval=None, maxval=None, axis=None, eps=1e-7):
    raw = cast(raw, "float32")

    if torch.is_tensor(raw):
        return _normalize_torch(raw, minval=minval, maxval=maxval, axis=axis, eps=eps)

    minval = raw.min(axis=axis, keepdims=True) if minval is None else minval
    raw -= minval

    maxval = raw.max(axis=axis, keepdims=True) if maxval is None else maxval
    raw /= (maxval + eps)

    return raw


def normalize_percentile(raw, lower=1.0, upper=99.0, axis=None, eps=1e-7):
    v_lower = np.percentile(raw, lower, axis=axis, keepdims=True)
    v_upper = np.percentile(raw, upper, axis=axis, keepdims=True) - v_lower
    return normalize(raw, v_lower, v_upper, eps=eps)


#
# intensity augmentations / noise augmentations
#

# modified from https://github.com/kreshuklab/spoco/blob/main/spoco/transforms.py
class RandomContrast():
    """
    Adjust contrast by scaling image to `mean + alpha * (image - mean)`.
    """
    def __init__(self, alpha=(0.5, 2), mean=0.5, clip_kwargs={'a_min': 0, 'a_max': 1}):
        self.alpha = alpha
        self.mean = mean
        self.clip_kwargs = clip_kwargs

    def __call__(self, img):
        alpha = np.random.uniform(self.alpha[0], self.alpha[1])
        result = self.mean + alpha * (img - self.mean)
        if self.clip_kwargs:
            return np.clip(result, **self.clip_kwargs)
        return result


class AdditiveGaussianNoise():
    """
    Add random Gaussian noise to image.
    """
    def __init__(self, scale=(0.0, 0.3), clip_kwargs={'a_min': 0, 'a_max': 1}):
        self.scale = scale
        self.clip_kwargs = clip_kwargs

    def __call__(self, img):
        std = np.random.uniform(self.scale[0], self.scale[1])
        gaussian_noise = np.random.normal(0, std, size=img.shape)
        if self.clip_kwargs:
            return np.clip(img + gaussian_noise, 0, 1)
        return img + gaussian_noise


class AdditivePoissonNoise():
    """
    Add random Poisson noise to image.
    """
    # TODO: not sure if Poisson noise like this does make sense
    # for data that is already normalized
    def __init__(self, lam=(0.0, 0.1), clip_kwargs={'a_min': 0, 'a_max': 1}):
        self.lam = lam
        self.clip_kwargs = clip_kwargs

    def __call__(self, img):
        lam = np.random.uniform(self.lam[0], self.lam[1])
        poisson_noise = np.random.poisson(lam, size=img.shape) / lam
        if self.clip_kwargs:
            return np.clip(img + poisson_noise, 0, 1)
        return img + poisson_noise


class PoissonNoise():
    """
    Add random data-dependent Poisson noise to image.
    """
    def __init__(self, multiplier=(5.0, 10.0), clip_kwargs={'a_min': 0, 'a_max': 1}):
        self.multiplier = multiplier
        self.clip_kwargs = clip_kwargs

    def __call__(self, img):
        multiplier = np.random.uniform(self.multiplier[0], self.multiplier[1])
        offset = img.min()
        poisson_noise = np.random.poisson((img - offset) * multiplier)
        if isinstance(img, torch.Tensor):
            poisson_noise = torch.Tensor(poisson_noise)
        poisson_noise = poisson_noise / multiplier + offset
        if self.clip_kwargs:
            return np.clip(poisson_noise, **self.clip_kwargs)
        return poisson_noise


class GaussianBlur():
    """
    Blur the image.
    """
    def __init__(self, kernel_size=(2, 12), sigma=(0, 2.5)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        # sample kernel_size and make sure it is odd
        kernel_size = 2 * (np.random.randint(self.kernel_size[0], self.kernel_size[1]) // 2) + 1
        # switch boundaries to make sure 0 is excluded from sampling
        sigma = np.random.uniform(self.sigma[1], self.sigma[0])
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        out = transforms.GaussianBlur(kernel_size, sigma=sigma)(img)
        return out


#
# default transformation:
# apply intensity augmentations and normalize
#

class RawTransform:
    def __init__(self, normalizer, augmentation1=None, augmentation2=None):
        self.normalizer = normalizer
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def __call__(self, raw):
        if self.augmentation1 is not None:
            raw = self.augmentation1(raw)
        raw = self.normalizer(raw)
        if self.augmentation2 is not None:
            raw = self.augmentation2(raw)
        return raw


def get_raw_transform(normalizer=standardize, augmentation1=None, augmentation2=None):
    return RawTransform(normalizer,
                        augmentation1=augmentation1,
                        augmentation2=augmentation2)


# The default values are made for an image with pixel values in
# range [0, 1]. That the image is in this range is ensured by an
# initial normalizations step.
def get_default_mean_teacher_augmentations(
    p=0.3, norm=None,
    blur_kwargs=None, poisson_kwargs=None, gaussian_kwargs=None
):
    if norm is None:
        norm = normalize
    aug1 = transforms.Compose([
        norm,
        transforms.RandomApply([GaussianBlur(**({} if blur_kwargs is None else blur_kwargs))], p=p),
        transforms.RandomApply([PoissonNoise(**({} if poisson_kwargs is None else poisson_kwargs))], p=p/2),
        transforms.RandomApply([AdditiveGaussianNoise(**({} if gaussian_kwargs is None else gaussian_kwargs))], p=p/2),
    ])
    aug2 = transforms.RandomApply(
        [RandomContrast(clip_kwargs={"a_min": 0, "a_max": 1})], p=p
    )
    return get_raw_transform(
        normalizer=norm,
        augmentation1=aug1,
        augmentation2=aug2
    )


class nnUNetRawTransformBase:
    """nnUNetRawTransformBase is an interface to implement specific raw transforms for nnUNet.

    Adapted from: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/preprocessing/normalization
    """
    def __init__(
            self,
            plans_file: str,
            expected_dtype: type = np.float32,
            tolerance: float = 1e-8
    ):
        self.expected_dtype = expected_dtype
        self.tolerance = tolerance

        self.intensity_properties = self.load_json(plans_file)
        self.intensity_properties = self.intensity_properties["foreground_intensity_properties_per_channel"]

    def load_json(self, _file: str):
        # credits: `batchgenerators.utilities.file_and_folder_operations`
        with open(_file, 'r') as f:
            a = json.load(f)
        return a

    def __call__(
            self,
            raw: np.ndarray,
            modality: str
    ) -> np.ndarray:  # the transformed raw inputs
        """Returns the raw inputs after applying the pre-processing from nnUNet.

        Args:
            raw: The raw array inputs
                Expectd a float array of shape H * W * C

        Returns:
            The transformed raw inputs (the same shape as inputs)
        """
        raise NotImplementedError("It's a class template for raw transforms from nnUNet. \
                                  Use a child class that implements the expected raw transform instead")


class nnUNet_CT_RawTransform(nnUNetRawTransformBase):
    """Apply transformation on the raw inputs (adapted from nnUNetv2's `CTNormalization`)

    You can use this class to apply the necessary raw transformations on CT and PET volume channels.

    Here's an example for how to use this class:
    ```python
    # Initialize the raw transform.
    raw_transform = nnUNet_CT_RawTransform(plans_file="...nnUNetplans.json")

    # Apply transformation on the inputs.
    ct_raw = raw_transform(ct_volume)
    pet_raw = raw_transform(pet_volume)
    ```
    """
    def __call__(
            self,
            raw: np.ndarray,
            modality_index: str
    ) -> np.ndarray:
        assert self.intensity_properties is not None, \
            "Intensity properties are required here. Please make sure that you pass the `nnUNetplans.json correctly."

        raw = raw.astype(self.expected_dtype)

        # intensity properties for the respective modality
        props = self.intensity_properties[modality_index]

        mean = props['mean']
        std = props['std']
        lower_bound = props['percentile_00_5']
        upper_bound = props['percentile_99_5']
        raw = np.clip(raw, lower_bound, upper_bound)
        raw = (raw - mean) / max(std, self.tolerance)
        return raw
