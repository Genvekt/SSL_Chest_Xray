from torchvision import transforms
from pl_bolts.transforms.self_supervised import Patchify
import numpy as np
from transforms.finetuning import SquarePad

# MOCO

class Moco2TrainTransforms:
    """
    Moco 2 modified  augmentation:
    Classic: https://arxiv.org/pdf/2003.04297.pdf
    Chest:   https://arxiv.org/pdf/2010.05352v1.pdf
    """

    def __init__(self, height: int = 128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((height,height)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(15, translate=(0.05,0.05), scale=(0.95, 1.05), shear=None, resample=0, fillcolor=0),
            transforms.ToTensor(),
            # ImageNet normalization
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
        
        q = self.train_transform(inp).contiguous()
        k = self.train_transform(inp).contiguous()
        return q, k


class Moco2ValTransforms:
    """
    Moco 2 modified  augmentation:
    Classic: https://arxiv.org/pdf/2003.04297.pdf
    Chest:   https://arxiv.org/pdf/2010.05352v1.pdf
    """

    def __init__(self, height: int = 128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.Resize((height,height)),
            transforms.ToTensor(),
            # ImageNet normalization
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
        
        q = self.train_transform(inp).contiguous()
        k = self.train_transform(inp).contiguous()
        return q, k    


#  CPCV2
class CPCTrainTransforms:
    """
    Transforms used for CPC:
    Transforms::
        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)
    Example::
        # in a regular dataset
        Imagenet(..., transforms=CPCTrainTransformsImageNet128())
        # in a DataModule
        module = ImagenetDataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCTrainTransformsImageNet128())
    """
    def __init__(self, patch_size: int = 32, overlap: int = 16, height=256):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        reshape = transforms.Resize((height,height))
        col_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)], p=0.8)

        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=self.patch_size, overlap_size=self.overlap),
        ])

        

        self.transforms = transforms.Compose([
            reshape,
            col_jitter,
            post_transform
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)

        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1


class CPCValTransforms:
    """
    Transforms used for CPC:
    Transforms::
        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)
    Example::
        # in a regular dataset
        Imagenet(..., transforms=CPCEvalTransformsImageNet128())
        # in a DataModule
        module = ImagenetDataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCEvalTransformsImageNet128())
    """

    def __init__(self, patch_size: int = 32, overlap: int = 16, height = 256):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.transforms = transforms.Compose([
            transforms.Resize((height,height)),
            post_transform
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
            
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1

class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform

        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ) -> None:

        
        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength)

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomRotation(20)
        ]


        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_height),
            transforms.RandomHorizontalFlip(), 
            self.final_transform
        ])

    def __call__(self, inp):
        transform = self.train_transform

        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)

        xi = transform(inp)
        xj = transform(inp)

        return xi, xj, self.online_transform(inp)


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """
    Transforms for SimCLR

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ):
        super().__init__(
            normalize=normalize,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose([
            transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
            transforms.CenterCrop(self.input_height),
            self.final_transform,
        ])