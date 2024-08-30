import torch
import os
import pdb
import torch.nn.functional as F
from torchvision import transforms as v2
from PIL import Image
from kornia.augmentation import (
    CenterCrop,
    ColorJiggle,
    ColorJitter,
    PadTo,
    RandomAffine,
    RandomBoxBlur,
    RandomBrightness,
    RandomChannelShuffle,
    RandomContrast,
    RandomCrop,
    RandomCutMixV2,
    RandomElasticTransform,
    RandomEqualize,
    RandomErasing,
    RandomFisheye,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomHue,
    RandomInvert,
    RandomJigsaw,
    RandomMixUpV2,
    RandomMosaic,
    RandomMotionBlur,
    RandomPerspective,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomPosterize,
    RandomResizedCrop,
    RandomRGBShift,
    RandomRotation,
    RandomSaturation,
    RandomSharpness,
    RandomSolarize,
    RandomThinPlateSpline,
    RandomVerticalFlip,
    RandomShear,
    Resize,
    RandomShear
)
import numpy as np
from torchvision import transforms

class EvolutionDiversity:
    # def __init__(self, cross_n=1, crossover_rate=0.3, mutation_rate=0.1, cross_factor=0.1, mutation_factor=0.03):
    # def __init__(self, cross_n=1, crossover_rate=0.3, mutation_rate=0.05, cross_factor=0.1, mutation_factor=0.03):
    # def __init__(self, cross_n=3, mutation_n=1, crossover_rate=0.3, mutation_rate=0.05, cross_factor=0.1, mutation_factor=0.03):
    def __init__(self, cross_n=3, mutation_n=1, crossover_rate=0.3, mutation_rate=0.05, cross_factor=0.1, mutation_factor=0.03):
        self.cross_n = cross_n
        self.mutation_n = mutation_n
        self.crossover_rate = crossover_rate 
        self.mutation_rate = mutation_rate
        # self.mutation_choice = 'blend'
        self.mutation_choice = 'attach'
        # self.mutation_choice = 'constant'
        self.cross_choice = 'blend'
        self.cross_factor = cross_factor
        self.mutation_factor = mutation_factor

    def _mutation(self, imgs):
        new_imgs = imgs.clone()
        indexes = torch.randperm(len(imgs))
        for i in range(imgs.shape[0]):
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), box_rate=self.mutation_rate)
            if self.mutation_choice == 'attach':
                new_imgs[i][:, bbx1:bbx2, bby1:bby2] = torch.rand_like(imgs[i][:, bbx1:bbx2, bby1:bby2])
            elif self.mutation_choice == 'blend':
                new_imgs[i][:, bbx1:bbx2, bby1:bby2] = self.mutation_factor * torch.rand_like(imgs[i][:, bbx1:bbx2, bby1:bby2]) + \
                        (1 - self.mutation_factor) * imgs[i][:, bbx1:bbx2, bby1:bby2]
            elif self.mutation_choice == 'constant':
                new_imgs[i][:, bbx1:bbx2, bby1:bby2] = torch.ones_like(imgs[i][:, bbx1:bbx2, bby1:bby2]) * np.random.rand()
        return new_imgs

    def _crossover(self, imgs):
        indexes = torch.randperm(len(imgs))
        _, _, W, H = imgs.shape
        new_imgs = imgs.clone()
        for i in range(imgs.shape[0]):
            if self.cross_choice == 'joint':
                v = np.random.rand() < 0.5
                left = np.random.rand() < 0.5
                if v and left:
                    w = np.random.randint(0, W) 
                    new_imgs[i][:, 0:w, :] = imgs[indexes][i][:, 0:w, :] 
                elif v and not left:
                    w = np.random.randint(0, W) 
                    new_imgs[i][:, -w:, :] = imgs[indexes][i][:, -w:, :] 
                elif not v and left:
                    h = np.random.randint(0, H) 
                    new_imgs[i][:, :, 0:h] = imgs[indexes][i][:, :, 0:h]
                elif not v and not left:
                    h = np.random.randint(0, H) 
                    new_imgs[i][:, :, -h:] = imgs[indexes][i][:, :, -h:]
            elif self.cross_choice == 'blend':
                new_imgs[i] = imgs[i] * (1 - self.cross_factor) + imgs[indexes][i] * self.cross_factor
            elif self.cross_choice == 'cross_blend':
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), box_rate=self.crossover_rate)
                bbx3, bby3, bbx4, bby4 = self.rand_bbox(imgs.size(), ws=bbx2-bbx1, hs=bby2-bby1) 
                new_imgs[i][:, bbx1:bbx2, bby1:bby2] = \
                        imgs[i][:, bbx1:bbx2, bby1:bby2] * (1 - self.cross_factor) + \
                        imgs[indexes][i][:, bbx3:bbx4, bby3:bby4] * self.cross_factor

            elif self.cross_choice == 'pixel':
                cross_mask = torch.rand_like(imgs[i]) < self.crossover_rate
                new_imgs[i][cross_mask] = imgs[indexes][i][cross_mask]

        return new_imgs 

    def rand_bbox(self, size, ws=None, hs=None, box_rate=0.3):
        W = size[2]
        H = size[3]
        if ws is None or hs is None:
            cut_rat = box_rate 
            cut_w = np.int32(W * cut_rat)
            cut_h = np.int32(H * cut_rat)
        else:
            cut_w = ws
            cut_h = hs

        # uniform
        if ws is not None and hs is not None:
            cx = np.random.randint(W - ws)
            cy = np.random.randint(H - hs)

            bbx1 = cx
            bbx2 = cx + ws 
            bby1 = cy
            bby2 = cy + hs
        else:
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, imgs):
        # crossover
        new_imgs = imgs
        for _ in range(self.cross_n):
            new_imgs = self._crossover(new_imgs)
        # mutation 
        for _ in range(self.mutation_n):
            new_imgs = self._mutation(new_imgs)
        return new_imgs

class EvolutionDiversityExt:
    def __init__(self, mute_rate=0.5, scale_factor=0.1, inv_ratio=0.05): 
        self.mute_rate = mute_rate
        self.scale_factor = scale_factor
        self.inv_ratio = inv_ratio

    def _mutation(self, imgs_a, imgs_b, imgs_c):
        with torch.no_grad():
            mute_mask = torch.rand_like(imgs_a) < self.mute_rate
            choice_mask = torch.rand_like(imgs_a)
            inv_mask = mute_mask & (choice_mask < self.inv_ratio)
            scale_mask = mute_mask & (choice_mask >= self.inv_ratio)

        new_imgs = imgs_a.clone()
        new_imgs[scale_mask] = imgs_a[scale_mask] + \
                self.scale_factor * (imgs_b[scale_mask] - imgs_c[scale_mask])
        new_imgs[inv_mask] = 1.0 - imgs_a[inv_mask]
        return torch.clamp(new_imgs, 0, 1)

    def __call__(self, imgs):
        length = len(imgs)
        indexes1 = torch.randperm(length)
        indexes2 = torch.randperm(length)
        # mutation 
        new_imgs = self._mutation(imgs, imgs[indexes1], imgs[indexes2])
        return new_imgs


class RandomResize:
    def __init__(self, resize_ratio, p):
        self.resize_ratio = resize_ratio
        self.p = p

    def __call__(self, imgs):
        choices = np.random.rand(len(imgs)) < self.p
        if np.sum(choices) > 0:
            width = imgs.shape[-1]
            resize = int(width * self.resize_ratio)

            rndsize = np.random.randint(min(width, resize), max(width, resize))
            resize_method = Resize([rndsize, rndsize])
            pad_method = PadTo(imgs.shape[-2:], 'constant', 0, keepdim=False)

            imgs[choices] = pad_method(resize_method(imgs[choices]))
        return imgs

class MySelf:
    def __init__(self):
        pass

    def __call__(self, imgs):
        return imgs

class RandomLocalMix:
    def __init__(self, mix_num=1, alpha=0.4, ratio=0.7):
        self.mix_num = mix_num
        self.alpha = alpha
        self.ratio = ratio

    def localmix(self, imgs):
        out_imgs = imgs.clone()
        length = len(imgs)
        sz = imgs.size()
        for _ in range(self.mix_num):
            indexes = torch.randperm(length)
            new_imgs = out_imgs[indexes]
            for i in range(length):
                lam = np.random.beta(self.alpha, self.alpha)
                lam = max(lam, 1 - lam)

                # (1 - lam) * new_imgs[i][:, bbx3:bbx4, bby3:bby4]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(sz, lam)
                bbx3, bby3, bbx4, bby4 = self.rand_bbox(sz, lam, bbx2-bbx1, bby2-bby1)
                out_imgs[i][:, bbx1:bbx2, bby1:bby2] = lam * out_imgs[i][:, bbx1:bbx2, bby1:bby2] + \
                       (1 - lam) * new_imgs[i][:, bbx3:bbx4, bby3:bby4]

        return out_imgs

    def rand_bbox(self, size, lam, ws=None, hs=None):
        W = size[2]
        H = size[3]
        if ws is None or hs is None:
            # cut_rat = np.sqrt(1. - lam)
            cut_rat = self.ratio 
            cut_w = np.int32(W * cut_rat)
            cut_h = np.int32(H * cut_rat)
        else:
            cut_w = ws
            cut_h = hs

        # uniform
        if ws is not None and hs is not None:
            cx = np.random.randint(W - ws)
            cy = np.random.randint(H - hs)

            bbx1 = cx
            bbx2 = cx + ws 
            bby1 = cy
            bby2 = cy + hs
        else:
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, imgs):
        return self.localmix(imgs)

class RandomMix:
    def __init__(self, alpha=0.4, ratio=0.7):
        self.alpha = alpha
        self.ratio = ratio

    def localmix(self, front_imgs, back_imgs):
        length = len(back_imgs)
        indexes = torch.randperm(length)
        lam = max(self.alpha, 1.0 - self.alpha)
        out_imgs = lam * back_imgs + (1 - lam) * front_imgs[indexes]

        return out_imgs

    def __call__(self, front_imgs, back_imgs):
        return self.localmix(front_imgs, back_imgs)

class BlendMix:
    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def localmix(self, front_imgs, back_imgs):
        length = len(back_imgs)
        indexes = torch.randperm(length)
        out_imgs = alpha * front_imgs + (1 - alpha) * back_imgs[indexes]
        return out_imgs

    def __call__(self, front_imgs, back_imgs):
        return self.localmix(front_imgs, back_imgs)

class RandomFill:
    def __init__(self, fill_rate=(0.01, 0.05)):
        self.fill_rate = fill_rate 

    def _fill(self, imgs):
        new_imgs = imgs.clone()
        for i in range(len(imgs)):
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), 
                    box_rate=np.random.uniform(self.fill_rate[0], self.fill_rate[1]))
            new_imgs[i][:, bbx1:bbx2, bby1:bby2] = torch.zeros_like(imgs[i][:, bbx1:bbx2, bby1:bby2])
        return new_imgs

    def rand_bbox(self, size, ws=None, hs=None, box_rate=0.3):
        W = size[2]
        H = size[3]
        if ws is None or hs is None:
            cut_rat = box_rate 
            cut_w = np.int32(W * cut_rat)
            cut_h = np.int32(H * cut_rat)
        else:
            cut_w = ws
            cut_h = hs

        # uniform
        if ws is not None and hs is not None:
            cx = np.random.randint(W - ws)
            cy = np.random.randint(H - hs)

            bbx1 = cx
            bbx2 = cx + ws 
            bby1 = cy
            bby2 = cy + hs
        else:
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, imgs):
        new_imgs = self._fill(imgs)
        return new_imgs

class IconMix:
    def __init__(self, icon_path='/home/zero/zero/adversarial_examples/adversarial_examples/libraries/adversarial-attacks-pytorch-master/icons/icons/choices', width=299, height=299):
        self.icon_path = icon_path
        fnames = os.listdir(self.icon_path)
        self.fnames = [os.path.join(self.icon_path, fname) for fname in fnames if fname.endswith('.png')]
        self.transform = transforms.ToTensor()
        self.width, self.height = width, height

    def read_icon(self, fpath):
        img = Image.open(fpath)
        tensor = self.transform(img)
        return tensor[:3, :, :]

    def mix(self, imgs):
        new_imgs = imgs.clone()
        for i in range(len(imgs)):
            fpath = np.random.choice(self.fnames)
            icon = self.read_icon(fpath)
            x, y = np.random.randint(0, self.width-32), np.random.randint(0, self.height-32)
            new_imgs[i, :, x:x+32, y:y+32] = icon
        return new_imgs

class TransformWrapper:
    def __init__(self, aug_p=1.0):
        self.scale_dict = {
            'perspective': [0.1, 0.5],
            'spline': [0.1, 0.3],
            'rotation': [10.0, 30.0],
            'resize': [0.7, 1.0],
            'erasing': [0.01, 0.04],
            # 'gaussian_blur': [3, 22], 
            'gaussian_blur': [5, 21], 
            'gaussian_noise': [0.03, 0.1],
            'motion_blur': [5, 21], 
        }
        self.geo_dict = {
        #    'perspective': self._random_perspective,
            'spline': self._random_thin_plate_spline,
            'resize': self._random_resize,
            'rotation': self._random_rotation,
        }
        self.tran_dict = {
            'perspective': self._random_perspective,
            'spline': self._random_thin_plate_spline,
            'resize': self._random_resize,
            'affine': self._random_affine,
            'shear': self._random_shear,
            'rotation': self._random_rotation,
            'hflip': self._random_hflip,
            'vflip': self._random_vflip,
            'erasing': self._random_erasing,
            'gaussian_blur': self._random_gaussian_blur,
            'gaussian_noise': self._random_gaussian_noise,
            'motion_blur': self._random_motion_blur,
        }
        self.aug_p = aug_p

    def _random_affine(self, imgs, v):
        trans = RandomAffine((-1.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest", # resample
                  padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p,)
        return trans(imgs)

    def _random_shear(self, imgs, v):
        trans = RandomShear((-5., 2., 5., 10.), same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_perspective(self, imgs, v):
        trans = RandomPerspective(v, "nearest", align_corners=True, same_on_batch=False, 
                keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_thin_plate_spline(self, imgs, v):
        trans = RandomThinPlateSpline(v, align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_resize(self, imgs, v):
        trans = RandomResize(v, p=self.aug_p)
        return trans(imgs)

    def _random_hflip(self, imgs, v):
        trans = RandomHorizontalFlip(same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_vflip(self, imgs, v):
        trans = RandomVerticalFlip(same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_rotation(self, imgs, v):
        trans = RandomRotation(v, "nearest", align_corners=True, 
                same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_erasing(self, imgs, v):
        trans = RandomErasing(scale=(0.01, v), ratio=(0.3, 1.0), value=1, same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_gaussian_blur(self, imgs, v):
        trans = RandomGaussianBlur((v, v), (0.2, 1.3), 'reflect', same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_gaussian_noise(self, imgs, v):
        trans = RandomGaussianNoise(mean=0.0, std=v, same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def _random_motion_blur(self, imgs, v):
        trans = RandomMotionBlur((v, v), 35.0, 0.5, 'reflect', 'nearest', same_on_batch=False, keepdim=False, p=self.aug_p)
        return trans(imgs)

    def transform(self, imgs, trans, vs, short_cut=False):
        for i, (tran, v) in enumerate(zip(trans, vs)):
            try:
                if tran in self.scale_dict:
                    lower, upper = self.scale_dict[tran]
                    v = v * (upper - lower) + lower 
                if 'blur' in tran:
                    v = int(v + 0.5)
                    v = v if v % 2 == 1 else v - 1 # ensuring kernel size to be odd

                imgs = self.tran_dict[tran](imgs, v)
                if short_cut and i + 2 == len(trans):
                    prev_imgs = imgs
                elif short_cut and i + 1 == len(trans):
                    w = np.random.beta(0.4, 0.4)
                    w = max(w, 1.0 - w)
                    imgs = w * imgs + (1.0 - w) * prev_imgs
            except Exception as e:
                pdb.set_trace()
                print('debug')
        return imgs

class SimpleTransformWrapper:
    def __init__(self, p=0.8):
        self.tran_dict = {
            'spline': self._random_spline(p),
            'perspective': self._random_perspective(p),
            'resize': self._random_resize(p),
            'rotate': self._random_rotate(p),
            'hflip': self._random_hflip(p),
            'vflip': self._random_vflip(p),
            'gaussian_blur': self._random_gaussian_blur(p),
            'gaussian_noise': self._random_gaussian_noise(p),
            'erasing': self._random_erasing(p),
            'shearx': self._random_shearx(p),
            'sheary': self._random_sheary(p),
        }
        self.p = p

    def _random_spline(self, p=0.5, mag=0.3):
        return RandomThinPlateSpline(mag, align_corners=True, same_on_batch=False, keepdim=False, p=p)

    def _random_hflip(self, p=0.5):
        return v2.RandomHorizontalFlip(p)

    def _random_vflip(self, p=0.5):
        return v2.RandomVerticalFlip(p) 
    
    def _random_rotate(self, p=0.5, degree=30):
        return v2.RandomRotation(degrees=30) 

    def _random_perspective(self, p=0.5, scale=0.15):
        return v2.RandomPerspective(distortion_scale=scale, p=p)

    def _random_gaussian_blur(self, p=0.5, kernel_size=(21, 21)):
        return v2.GaussianBlur(kernel_size=kernel_size)

    def _random_erasing(self, p=0.5, scale=(0.01, 0.04), ratio=(0.3, 1.0)):
        return v2.RandomErasing(p=p, scale=scale, ratio=ratio)

    def _random_resize(self, p=0.5, scale_factor=0.8):
        def resize(imgs):
            if np.random.rand() < p:
                img_size = int(imgs.shape[-1] * scale_factor)
                img_resize = imgs.shape[-1]

                rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
                new_imgs = F.interpolate(
                    imgs, size=[rnd, rnd], mode="bilinear", align_corners=False
                )
                h_rem = img_resize - rnd
                w_rem = img_resize - rnd
                pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
                pad_bottom = h_rem - pad_top
                pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
                pad_right = w_rem - pad_left

                new_imgs = F.pad(
                    new_imgs,
                    [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
                    value=0,
                )
                return new_imgs
            else:
                return imgs
        return resize

    def _random_gaussian_noise(self, p=0.5, mean=0.0, std=0.1):
        def noise(imgs):
            if np.random.rand() < p:
                new_imgs = imgs + (torch.randn_like(imgs) * std + mean)
                return torch.clamp(new_imgs, 0, 1)
            else:
                return imgs
        return noise

    def _random_shearx(self, p=0.5, mag=15.0):
        def shearx(imgs):
            if np.random.rand() < p:
                return v2.functional.affine(imgs,
                        angle=0.0, translate=[0, 0],
                        scale=1.0, shear=[mag, 0.0])
            else:
                return imgs

        return shearx

    def _random_sheary(self, p=0.5, mag=15.0):
        def sheary(imgs):
            if np.random.rand() < p:
                return v2.functional.affine(imgs,
                        angle=0.0, translate=[0, 0],
                        scale=1.0, shear=[0.0, mag])
            else:
                return imgs

        return sheary

    def transform(self, imgs, trans):
        for i, tran in enumerate(trans):
            imgs = self.tran_dict[tran](imgs)
        return imgs
