import torch
from PIL import Image
from torchvision import transforms


class Aug(object):
    def __init__(self, args):
        self.args = args
        self.crop, self.flip, self.jitter, self.randaug = True, True, False, False
        self.resized_crop = transforms.RandomResizedCrop(args.img_size, scale=(args.min_scale, 1))
        self.rand_flip = transforms.RandomHorizontalFlip()
        p = .1
        self.jitter_op = transforms.ColorJitter(p, p, p, p)
        self.post_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def test_aug(self, image, bs=16):
        inputs = []

        for i in range(bs):
            img = image
            if self.flip:
                img = self.rand_flip(img)

            if self.jitter:
                img2 = self.jitter_op(img)
                img = Image.blend(img, img2, alpha=0.5)

            if self.randaug:
                img = self.randaug_op.aug_batch([img])[0]

            if self.crop:
                img = self.resized_crop(img)

            img = self.post_t(img)
            inputs.append(img)

        inputs = torch.stack(inputs, 0)
        return inputs
