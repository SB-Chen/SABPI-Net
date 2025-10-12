from torch import nn
from torch.nn import functional as F
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp


class HFCFilter(nn.Module):
    def __init__(self,
                 # device,
                 filter_width=23,
                 nsig=20,
                 ratio=4,
                 sub_low_ratio=1,
                 sub_mask=False,
                 is_clamp=True):
        super(HFCFilter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(
            # device,
            filter_width, nsig=nsig)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = ratio
        self.sub_low_ratio = sub_low_ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)
        gaussian_output = self.gaussian_filter(x)
        res = self.ratio * (x - self.sub_low_ratio * gaussian_output)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1

        return res

def get_kernel(kernel_len=16, nsig=10):
    # nsig:standard deviation
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel


class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # Obtain Gaussian convolution kernel
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))

    def forward(self, x):
        x = self.padding(x)
        # Apply convolution to each of the three channels separately.
        res = []
        for i in range(x.shape[1]):
            res.append(F.conv2d(x[:, i:i+1], self.weight))
        x_output = torch.cat(res, dim=1)
        return x_output


def hfc_mul_mask_list(hfc_filter_list, image, mask):
    res_list = []
    for hfc_filter in hfc_filter_list:
        hfc = hfc_filter(image, mask)
        res = (hfc + 1) * mask - 1
        res_list.append(res)
    concat_res = torch.cat(res_list, dim=1)
    return concat_res

def tensor2im(input_image, imtype=np.uint8, guide=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy,
                                    (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    if guide:
        temp = image_numpy.astype(imtype)
        return temp[:, :, :3]
    else:
        return image_numpy.astype(imtype)

if __name__ == '__main__':
    img_path = ''
    mask_path = ''
    img_save_path = ''
    sub_low_ratio = 1
    filters_width_list = [9, 19, 29]
    nsig_list = [5.0, 9.0, 13.0]
    hfc_filter_list = [
        HFCFilter(w, s, sub_low_ratio=sub_low_ratio, sub_mask=True, is_clamp=True)
        for w, s in zip(filters_width_list, nsig_list)
    ]
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    img_transforms = transforms.Compose([transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.ToTensor()])
    img = img_transforms(img).unsqueeze(0)
    mask = mask_transform(mask).unsqueeze(0)
    result_hfc = hfc_mul_mask_list(hfc_filter_list, img, mask)
    print(f"result_hfc:{result_hfc.shape}")
    save_img_tensor = result_hfc[:, 6:, :, :]
    image_numpy = tensor2im(save_img_tensor)
    image_pil = Image.fromarray(image_numpy)
    # save image
    image_pil.save(img_save_path)
