from torch.utils import data
from typing import List
import torch
import random
import pandas as pd
from PIL import Image
import os
import numpy as np
import os.path as osp
from scipy.fft import fft2, ifft2
from data_loading.data_transforms import (Compose, Resize_and_RandomCrop, RandomHorizontalFlip,  ToTensor, Normalize,  RandomVerticalFlip, Resize)
from data_loading.get_HFC import HFCFilter, hfc_mul_mask_list


class DatasetStats:
    STATS = {
        # Calculate the mean and standard deviation based on the custom dataset.
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    @classmethod
    def get_stats(cls, dataset_name):
        if dataset_name not in cls.STATS:
            raise ValueError(f"Dataset {dataset_name} not found in stats")
        return cls.STATS[dataset_name]['mean'], cls.STATS[dataset_name]['std']

    @classmethod
    def add_stats(cls, dataset_name, mean, std):
        cls.STATS[dataset_name] = {
            'mean': mean,
            'std': std
        }

def default_train_transforms(img_size=224,
                             flip_prob=0.5,
                             mean=None,
                             std=None):

    trans = [Resize_and_RandomCrop(size=img_size, target_fill=0),
             RandomHorizontalFlip(flip_prob),
             RandomVerticalFlip(flip_prob),
             ToTensor()]
    if mean is not None and std is not None:
        trans.append(Normalize(mean=mean, std=std))
    train_transforms = Compose(trans)
    return train_transforms

def default_test_transforms(img_size=224,
                            mean=None,
                            std=None):
    trans = [Resize(size=(img_size, img_size)),
             ToTensor()]
    if mean is not None and std is not None:
        trans.append(Normalize(mean=mean, std=std))
    test_transforms = Compose(trans)
    return test_transforms


# Build a base class for data loading
class LoadData4ExcelFile(data.Dataset):
    def __init__(self,
                 excel_file_info: dict,
                 img_path: str,
                 img_size=224,
                 train_flag=True,
                 transform=None,
                 args=None):
        super(LoadData4ExcelFile, self).__init__()
        # excel_file_info:{'file_path':str, 'sheet_name':str, 'column_name':['img_name', 'img_label']]}
        assert (isinstance(excel_file_info, dict) and 'file_path' in excel_file_info
                and 'sheet_name' in excel_file_info
                and 'column_name' in excel_file_info)
        self.excel_file_info = excel_file_info
        self.img_path = osp.join(img_path, 'image')
        self.mask_path = osp.join(img_path, 'mask')
        self.img_size = img_size
        self.train_flag = train_flag
        self.transform = transform
        self.num_classes = args.num_classes
        self.excel_data = self.read_excel_data()
        self.targets = [data_dict[excel_file_info['column_name'][1]] for data_dict in self.excel_data]
        self.data_names = [data_dict[excel_file_info['column_name'][0]] for data_dict in self.excel_data]
        self.dataset_name = args.dataset_name

    def read_excel_data(self) -> List:
        """
            'column_name' = ['imgName', 'label']
        """
        final_data = []
        excel_file_path = self.excel_file_info['file_path']
        excel_data = pd.read_excel(excel_file_path, sheet_name=self.excel_file_info['sheet_name'])
        excel_data = excel_data.dropna(subset=self.excel_file_info['column_name'])
        excel_column_name = self.excel_file_info['column_name']
        excel_data_dict = dict()
        for column_name in excel_column_name:
            excel_data_dict[column_name] = excel_data.loc[:, column_name].values

        data_size = len(excel_data_dict[excel_column_name[0]])
        item_dict = {}
        for index in range(data_size):
            for column_name in excel_column_name:
                item_dict[column_name] = excel_data_dict[column_name][index]

            final_data.append(item_dict)
            item_dict = {}
        print(f'Find {data_size} data in {excel_file_path}')
        return final_data

    def _get_class_dict(self):
        # Summarize the indices of each category’s data separately.
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def __len__(self):
        return len(self.excel_data)

    @staticmethod
    def get_img_mask(path:str, img_name:str):
        img_mask_path = osp.join(path, img_name)
        img_mask = Image.open(img_mask_path).convert('L')
        img_mask = np.array(img_mask) / 255
        img_mask = Image.fromarray(img_mask)
        return img_mask

    @staticmethod
    def get_img_data(path:str, img_name:str):
        img_data = Image.open(osp.join(path, img_name))
        img_data = img_data.convert("RGB")
        if img_data.mode != 'RGB':
            img_data = img_data.convert("RGB")
            # raise ValueError("image: {} isn't RGB mode.".format(img_name))
        return img_data

    def get_transform_data(self, img_data, img_mask):
        if self.transform is not None:
            img_data = self.transform(img_data, img_mask)
        else:
            mean, std = DatasetStats.get_stats(self.dataset_name)
            if self.train_flag:
                # img_data, img_mask = default_train_transforms(img_size=self.img_size)(img_data,img_mask)
                img_data, img_mask = default_train_transforms(img_size=self.img_size, mean=mean, std=std)(img_data, img_mask)
            else:
                img_data, img_mask = default_test_transforms(img_size=self.img_size, mean=mean, std=std)(img_data, img_mask)
                # img_data, img_mask = default_test_transforms(img_size=self.img_size)(img_data, img_mask)
        return img_data, img_mask

    def __getitem__(self, index) -> List:
        img_name = self.data_names[index]
        img_label = self.targets[index]
        img_data = self.get_img_data(self.img_path, img_name)
        img_mask = self.get_img_mask(self.mask_path, img_name)
        img_data, img_mask = self.get_transform_data(img_data, img_mask)
        return [img_data, img_label, img_name]


class LowFrequencyBufferRGB:
    def __init__(self, buffer_size, low_freq_size):
        """
        buffer_size: Buffer size (FIFO strategy)
        low_freq_size: The size of the extracted low-frequency component, e.g., (10, 10)
        """
        self.buffer_size = buffer_size
        self.low_freq_size = low_freq_size  # (height, width)
        self.buffer = []  # (low_freq_R, low_freq_G, low_freq_B)

    def extract_low_freq(self, image_channel):
        """
        Extract the low-frequency components of a single-channel image.
        :param image_channel: Input 2D image channel (height, width)
        :return: Extracted low-frequency components (low_freq_size)
        """
        freq_data = fft2(image_channel)  # Perform a 2D FFT on the channel.
        freq_data = np.fft.fftshift(freq_data)  # Move the low frequencies to the center

        # Extract the low-frequency components
        h, w = self.low_freq_size
        center_h, center_w = freq_data.shape[0] // 2, freq_data.shape[1] // 2
        low_freq = freq_data[center_h - h // 2:center_h + h // 2, center_w - w // 2:center_w + w // 2]
        return low_freq

    def add_data(self, image):
        """
        Add the low-frequency components of the RGB image to the buffer.
        :param image: input RGB image (height, width, 3)
        """
        low_freq_R = self.extract_low_freq(image[:, :, 0])
        low_freq_G = self.extract_low_freq(image[:, :, 1])
        low_freq_B = self.extract_low_freq(image[:, :, 2])

        # Combine the low-frequency components into a three-channel term
        low_freq_combined = (low_freq_R, low_freq_G, low_freq_B)

        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Remove the oldest low-frequency component from the buffer, first‑in‑first‑out.
        self.buffer.append(low_freq_combined)

    def get_transformed_data(self, image):
        """
        Obtain the low-frequency component of the input RGB image and replace it with the low-frequency component in the buffer that has the greatest difference.
        :param image: input RGB image (height, width, 3)
        :return: The new RGB image after replacing the low-frequency components.
        """
        
        low_freq_R = self.extract_low_freq(image[:, :, 0])
        low_freq_G = self.extract_low_freq(image[:, :, 1])
        low_freq_B = self.extract_low_freq(image[:, :, 2])
        low_freq_data = (low_freq_R, low_freq_G, low_freq_B)

        
        if not self.buffer:
            return image

        # Find the low-frequency component with the greatest difference in the buffer
        max_diff = -1
        max_low_freq = None
        for buffer_low_freq in self.buffer:
            # Calculate the differences for the R, G, and B channels separately.
            diff_R = np.linalg.norm(low_freq_data[0] - buffer_low_freq[0])
            diff_G = np.linalg.norm(low_freq_data[1] - buffer_low_freq[1])
            diff_B = np.linalg.norm(low_freq_data[2] - buffer_low_freq[2])
            total_diff = diff_R + diff_G + diff_B

            if total_diff > max_diff:
                max_diff = total_diff
                max_low_freq = buffer_low_freq

        # Replace the low‑frequency component of each channel of the input image with the buffer’s low‑frequency component that has the greatest difference.
        transformed_image = np.zeros_like(image, dtype=np.float32)
        for i, (channel, max_channel_low_freq) in enumerate(
                zip((image[:, :, 0], image[:, :, 1], image[:, :, 2]), max_low_freq)):
            
            freq_data = fft2(channel)
            freq_data = np.fft.fftshift(freq_data)
            h, w = self.low_freq_size
            center_h, center_w = freq_data.shape[0] // 2, freq_data.shape[1] // 2
            freq_data[center_h - h // 2:center_h + h // 2, center_w - w // 2:center_w + w // 2] = max_channel_low_freq
            freq_data = np.fft.ifftshift(freq_data)  
            transformed_image[:, :, i] = np.abs(ifft2(freq_data))  

        return transformed_image


class Load_data_with_AIAM_and_LFBuffer(LoadData4ExcelFile):
    def __init__(self,
                 excel_file_info: dict,
                 img_path: str,
                 img_size=224,
                 train_flag=True,
                 transform=None,
                 args=None,
                 class_map=None,
                 class_balanced_data=None,
                 buffer_size=32,
                 low_freq_size=(3, 3)):
        super().__init__(excel_file_info, img_path, img_size, train_flag, transform, args)
        # For HFC
        self.filters_width_list = [9, 19, 29]
        self.nsig_list = [5.0, 9.0, 13.0]
        self.hfc_filter_list = [
            HFCFilter(w, s, sub_low_ratio=1, sub_mask=True, is_clamp=True)
            for w, s in zip(self.filters_width_list, self.nsig_list)
        ]
        # Create buffer object
        self.buffer = LowFrequencyBufferRGB(buffer_size, low_freq_size)
        self.class_balanced_data = class_balanced_data
        self.class_dict = self._get_class_dict()


    def __getitem__(self, index) -> List:
        img_name = self.data_names[index]
        img_label = self.targets[index]
        img_data = self.get_img_data(self.img_path, img_name)
        img_data_np = np.array(img_data)
        img_mask = self.get_img_mask(self.mask_path, img_name)
        img_data, img_mask_1 = self.get_transform_data(img_data, img_mask)

        img_mask_1 = img_mask_1.unsqueeze(0)
        img_data = img_data.unsqueeze(0)
        # print(f"img_torch:{img_torch.shape}, mask:{mask.shape}, img_hsv_torch:{img_hsv_torch.shape}")
        img_hfc = hfc_mul_mask_list(self.hfc_filter_list, img_data, img_mask_1)
        img_hfc = img_hfc[:, 6:, :, :]
        img_data = img_data.squeeze(0)
        img_hfc = img_hfc.squeeze(0)

        if not self.class_balanced_data:
            if self.train_flag:
                # Add image to buffer and get the transformed image
                self.buffer.add_data(img_data_np)
                transformed_image_np = self.buffer.get_transformed_data(img_data_np)
                
                img_data_changed_lf = Image.fromarray(np.clip(transformed_image_np, 0, 255).astype(np.uint8)) 
                img_data_changed_lf, img_mask_2 = self.get_transform_data(img_data_changed_lf, img_mask)

                # Transformed image
                img_mask_2 = img_mask_2.unsqueeze(0)
                img_data_changed_lf = img_data_changed_lf.unsqueeze(0)
                img_data_changed_lf_hfc = hfc_mul_mask_list(self.hfc_filter_list, img_data_changed_lf, img_mask_2)
                img_data_changed_lf_hfc = img_data_changed_lf_hfc[:, 6:, :, :]
                img_data_changed_lf = img_data_changed_lf.squeeze(0)
                img_data_changed_lf_hfc = img_data_changed_lf_hfc.squeeze(0)

            return [img_data, img_hfc, img_data_changed_lf, img_data_changed_lf_hfc, img_label, img_name] if self.train_flag \
                else [img_data, img_hfc, img_label, img_name]
        else:
            # Get class-balanced data for Tail finetuning strategy training
            # Randomly select a class of data, with each class having a sampling probability of (1/self.num_classes).
            sample_class = random.randint(0, self.num_classes - 1)
            # Get the image index of the selected category
            sample_indexes = self.class_dict[sample_class]
            # Randomly select an image from the data of the chosen category.
            sample_index = random.choice(sample_indexes)
            sample_img_name = self.data_names[sample_index]
            sample_img_label = self.targets[sample_index]
            sample_img_data = self.get_img_data(self.img_path, sample_img_name)
            sample_img_mask = self.get_img_mask(self.mask_path, sample_img_name)
            sample_img_data, sample_img_mask = self.get_transform_data(sample_img_data, sample_img_mask)

            sample_img_mask = sample_img_mask.unsqueeze(0)
            sample_img_data = sample_img_data.unsqueeze(0)
            sample_img_hfc = hfc_mul_mask_list(self.hfc_filter_list, sample_img_data, sample_img_mask)
            sample_img_hfc = sample_img_hfc[:, 6:, :, :]
            sample_img_data = sample_img_data.squeeze(0)
            sample_img_hfc = sample_img_hfc.squeeze(0)

            return [sample_img_data, sample_img_hfc, img_data, img_hfc, sample_img_label]
