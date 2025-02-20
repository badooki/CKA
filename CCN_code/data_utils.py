import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset


def load_data(DATAROOT, session='train'):

    PROCESSED_ROOT = os.path.join(DATAROOT, 'TVSD', 'processed_data')
    os.makedirs(PROCESSED_ROOT, exist_ok=True)
    all_data_path = os.path.join(PROCESSED_ROOT, f'tvsd_processed_{session}.npz')

    if os.path.exists(all_data_path):
        # Load data_dict
        data_dict = np.load(all_data_path, allow_pickle=True)['data_dict'].tolist()
    else:
        # Generate data_dict
        data_dict = {name: get_tvsd_dataset(DATAROOT, name, session) for name in ['F', 'N']}
        np.savez(all_data_path, data_dict=data_dict)

    return data_dict


def get_tvsd_dataset(ROOT, monkey_name, session='train'):

    assert monkey_name in ['F', 'N'], 'Invalid monkey name'

    import mat73
    import zipfile
    import urllib

    DATASET_PATH = os.path.join(ROOT, 'TVSD')

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

        # URL of the zip file
        url = "https://gin.g-node.org/paolo_papale/TVSD/archive/master.zip"

        # Path to save the downloaded zip file
        zip_path = os.path.join(ROOT, "./master.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ROOT)

        # Rename the extracted folder
        old_folder_name = os.path.join(ROOT, "TVSD-master")
        new_folder_name = os.path.join(ROOT, "TVSD")
        os.rename(old_folder_name, new_folder_name)

    datapath = os.path.join(DATASET_PATH, f'monkey{monkey_name}')

    mat_data = mat73.loadmat(os.path.join(datapath, 'THINGS_normMUA.mat'))
    image_labels = mat73.loadmat(os.path.join(datapath, '_logs', 'things_imgs.mat'))

    if session == 'test':
        MUA = np.asarray(mat_data[f'{session}_MUA_reps'])
    else:
        MUA = np.asarray(mat_data[f'{session}_MUA'])
    SNR = np.array(mat_data['SNR'])
    arrayID = np.repeat(np.arange(16), 64)

    things_path = np.asarray(image_labels[f'{session}_imgs']['things_path'])
    classes = np.asarray(image_labels[f'{session}_imgs']['class'])

    # For monkey N, this plot should show that array 5 is empty.
    # So we need to create a mask for that monkey
    if monkey_name == 'N':

        rois = np.zeros(1024)  # V1
        rois[512:768] = 1  # V4
        rois[768:] = 2  # IT

        validArrays = arrayID != 5
        rois = rois[validArrays]
        arrayID = np.repeat(np.arange(15), 64)
        SNR = SNR[validArrays, :]
        MUA = MUA[validArrays, :]

    if monkey_name == 'F':

        rois = np.zeros(1024)  # ; % V1
        rois[512:832] = 2  # ; % IT
        rois[832:] = 1  # ; % V4

        idx = np.arange(1024)
        idx = np.concatenate([idx[rois == 0], idx[rois == 1], idx[rois == 2]])
        MUA = MUA[idx, :]
        rois = rois[idx]
        # DO NOT REORDER ARRAY ID.

    data_dict = {'V1': MUA[rois == 0, :].swapaxes(0, 1).tolist(),
                 'V4': list(MUA[rois == 1, :].swapaxes(0, 1)),
                 'IT': list(MUA[rois == 2, :].swapaxes(0, 1)),
                 'things_path': things_path,
                 'classes': classes}

    df = pd.DataFrame().from_dict(data_dict, orient='columns')

    dataV1 = np.asarray(df['V1'].tolist())
    dataV4 = np.asarray(df['V4'].tolist())
    dataIT = np.asarray(df['IT'].tolist())

    dataAll = np.concatenate([dataV1, dataV4, dataIT], axis=1)

    returns = dict(df=df, arrayID=arrayID, SNR=SNR, rois=rois, dataAll=dataAll)

    return returns


class THINGSDataset(Dataset):

    def __init__(self, root, path_dict, transform=None, target_transform=None, only_img=True):
        """
        Input must be of form path_dict = {concept: [path1, path2, ...], ...}

        Followed this link: https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html
        """

        self.classes = sorted(list(path_dict.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self.make_dataset(root, path_dict)
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform
        self.only_img = only_img

    def make_dataset(self, root, path_dict) -> List[Tuple[str, int]]:

        instances = []
        for target_class in sorted(self.class_to_idx.keys()):

            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)

            for fname in sorted(path_dict[target_class]):
                path = os.path.join(target_dir, fname)
                item = path, class_index
                instances.append(item)

        return instances

    def __getitem__(self, index):

        from torchvision.datasets.folder import default_loader

        path, target = self.samples[index]

        # sample = get_sample(path)
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.only_img:
            return sample
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def get_sample(path):

    from PIL import Image
    with open(path, "rb") as f:
        img = Image.open(f)
        sample = img.convert("RGB")

    return sample
