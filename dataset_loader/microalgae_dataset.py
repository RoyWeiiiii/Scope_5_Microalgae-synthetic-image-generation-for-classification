import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from constants.labels import CLASS_NAME_TO_LABELS_MAP
from PIL import Image

class MicroalgaeDataset(Dataset):
    def __init__(self, dataset_path: str, transforms = None, weight_transforms = None, has_classes=True):
        self._dataset_path = dataset_path
        self._transforms = transforms
        self._weight_transforms = weight_transforms
        self._has_classes = has_classes
        self._data = self.__prepare_dataset()
        
    def __prepare_dataset(self):
        dataset = []
        if self._has_classes == True:
            for class_name in os.listdir(self._dataset_path):
                specified_images = os.path.join(self._dataset_path, class_name)
                for image_name in tqdm(os.listdir(specified_images), desc=f"Class: {class_name}"):
                    specified_image_path = os.path.join(specified_images, image_name)
                    details = (specified_image_path, CLASS_NAME_TO_LABELS_MAP[class_name])
                    dataset.append(details)
        else:
            for image_name in tqdm (os.listdir(self._dataset_path)):
                specified_image_path = os.path.join(self._dataset_path, image_name)
                dataset.append(specified_image_path)
        return dataset
            
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        specified_image_path, label = None, None
        if isinstance(self._data[index], tuple):
            specified_image_path, label = self._data[index]
        else:
            specified_image_path = self._data[index]
            label = specified_image_path
        
        current_image = Image.open(specified_image_path)
        image = None
        if self._transforms:
            compose =  Compose(self._transforms)
            image = compose(current_image)
        elif self._weight_transforms:
            image = self._weight_transforms(current_image)
        current_image.close()
        return image, label
