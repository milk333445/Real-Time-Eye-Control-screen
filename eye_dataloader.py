import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image



class EyeDataset(Dataset):
    def __init__(self, directory, predict="x", resize=(448, 448)):
        self.directory = directory
        self.predict = predict
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
        ])
        self.resize = resize
        self.filenames = os.listdir(directory)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.directory, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resize)
        
        
        parts = img_name[:-4].split("_")
        x_label = float(parts[0][1:])
        y_label = float(parts[1][1:])
        labels = torch.Tensor([x_label if self.predict == "x" else y_label])
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    
def create_eye_dataloader(directory, predict, batch_size=32, resize=(448, 448), test_split=0.5, shuffle=True):
    if isinstance(resize, int):
        resize = (resize, resize)
    dataset = EyeDataset(directory, predict, resize)
    
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size   
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, test_dataloader
        
        
if __name__ == "__main__":
    train_dataloader, test_dataloader = create_eye_dataloader("eye_images", "x", resize=(224, 224))
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break