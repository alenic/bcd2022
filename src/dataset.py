import os
import PIL
from .transforms import *

class BCDDataset:
    def __init__(self, root, df, test=False, extension="png", transform=None):
        self.path = [os.path.join(root, str(p), str(im)+"."+extension) for p, im in zip(df["patient_id"].values, df["image_id"].values)]
        if not test:
            self.label = df["cancer"].values
        
        self.test = test
        self.transform = transform
    

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        
        image = PIL.Image.open(self.path[index]).convert("L")
        image = crop_breast(image)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.test:
            return image
        
        return image, self.label[index]
