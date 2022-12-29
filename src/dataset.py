import os
import PIL
from .transforms import *
import torch
from sklearn.preprocessing import LabelEncoder

def cv2_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def pil_loader(path):
    img = PIL.Image.open(path).convert("RGB")
    return img



class BCDDataset:
    def __init__(self, root, df, aux_cols, target, test=False, in_chans=1, extension="png", transform=None, breast_crop=False, return_path=False):
        self.path = [os.path.join(root, f"{p}_{im}.{extension}") for p, im in zip(df["patient_id"].values, df["image_id"].values)]

        self.target = df[target].values
        self.test = test
        self.transform = transform
        
        self.aux_cols = [target]+aux_cols
        self.return_path = return_path
        self.in_chans = in_chans
        
        self.df_col = df.loc[:, self.aux_cols]
        self.df_col.index = np.arange(len(self.df_col))
        self.breast_crop = breast_crop
    

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image = cv2.imread(self.path[index], 0)

        if self.breast_crop:
            image = crop_breast(image)

        if self.in_chans == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.test:
            if self.return_path:
                return image, self.path[index]
            return image
        

        if self.return_path:
            return image, torch.from_numpy(self.df_col.values[index, :]), self.path[index]

        return image, torch.from_numpy(self.df_col.values[index, :])
