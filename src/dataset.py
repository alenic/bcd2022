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
    def __init__(self, root, df, multi_cols=None, test=False, extension="png", transform=None):
        self.path = [os.path.join(root, f"{p}_{im}.{extension}") for p, im in zip(df["patient_id"].values, df["image_id"].values)]
        if not test:
            self.label = df["cancer"].values
        
        self.test = test
        self.transform = transform
        
        self.multi_cols = multi_cols
        

        if self.multi_cols is not None:
            self.df_col = df.loc[:, multi_cols]
            for col in self.multi_cols:
                le = LabelEncoder()
                self.df_col[col] = le.fit_transform(self.df_col[col].values)
    

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image = cv2.imread(self.path[index], 0)
        image = crop_breast(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.test:
            return image
        
        if self.multi_cols is not None:
            return image, torch.from_numpy(self.df_col.values[index, :])


        return image, torch.tensor(self.label[index], dtype=torch.float32)





