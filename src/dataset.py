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



class BCDDatasetNPZ:
    def __init__(self,
                root,
                df, 
                aux_cols=[],
                target="cancer",
                test=False,
                in_chans=1,
                extension="npz",
                transform=None,
                breast_crop=True,
                return_breast_id=False,
                return_path=False):

        self.path = [os.path.join(root, f"{p}_{im}.{extension}") for p, im in zip(df["patient_id"].values, df["image_id"].values)]

        self.intepret = df["interpret"].values
        self.window_center = df["window_center"].values
        self.window_width = df["window_width"].values
        self.target = df[target].values
        self.test = test
        self.transform = transform
        self.breast_id = df["breast_id"].values
        
        self.aux_cols = [target]+aux_cols
        self.return_path = return_path
        self.in_chans = in_chans
        
        self.df_col = df.loc[:, self.aux_cols]
        self.df_col.index = np.arange(len(self.df_col))
        self.breast_crop = breast_crop
        self.return_breast_id = return_breast_id

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image = np.load(self.path[index])["data"]
        if self.intepret[index] == "MONOCHROME1":
            image = image.max()-image
        if self.breast_crop:
            image = crop_breast(image)
        #image = npz_preprocessing(image, self.intepret[index], (self.window_center[index], self.window_width[index]))

        image = npz_preprocessing(image)
        if self.in_chans == 3:
            image_medium =  npz_preprocessing(image, quant=(0.1, 0.9))
            image_high =  npz_preprocessing(image, quant=(0.5, 1))
            image3 = np.zeros(list(image.shape)+[3], dtype=np.uint8)
            image3[:, :, 0] = image
            image3[:, :, 1] = image_medium
            image3[:, :, 2] = image_high
            image = image3

        if self.transform is not None:
            image = self.transform(image)

        if self.test:
            if self.return_path:
                return image, self.path[index]
            return image
        
        return_list = []

        if self.return_breast_id:
            return_list += [self.breast_id[index]]

        if self.return_path:
            return_list += [self.path[index]]

        return (image, torch.from_numpy(self.df_col.values[index, :])) + tuple(return_list)




class BCDDataset:
    def __init__(self, root, df, aux_cols=[], target="cancer", test=False, in_chans=1, extension="png", transform=None, breast_crop=True, return_path=False):
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


