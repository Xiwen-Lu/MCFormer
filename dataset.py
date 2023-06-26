from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import min,max,from_numpy
import numpy as np
import time
import os
import yaml

class ChannelDataset(Dataset):
    def __init__(self, datafilepath,batch_length=500,is_slide=False,transforms=None):
        self.batch_length = batch_length
        datasets = os.listdir(datafilepath)
        self.is_slide = is_slide
        self.transforms = transforms
        data = []
        for i in range(len(datasets)):
            data.append(np.load(os.path.join(datafilepath,datasets[i])))
        self.data = np.array(data).transpose([0,2,1]).reshape([-1,2])


    def __len__(self):
        if self.is_slide:
            return len(self.data)-self.batch_length+1
        else:
            return len(self.data)//self.batch_length

    def __getitem__(self, idx):
        if self.is_slide:
            label = from_numpy(self.data[idx:idx+self.batch_length,0])
            data = from_numpy(self.data[idx:idx+self.batch_length,1])
        else:
            label = from_numpy(self.data[idx*self.batch_length:(idx+1)*self.batch_length,0])
            data = from_numpy(self.data[idx*self.batch_length:(idx+1)*self.batch_length,1])
        if self.transforms:
            data = self.transforms(data)
        return data,label

class NaiveDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        label = from_numpy(self.data[0])
        data = from_numpy(self.data[1])
        if self.transforms:
            data = self.transforms(data)
        return data,label


if __name__ == "__main__":
    settings = yaml.safe_load(open("settings_45.yaml"))
    t1 = time.time()
    dataset = ChannelDataset(settings['data_filepath']['train'])
    t2 = time.time()
    dataloader = DataLoader(dataset,batch_size=1024,shuffle=False)
    for X,Y in dataloader:
        print("Y.shape: {} \t X.shape: {} .".format(np.shape(Y),np.shape(X)))
    t3 = time.time()
    print("load time of dataset is : {}".format(t2-t1))
    print("load time of dataloader is : {}".format(t3-t2))


