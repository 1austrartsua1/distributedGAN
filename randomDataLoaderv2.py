
import torch
from torch.utils.data import Dataset, DataLoader

class RandomDataSet(Dataset):
    def __init__(self,dim1,dim2,numDataPoints,numLabels,numColorChannels=3,seedToUse = None):
        self.dim1 = dim1
        self.dim2 = dim2
        self.numDataPoints = numDataPoints
        self.numLabels = numLabels
        self.numColorChannels = numColorChannels
        self.seed = seedToUse
        if self.seed:
            torch.seed(self.seed)

    def __len__(self):
        return self.numDataPoints

    def __getitem__(self,idx):
        label = torch.randint(0,self.numLabels,(1,))
        return(torch.randn(self.numColorChannels,self.dim1,self.dim2),label[0])

class RandomFlatDataSet(Dataset):
    def __init__(self,dim,numDataPoints,numLabels):
        self.dim = dim
        self.numDataPoints = numDataPoints
        self.numLabels = numLabels



    def __len__(self):
        return self.numDataPoints

    def __getitem__(self,idx):
        label = torch.randint(0,self.numLabels,(1,))
        return torch.randn(self.dim),label[0]


if __name__ == "__main__":
    dim1 = 10
    dim2 = 20
    n = 100
    nlabels = 10
    aRandomDataset = RandomDataSet(dim1,dim2,n,nlabels)
    myLoader = DataLoader(aRandomDataset,batch_size=10,shuffle=True)
    for batchindex,(data,target) in enumerate(myLoader):
        print(batchindex)
        print(data.shape)
        print(target.shape)
    print('done')
