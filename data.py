import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset,Sampler
from torchvision import transforms
import glob

class PokemonData(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        df = pd.read_csv('./pokemon_data/pokemon.csv',\
                         usecols=['type1','type2','is_mega','is_gmax','is_legend',\
                                  'is_mythical','is_ultra_beast','is_paradox','color','shape'])
        
        type1 = pd.get_dummies(df['type1'])
        type2 = pd.get_dummies(df['type2'])
        types = type1.add(type2)
        sum = types.sum(axis=1)**0.5
        types = types.div(sum,axis='index').fillna(0)
        special = df[['is_mega','is_gmax','is_legend','is_mythical','is_ultra_beast']].astype("int32")
        paradox = df['is_paradox'].fillna(0)
        paradox = pd.get_dummies(paradox)
        colors = pd.get_dummies(df['color'])
        shapes = pd.get_dummies(df['shape'])
        label = pd.concat((types,\
                            special,\
                            paradox,colors,shapes),axis=1).to_numpy(dtype='float64')
        self.label = torch.tensor(label)
        
        images = glob.glob("./pokemon_data/images/images/*.png")
        bg = Image.new('RGBA',(512,512),(0,0,0))
        self.image = []
        for i in range(len(images)):
            with open(images[i], 'rb') as file:
                img = Image.open(file).convert('RGBA')
                img = Image.alpha_composite(bg,img).convert('RGB')
                self.image.append(transforms.ToTensor()(img))
                # print(np.array(img))
                # img.show()
                

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]
        if self.transform:
            image = self.transform(image)
        return image,label
    


def random_label(num):
    res = np.zeros((num,54))
    for i in range(num):
        #types
        type_num = np.random.choice((1,2))
        type_index = np.random.randint(0,20,(type_num,))
        types = np.zeros((1,20))
        for t in type_index:
            types[0,t] = 1
        
        is_giant = np.random.rand()>=0.5
        giant = np.zeros((1,2))
        if is_giant:
            giant_index = np.random.choice((0,1))
            giant[0,giant_index] = 1

        is_rare = np.random.rand()>=0.5
        rare = np.zeros((1,3))
        if is_rare:
            rare_index = np.random.choice((0,1,2))
            rare[0,rare_index] = 1

        paradox = np.zeros((1,3))
        if not is_giant:
            is_paradox = np.random.rand()>=0.5
            if is_paradox:
                paradox_index = np.random.choice((0,1,2))
                paradox[0,paradox_index] = 1

        color = np.zeros((1,11))
        color_index = np.random.randint(0,11)
        color[0,color_index] = 1

        shape = np.zeros((1,15))
        shape_index = np.random.randint(0,15)
        shape[0,shape_index] = 1

        res[i] = np.concatenate((types,giant,rare,paradox,color,shape),axis=1).astype('float64')
    return res


class PokemonData2(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        df = pd.read_csv('./pokemon_data/pokemon.csv',\
                         usecols=['type1','type2','shape'])
        
        type1 = pd.get_dummies(df['type1'])
        type2 = pd.get_dummies(df['type2'])
        types = type1.add(type2)
        sum = types.sum(axis=1)**0.5
        types = types.div(sum,axis='index').fillna(0)
        shapes = pd.get_dummies(df['shape'])
        label = pd.concat((types,shapes),axis=1).to_numpy(dtype='float64')
        self.label = torch.tensor(label)
        
        images = glob.glob("./pokemon_data/images/images/*.png")
        bg = Image.new('RGBA',(512,512),(0,0,0))
        self.image = []
        for i in range(len(images)):
            with open(images[i], 'rb') as file:
                img = Image.open(file).convert('RGBA')
                img = Image.alpha_composite(bg,img).convert('RGB')
                self.image.append(transforms.ToTensor()(img))
                # print(np.array(img))
                # img.show()
                

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]
        if self.transform:
            image = self.transform(image)
        return image,label
    


def random_label2(num):
    res = np.zeros((num,35))
    for i in range(num):
        #types
        type_num = np.random.choice((1,2))
        type_index = np.random.randint(0,20,(type_num,))
        types = np.zeros((1,20))
        for t in type_index:
            types[0,t] = 1

        shape = np.zeros((1,15))
        shape_index = np.random.randint(0,15)
        shape[0,shape_index] = 1

        res[i] = np.concatenate((types,shape),axis=1).astype('float64')
    return res
