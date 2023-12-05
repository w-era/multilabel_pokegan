from locale import normalize
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset,Sampler,DataLoader
from torchvision import transforms
import glob
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import make_grid
import pickle



torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class PokemonData(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        df = pd.read_csv('./pokemon_data/pokemon.csv',\
                         usecols=['name','type1','type2','is_mega','is_gmax','is_legend',\
                                  'is_mythical','is_ultra_beast','is_paradox','color','shape'])
        self.df = df
        
        self.name = df['name'].to_numpy()
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
        name = self.name[index]
        return image,label,name


torch.set_default_dtype(torch.float64)

### variables
label_size =54
batch_size = 32
image_size = 512
noise_size = 16
feature_size = 64
num_epochs = 500
G_lr = 0.0003
D_lr = 0.0001
beta1 = 0.4
num_gpu = 4
retrain = True
pretrained = './models/dc_rgan_bcel19'

class Generator(nn.Module):
    def __init__(self,num_gpu):
        super(Generator,self).__init__()
        self.num_gpu = num_gpu
        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(noise_size+label_size,feature_size*32,4,1,0,bias=False),
            nn.BatchNorm2d(feature_size*32),
            nn.ReLU(True),
            #4*4
            nn.ConvTranspose2d(feature_size*32,feature_size*16,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*16),
            nn.ReLU(True),
            #8*8
            nn.ConvTranspose2d(feature_size*16,feature_size*16,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*16),
            nn.ReLU(True),
            #16*16
            nn.ConvTranspose2d(feature_size*16,feature_size*8,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*8),
            nn.ReLU(True),
            #32*32
            nn.ConvTranspose2d(feature_size*8,feature_size*4,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*4),
            nn.ReLU(True),
            #64*64
            nn.ConvTranspose2d(feature_size*4,feature_size*2,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*2),
            nn.ReLU(True),
            #128*128
            nn.ConvTranspose2d(feature_size*2,feature_size,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            #256*256
            nn.ConvTranspose2d(feature_size,3,4,2,1,bias=False),
            nn.Tanh()
            #512*512
        )
    
    def forward(self,noise,label):
        x = torch.cat((noise,label),1)
        x = x.view(x.size(0),noise_size+label_size,1,1)
        out = self.deconvolutions(x)
        return out.view(out.size(0),3,image_size,image_size)

 


transform = transforms.Compose([\
                        transforms.RandomHorizontalFlip(0.5),\
                        transforms.RandomRotation(10),\
                        transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.85,1),antialias=True),\
                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])\
                        ])

dataset = PokemonData(transform)
data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
device = torch.device("cuda:5" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

def fixed_noise(batch_size,noise_size):
    return torch.randn(batch_size, noise_size).to(device=device)

torch.cuda.empty_cache()
G = Generator(num_gpu).to(device)
state_dict = torch.load(pretrained+'_generator.pth',map_location=device)
G.load_state_dict(state_dict)
G.eval()

target_image,target_label,target_name =next(iter(data_loader))

with torch.no_grad():
    fake = G(fixed_noise(batch_size,noise_size),target_label.to(device=device)).detach().cpu()
curr_img = make_grid(fake,padding=2,normalize=True).numpy()
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(make_grid(target_image.to(device),padding=5,normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(curr_img,(1,2,0)))
plt.savefig(pretrained+'_comparison.png')

# with open(pretrained+'_log','rb') as handle:
#     log = pickle.load(handle)
# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(log['G_losses'],label="G")
# plt.plot(log['D_losses'],label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig(pretrained+'_loss.png')