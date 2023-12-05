import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data import *
import pickle
from PIL import Image

torch.set_default_dtype(torch.float64)

### variables
label_size =54
batch_size = 32
image_size = 512
noise_size = 0
feature_size = 64
num_epochs = 500
G_lr = 0.0004
D_lr = 0.0002
beta1 = 0.4
num_gpu = 4
retrain = True
pretrained = './models/dc_rgan_bcel15'

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
    
    def forward(self,label):
        # x = torch.cat((noise,label),1)
        x = label.view(label.size(0),noise_size+label_size,1,1)
        out = self.deconvolutions(x)
        return out.view(out.size(0),3,image_size,image_size)

class Discriminator(nn.Module):
    def __init__(self,num_gpu):
        super(Discriminator,self).__init__()
        self.num_gpu = num_gpu
        self.convolution = nn.Sequential(
            nn.Conv2d(3,feature_size,4,2,1,bias=False),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #256*256
            nn.Conv2d(feature_size,feature_size*2,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #128*128
            nn.Conv2d(feature_size*2,feature_size*4,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*4),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #64*64
            nn.Conv2d(feature_size*4,feature_size*8,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*8),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #32*32
            nn.Conv2d(feature_size*8,feature_size*16,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*16),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #16*16
            nn.Conv2d(feature_size*16,feature_size*16,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*16),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #8*8
            nn.Conv2d(feature_size*16,feature_size*32,4,2,1,bias=False),
            nn.BatchNorm2d(feature_size*32),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.4),
            #4*4
            nn.Conv2d(feature_size*32,label_size+1,4,1,0,bias=False),
        )

    def forward(self,input):
        return self.convolution(input)
    
if __name__=='__main__':

    ### data augmentation
    transform = transforms.Compose([\
                            transforms.RandomHorizontalFlip(0.5),\
                            transforms.RandomRotation(10),\
                            transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.85,1),antialias=True),\
                            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])\
                            ])

    ### data loader
    dataset = PokemonData(transform)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    device = torch.device("cuda:4" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

    ### Model Creation
    def weight_initialization(model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(model.weight.data)
        elif classname.find('Batchnorm') != -1:
            nn.init.kaiming_normal_(model.weight.data)
            nn.init.constant_(model.bias.data,0)

    torch.cuda.empty_cache()
    G = Generator(num_gpu).to(device)
    if (device.type=='cuda') and (num_gpu>1):
        G = nn.DataParallel(G,(4,5,6,7),dim=0)
    if retrain:
        G.apply(weight_initialization)
    else:
        state_dict = torch.load(pretrained+'_generator.pth')
        G.load_state_dict(state_dict)
        G.eval()
    print(G)

    D = Discriminator(num_gpu).to(device)
    if (device.type=='cuda') and (num_gpu>1):
        D = nn.DataParallel(D,(4,5,6,7),dim=0)
    if retrain:
        D.apply(weight_initialization)
    else:
        state_dict = torch.load(pretrained+'_discriminator.pth')
        D.load_state_dict(state_dict)
        D.eval()
    print(D)


    ### Optimizer
    w = torch.tensor([5,\
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
                      1,1,1,1,1,1,1,1,\
                      1,1,1,1,1,1,1,1,1,1,1,\
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).to(device)[None,:,None,None]
    criterion = nn.BCEWithLogitsLoss(w)
    D_optimizer = torch.optim.Adam(D.parameters(),lr=D_lr,betas=(beta1,0.992))
    G_optimizer = torch.optim.Adam(G.parameters(),lr=G_lr,betas=(beta1,0.992))
    D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(D_optimizer,num_epochs*(len(data_loader)))
    G_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_optimizer,num_epochs*(len(data_loader)))


    ### Visualizer
    fixed_labels = Variable(transforms.ToTensor()(random_label(batch_size))[0,:,:,None]).to(device=device)
    # fixed_noise = Variable(torch.randn(batch_size,noise_size)).to(device=device)[:,:,None]


    ### training step
    def G_train(batch_size,D,G,G_optimizer,criterion):
        G_optimizer.zero_grad()
        # noise = Variable(torch.randn(batch_size, noise_size)).to(device=device)[:,:,None]
        fake_labels = Variable(transforms.ToTensor()(random_label(batch_size))[0,:,:,None]).to(device=device)
        fake_images = G(fake_labels)
        scores = D(fake_images)
        target = torch.cat((torch.ones(batch_size,1).to(device=device),fake_labels[:,:,0]),1)[:,:,None,None]
        G_loss = criterion(scores,Variable(target)).to(device=device)
        G_loss.backward()
        G_optimizer.step()
        G_scheduler.step()
        return G_loss.item()

    def D_train(batch_size,D,G,D_optimizer,criterion,real_images,real_labels):
        D_optimizer.zero_grad()
        real_scores = D(real_images)
        real_target = torch.cat((torch.ones(batch_size,1).to(device=device),real_labels),1)[:,:,None,None]
        real_loss = criterion(real_scores,Variable(real_target)).to(device=device)
        # noise = Variable(torch.randn(batch_size, noise_size)).to(device=device)[:,:,None]
        fake_labels = Variable(transforms.ToTensor()(random_label(batch_size))[0,:,:,None]).to(device=device)
        fake_images = G(fake_labels)
        fake_scores = D(fake_images)
        fake_target = torch.cat((torch.zeros(batch_size,1).to(device=device),fake_labels[:,:,0]),1)[:,:,None,None]
        fake_loss = criterion(fake_scores,Variable(fake_target)).to(device=device)
        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()
        D_scheduler.step()
        real_accuracy = nn.Sigmoid()(real_scores)[:,0].mean()
        D_fake_accuracy = nn.Sigmoid()(fake_scores)[:,0].mean()
        return D_loss.item(),real_accuracy.item(),D_fake_accuracy.item()


    ### Log
    log = {'G_losses':[],'D_losses':[]}


    ### train
    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch))
        for i, (images,labels) in enumerate(data_loader):
            print('Trainin: {}/{}'.format(i,len(data_loader)-1))
            real_images = Variable(images).to(device=device)
            labels = Variable(labels).to(device=device)
            G.train()
            D_loss, D_real_accuracy, D_fake_accuracy = D_train(batch_size=len(real_images),D=D,G=G,D_optimizer=D_optimizer,criterion=criterion,\
                            real_images=real_images,real_labels=labels)
            G_loss = G_train(batch_size=batch_size,D=D,G=G,G_optimizer=G_optimizer,criterion=criterion)
            G.eval()
            log['D_losses'].append(D_loss)
            log['G_losses'].append(G_loss)
            if (((epoch%10==0) or (epoch==num_epochs-1)) and (i==len(data_loader)-1)):
                with torch.no_grad():
                    fake = G(fixed_labels).detach().cpu()
                curr_img = make_grid(fake,padding=2,normalize=True).numpy()
                plt.imsave(pretrained+'_epoch_'+str(epoch)+'.png',np.transpose(curr_img,(1,2,0)))
                torch.save(D.module.state_dict(),pretrained+'_discriminator.pth')
                torch.save(G.module.state_dict(),pretrained+'_generator.pth')
                with open(pretrained+'_log','wb') as handle:
                    pickle.dump(log,handle,protocol=pickle.HIGHEST_PROTOCOL)
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(Real): %.4f\tD(Fake)): %.4f'
                    % (epoch, num_epochs, i, len(data_loader)-1,
                        D_loss, G_loss, D_real_accuracy, D_fake_accuracy))


    ### statistics
    with open(pretrained+'_log','rb') as handle:
        log = pickle.load(handle)
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(log['G_losses'],label="G")
    plt.plot(log['D_losses'],label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(pretrained+'_loss.png')