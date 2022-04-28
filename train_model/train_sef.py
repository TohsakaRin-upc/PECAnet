import os, time

import sys

sys.path.append("/hy-tmp/PECAnet")

import torch
from config import *
import torchvision
from torchvision import transforms, datasets
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.utils.tensorboard as tb
from method.LocalMax_GlobalMin import *
from train_model.para import count_params
# from torch.cuda.amp import autocast as autocast
progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)
import learning.sef_learning
from networks import PECAnet

device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
device_name = device.type + ':' + str(device.index) if device.type == 'cuda' else 'cpu'

########################################################################################################## initialize params
datasetname = "stcars"
image_size = 448
batchsize = 16
nthreads = 8
lr = 1e-2
lmgm = 1
entropy = 1
soft = 0.05
epochs = 50
optmeth = 'sgd'
regmeth = 'cms'
lr_milestones = [60, 100]
attention_flag = True
maxent_flag = False
# number of attentions for different datasets
if datasetname in ['cubbirds']:
    nparts = 4
elif datasetname in ['vggaircraft']:
    nparts = 3
elif datasetname in ['stdogs', 'stcars']:
    nparts = 2
else:
    nparts = 1  # number of parts you want to use for your dataset

# 'resnet50attention' for sef, 'resnet50maxent' for resnet with MaxEnt, 'resnet50vanilla' for the vanilla resnet
networkname = 'resnet50attention'

############################################################################################################## displaying logs
timeflag = time.strftime("%d-%b-%Y-%H:%M")
# writer = tb.SummaryWriter(log_dir='./runs/'+datasetname+'/'+networkname+time.strftime("%d-%b-%Y"))
log_items = r'{}-net{}-att{}-lmgm{}-entropy{}-soft{}-lr{}-imgsz{}-bsz{}'.format(
    datasetname, int(networkname[6:8]), nparts, lmgm, entropy, soft, lr, image_size, batchsize)
writer = tb.SummaryWriter(comment='-' + log_items)
logfile = open(TRAINED_MODEL + '/sef/results/' + log_items + '.txt', 'w')
modelname = log_items + '.model'

############################################################################################################## model zoo and dataset path
datapath = DATASET_DIR
# modelzoopath = '/path/to/the/vanilla/resnet/models'
sys.path.append(modelzoopath)
datasetpath = os.path.join(datapath, datasetname)
modelpath = os.path.join(progpath, 'models')
resultpath = os.path.join(progpath, 'results')

###########################################################################################################################  organizing data
if datasetname in ['cubbirds', 'vggaircraft']:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.RandomCrop((448, 448)),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.CenterCrop((448, 448)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
else:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# organizing datasets
datasplits = {x: datasets.ImageFolder(os.path.join(datasetpath, x), data_transform[x])
              for x in ['trainval', 'test']}

# preparing dataloaders for datasets
dataloader = {x: torch.utils.data.DataLoader(datasplits[x], batch_size=batchsize, shuffle=True, num_workers=nthreads)
              for x in ['trainval', 'test']}

datasplit_sizes = {x: len(datasplits[x]) for x in ['trainval', 'test']}

class_names = datasplits['trainval'].classes
num_classes = len(class_names)

####################################################################################################### constructing or loading model

###load model

model = PECAnet.ecanet50(pretrained=False, model_dir=modelzoopath, num_classes=num_classes, nparts=nparts, attention=attention_flag, device=device)

state_dict_path = os.path.join(modelzoopath, "resnet50-19c8e357.pth")

state_params = torch.load(state_dict_path)

# pop redundant params from laoded states
state_params.pop('fc.weight')
state_params.pop('fc.bias')

# modify output layer
in_channels = model.fc.in_features
new_fc = nn.Linear(in_channels, num_classes, bias=True)
model.fc = new_fc

# initializing model using pretrained params except the modified layers
model.load_state_dict(state_params, strict=False)

# tensorboard writer
images, _ = next(iter(dataloader['test']))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)
writer.add_graph(model, images)

# to gpu if available
model.cuda(device)
count_params(model)
########################################################################################################### creating loss functions
# cross entropy loss
cls_loss = nn.CrossEntropyLoss()

# semantic group loss
lmgm_loss = LocalMaxGlobalMin(rho=lmgm, nchannels=512 * 4, nparts=nparts, device=device)

criterion = [cls_loss, lmgm_loss]

#################################################################################################################### creating optimizer
# optimizer
optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)

# optimization scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)


# scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=0,last_epoch=-1,verbose=False)
######################################################################################################################### train model
isckpt = False  # set True to load learned models from checkpoint, defatult False

indfile = "{}: opt={}, lr={}, lmgm={}, nparts={}, entropy={}, soft={}, epochs={}, imgsz={}, batch_sz={}".format(
    datasetname, optmeth, lr, lmgm, nparts, entropy, soft, epochs, image_size, batchsize)
print("\n{}\n".format(indfile))
print("\n{}\n".format(indfile), file=logfile)

model, train_rsltparams = learning.sef_learning.train(
    model, dataloader, criterion, optimizer, scheduler,
    datasetname=datasetname, isckpt=isckpt, epochs=epochs,
    networkname=networkname, writer=writer, device=device, maxent_flag=maxent_flag,
    soft_weights=soft, entropy_weights=entropy, logfile=logfile)

train_rsltparams['imgsz'] = image_size
train_rsltparams['epochs'] = epochs
train_rsltparams['init_lr'] = lr
train_rsltparams['batch_sz'] = batchsize

print('\nBest epoch: {}'.format(train_rsltparams['best_epoch']))
print('\nBest epoch: {}'.format(train_rsltparams['best_epoch']), file=logfile)
print("\n{}\n".format(indfile))
print("\n{}\n".format(indfile), file=logfile)
print('\nWorking on cluster: {}\n'.format(device_name))

logfile.close()

#################################################################################################################### save model
torch.save({'model_params': model.state_dict(), 'train_params': train_rsltparams}, os.path.join(modelpath, modelname))
torch.save({'model_params': TRAINED_MODEL, 'train_params': train_rsltparams}, os.path.join(modelpath, modelname))


