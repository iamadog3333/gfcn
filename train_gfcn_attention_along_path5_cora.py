"""
This is to train a GFCN model who's attention is along flows or paths.
How to run:
python train_gfcn_attention_along_path5_cora.py --dataset=cora

"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
from dgl.data import register_data_args, load_data
import my_data_loader_dgl_part2
import models_attention_along_path4 #change model number here 1

# python train_gfcn_attention_along_path5_cora.py --dataset=cora

# 1 layer but big gfcn model, firt decomposed paths , multi-pool, LeakyReLU
model_num = 'path_att_1'  #change model number here 2
num_labels = 7 # find the num at https://arxiv.org/pdf/1710.10903.pdf
parser = argparse.ArgumentParser(description='GAT')
register_data_args(parser)
parser.add_argument("--gpu", type=int, default=-1,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--no_cuda", type=bool, default=True,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=200,
                    help="number of training epochs")
parser.add_argument("--num-heads", type=int, default=1,
                    help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num_hidden", type=int, default=num_labels,
                    help="number of hidden units")
parser.add_argument("--seed", type=int, default=88,
                    help="number of seed")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                    help="input feature dropout")
parser.add_argument("--dropout", type=float, default=.6,
                    help="attention dropout")
parser.add_argument("--learning_rate", type=float, default=0.005,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default= 5e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--fastmode', action="store_true", default=False,
                    help="skip re-evaluate the validation set")
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')


args = parser.parse_args()
print(args)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 200
learning_rate = 0.005

# Data loader
dataloader = my_data_loader_dgl_part2.my_data_set(args) # data loader use multi process, which will cause error at windows 


# Convolutional neural network (two convolutional layers)
# The convolution layer keep the conveluted sequence length the same as input.
# input tensor size (batch_size, channels, 1)
#feature_len = 1433
#num_classes = 7


model = models_attention_along_path4.GAT(
                nfeat=num_labels, 
                nhid=args.num_hidden, 
                nclass=num_labels, 
                dropout=args.dropout, 
                nheads=args.num_heads, 
                alpha=args.alpha)  #change model number here 3
model = model.to(device)

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
#criterion = F.nll_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train the model
train_steps = dataloader.get_data_len('train')
val_steps = dataloader.get_data_len('val')
test_steps = dataloader.get_data_len('test')
print('train_steps', train_steps)
print('val_steps', val_steps)
print('test_steps', val_steps)

max_val_acc = 0
max_val_epoch = 0
set_patient = 10
patient = 0

for epoch in range(num_epochs):
    #print('train at epoch ', epoch)
    epoch_loss = 0
    for i in range(train_steps): #train_steps
        # if i%10000 == 0:
        #     print('train at step', i)
        datas, labels, gat_vec, node_idx = dataloader.get_a_path_data('train', i)
        datas = datas.to(device)
        #datas = torch.squeeze(datas)
        labels = labels.to(device)
        #print('labels shape ', labels.shape)
        labels = labels.reshape(-1)
        #print('labels reshaped ', labels.shape)
        labels = labels.long()
        gat_vec = gat_vec.to(device)
        
        # Forward pass
        outputs = model(datas)
        

        # outputs = torch.masked_select(outputs, masks.type(torch.ByteTensor))
        # labels = torch.masked_select(labels, masks.type(torch.ByteTensor))
        #print(outputs.shape)
        #print(labels.shape)
        loss = F.nll_loss(outputs, labels)

        epoch_loss = epoch_loss + loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if 1: #i%80000==0 and i>0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
           .format(epoch+1, num_epochs, i+1, train_steps, loss.item()), end='')
        print(' epoch_loss', epoch_loss, end='')
        epoch_loss = 0

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i in range(val_steps): # val_steps
                datas, labels, gat_vec, node_idx = dataloader.get_a_path_data('val', i)
                images = datas.to(device)
                labels = labels.to(device)
                labels = labels.reshape(-1)
                labels = labels.long()
                gat_vec = gat_vec.to(device)
                outputs = model(images, is_training=False)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('  Accuracy on val data: {} %'.format(100 * correct / total))
            val_acc = correct / total
            if epoch > 50 :
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    max_val_epoch = epoch
                    patient = 0
                    # Save the model checkpoint
                    torch.save(model.state_dict(), 'saved_'+str(model_num)+'_'+str(epoch)+'model.ckpt')
                else:
                    patient += 1
                    if patient > set_patient:
                        break

        model.train()


# Final test accuray
model.load_state_dict(torch.load('saved_'+str(model_num)+'_'+str(max_val_epoch)+'model.ckpt'))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(test_steps): # val_steps
        datas, labels, gat_vec, node_idx = dataloader.get_a_path_data('test', i)
        images = datas.to(device)
        labels = labels.to(device)
        labels = labels.reshape(-1)
        labels = labels.long()
        gat_vec = gat_vec.to(device)
        outputs = model(images, is_training=False)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy on test data: {} %'.format(100 * correct / total))


        

    