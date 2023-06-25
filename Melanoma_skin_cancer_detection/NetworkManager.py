import os
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.optim.lr_scheduler  as lr_scheduler
import Model
import time

path = 'model_weights'
if not os.path.exists(path):
    os.makedir(path)

def train(netname, config, use_cuda, dataset, dataset_size, num_classes ):
    net = Model.createNet(netname, config.channels, num_classes)
    since = time.time()
    val_acc_hist = []
    best_acc = 0.0
    criterion = nn.NLLLoss()
    device = None
    if use_cuda:
        device = torch.device('cuda' + str(config.cuda_idx))
    else:
        device = torch.device('cpu')
    criterion = criterion.to(device)
    opti = optimizer.Adam(net.parameters(), lr=config.base_lr,
                          weight_decay=config.wt_decay,betas=(config.beta1, 0.999))
    print(f'Dataset Sizes: {dataset_size}')
    for epoch in range(config.epochs):
        print(f'current Epoch: {epoch}')
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            running_loss = 0.0
            running_correct = 0.0
            for i, (inputs, labels) in enumerate(dataset[phase]):
                opti.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                net = net.to(device)
                outputs = net(inputs)

                with torch.set_grad_enabled(phase== 'train'):
                    if phase == 'train':
                        loss = criterion(outputs, labels)
                        loss.backward()
                        opti.step()
                    else:
                        loss = criterion(outputs, labels)
                    _, pred = torch.max(outputs.data,1)
                running_loss += loss.item()
                running_correct += torch.sum(pred == labels.data.long().item())
            epoch_loss += running_loss /dataset_size[phase]
            epoch_acc += running_correct/dataset_size[phase]
            if phase == 'train':
                train_epoch_loss = epoch_loss
                train_epoch_acc = epoch_acc
            else:
                val_epoch_loss = epoch_loss
                val_epoch_acc = epoch_acc
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and (epoch_acc > best_acc):
                best_acc = epoch_acc
                modelname = os.path.join(path, f'{netname}.pth')
                torch.save(net.state_dict(), modelname)
                print(f'Saving model: {modelname}')
                val_acc_hist.append(epoch_acc)
        print('ok')
        time_elapsed = time.time() - since
        print('Training comleted in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Val Accuracy: {:.4f}'.format(best_acc))