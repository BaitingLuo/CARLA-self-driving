import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
from model.lanenet.loss import DiscriminativeLoss, FocalLoss
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
# Set matplotlib to interactive mode
plt.ion()

# Create a figure and axes for displaying the images
fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
def compute_loss(net_output, binary_label, instance_label, device, loss_type = 'FocalLoss'):
    k_binary = 10    #1.7
    k_instance = 0.3
    k_dist = 1.0

    if(loss_type == 'FocalLoss'):
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif(loss_type == 'CrossEntropyLoss'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        # print("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()
    
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    #pix_embedding = net_output["instance_seg_logits"]
    #ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    #print(instance_label.shape, pix_embedding.shape)
    #var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    #var_loss = var_loss * k_instance
    #dist_loss = dist_loss * k_dist
    #instance_loss = var_loss + dist_loss
    #total_loss = binary_loss + instance_loss
    total_loss = binary_loss
    out = net_output["binary_seg_pred"]
    #print(binary_label)
    out_squeezed = torch.squeeze(out, 1)
    jaccard = BinaryJaccardIndex().to(device)
    miou = jaccard(out_squeezed, binary_label)
    return total_loss, binary_loss, binary_loss, out, miou
    #return total_loss, binary_loss, binary_loss, out


def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type = 'FocalLoss', num_epochs=25, save_path=None):
    since = time.time()
    training_log = {'epoch':[], 'training_loss':[], 'val_loss':[], 'training_IoU':[], 'val_IoU':[]}
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0
            running_miou = 0.0

            # Iterate over data.
            loop = tqdm(dataloaders[phase])
            loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))

            for _, (inputs, binarys, instances) in enumerate(loop):
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    '''
                    predicted_mask = outputs['binary_seg_pred'][0].cpu().numpy()
                    predicted_mask = np.squeeze(predicted_mask)

                    # Now the shape of predicted_mask should be (256, 512), which is valid for Matplotlib
                    ax2.imshow(predicted_mask, cmap='gray')
                    ax2.set_title('Predicted Mask')
                    ax2.axis('off')


                    # Update the figure
                    plt.draw()
                    plt.pause(0.0001)  # Pause to update the figure
                    '''
                    loss = compute_loss(outputs, binarys, instances, device, loss_type)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss[0].backward()
                        optimizer.step()

                loop.set_postfix(loss=loss[0].item(),
                            binary_loss=loss[1].item(),
                            instance_loss=loss[2].item(),
                            miou=loss[4].item())

                # statistics
                running_loss += loss[0].item() * inputs.size(0)
                running_loss_b += loss[1].item() * inputs.size(0)
                running_loss_i += loss[2].item() * inputs.size(0)
                running_miou += loss[4].item() * inputs.size(0)

            if phase == 'train':
                if scheduler != None:
                    scheduler.step()



            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            epoch_miou = running_miou / dataset_sizes[phase]
            print('=> Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f} mIoU: {:.4f}'.format(epoch_loss, binary_loss, instance_loss, epoch_miou))

            # deep copy the model
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
                training_log['training_IoU'].append(epoch_miou)
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                training_log['val_IoU'].append(epoch_miou)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_save_filename = '{}/loss={:.4f}_epoch={}.pth'.format(save_path, epoch_loss, epoch + 1)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': epoch_loss,
                    }, model_save_filename)
                    print("Best loss model is saved: {}".format(model_save_filename))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['training_IoU'] = np.array(training_log['training_IoU'])
    training_log['val_loss'] = np.array(training_log['val_loss'])
    training_log['val_IoU'] = np.array(training_log['val_IoU'])
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable