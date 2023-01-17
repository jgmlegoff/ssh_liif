import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from sklearn.metrics import mean_squared_error

import models
from utils import make_coord
from test import batched_predict

def resize_fn(img, size):
    return transforms.Resize(size, Image.BICUBIC)(img)

def predict(img) : 
    img_hr = transforms.ToTensor()(img)

    h_lr = 16
    w_lr = 16
    img = img_hr[:, :round(h_lr * int(args.scale)), :round(w_lr * int(args.scale))] # assume round int
    img_down = resize_fn(img, (h_lr, w_lr))
    crop_lr, crop_hr = img_down, img
    img_down = (img_down-0.5)/0.5
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = img_hr.shape[-2],img_hr.shape[-1]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, img_down.cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred*0.5 + 0.5).view(h, w, 1).permute(2, 0, 1).cpu()
    img_down = (img_down*0.5)+0.5

    bicubic = resize_fn(img_down, (img_hr.shape[-2],img_hr.shape[-1]))

    return(pred, bicubic, img_hr, img_down)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--savefig',default=False)
    parser.add_argument('--rmse',default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    npz_data = np.load(args.input)
    n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
    npz_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]

    '''
    img_hr = transforms.ToTensor()(npz_data[20])

    h_lr = math.floor(img_hr.shape[-2] / int(args.scale) + 1e-9)
    w_lr = math.floor(img_hr.shape[-1] / int(args.scale) + 1e-9)
    img = img_hr[:, :round(h_lr * int(args.scale)), :round(w_lr * int(args.scale))] # assume round int
    img_down = resize_fn(img, (h_lr, w_lr))
    crop_lr, crop_hr = img_down, img
    img_down = (img_down-0.5)/0.5
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = img_hr.shape[-2],img_hr.shape[-1]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, img_down.cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred*0.5 + 0.5).view(h, w, 1).permute(2, 0, 1).cpu()
    img_down = (img_down*0.5)+0.5

    bicubic = resize_fn(img_down, (img_hr.shape[-2],img_hr.shape[-1]))
    '''
    pred, bicubic, img_hr, img_down = predict(npz_data[20])
    ##Plotting results
    #plt.imshow(img_hr.permute(2, 0, 1))
    print(img_hr.shape, pred.shape)
    fig, axs = plt.subplots(2, 4,figsize=(18, 9))
    pos00 = axs[0,0].imshow(img_hr.permute(1, 2, 0),vmin = -0.5,vmax = 0.8)
    plt.colorbar(pos00, ax=axs[0,0])
    pos01 = axs[0,2].imshow(pred.permute(1, 2, 0),vmin = -0.5,vmax = 0.8)
    plt.colorbar(pos01, ax=axs[0,2])
    pos02 = axs[0,1].imshow((img_down).permute(1, 2, 0))
    plt.colorbar(pos02, ax=axs[0,1])
    pos03 = axs[0,3].imshow((bicubic).permute(1, 2, 0))
    plt.colorbar(pos03, ax=axs[0,3])
    axs[0,0].title.set_text('Ground Truth')
    axs[0,1].title.set_text('Input')
    axs[0,2].title.set_text('Prediction')
    axs[0,3].title.set_text('Bicubic')

    pos10 = axs[1,0].imshow(img_hr.permute(1, 2, 0) - img_hr.permute(1, 2, 0), cmap=mpl.cm.bwr)
    plt.colorbar(pos10, ax=axs[1,0])
    pos11 = axs[1,2].imshow(pred.permute(1, 2, 0) - img_hr.permute(1, 2, 0), cmap=mpl.cm.bwr,vmin = -0.5,vmax = 0.5)
    plt.colorbar(pos11, ax=axs[1,2])
    pos12 = axs[1,1].imshow((img_down).permute(1, 2, 0) - (img_down).permute(1, 2, 0), cmap=mpl.cm.bwr)
    plt.colorbar(pos12, ax=axs[1,1])
    pos13 = axs[1,3].imshow((bicubic).permute(1, 2, 0) - img_hr.permute(1, 2, 0), cmap=mpl.cm.bwr)
    plt.colorbar(pos13, ax=axs[1,3])

    plt.show()
    
    if args.savefig :
        fig, axs = plt.subplots(2, 2,figsize=(18, 18))
        pos0 = axs[0,0].imshow(img_hr.permute(1, 2, 0),vmin = -0.6,vmax = 0.8)
        axs[0,0].set_ylim(axs[0,0].get_ylim()[::-1])
        plt.colorbar(pos0, ax=axs[:,0], location = 'left', fraction=0.05)
        pos1 = axs[0,1].imshow((img_down).permute(1, 2, 0),vmin = -0.6,vmax = 0.8)
        #plt.colorbar(pos1, ax=axs[0,1])
        pos2 = axs[1,0].imshow(pred.permute(1, 2, 0),vmin = -0.6,vmax = 0.8)
        #plt.colorbar(pos2, ax=axs[1,0])
        
        pos3 = axs[1,1].imshow(pred.permute(1, 2, 0) - img_hr.permute(1, 2, 0), vmin = -0.35, vmax = 0.35, cmap=mpl.cm.bwr)
        plt.colorbar(pos3, ax=axs[:,1], fraction=0.05)
        axs[0,0].title.set_text('Ground Truth')
        axs[0,1].title.set_text('Input')
        axs[1,0].title.set_text('Prediction')
        axs[1,1].title.set_text('Error map')
        plt.savefig(f"results_resaclike_x{args.scale}.png")

    ##Plotting zoomed results
    fig, axs = plt.subplots(1, 3)
    pos = axs[0].imshow(img_hr.permute(1, 2, 0)[600:800,600:800,:])
    pos = axs[1].imshow(pred.permute(1, 2, 0)[600:800,600:800,:])
    pos = axs[2].imshow((bicubic).permute(1, 2, 0)[600:800,600:800,:])
    axs[0].title.set_text('Ground Truth')
    axs[1].title.set_text('Prediction')
    axs[2].title.set_text('Bicubic')
    plt.colorbar(pos, ax=axs.ravel().tolist())
    plt.show()

    print("Network prediction : ", math.sqrt(mean_squared_error(torch.flatten(pred), torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1)))))

    print("Bicubic prediction : ", math.sqrt(mean_squared_error(torch.flatten((bicubic).view(bicubic.shape[-2],bicubic.shape[-1],1)), torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1)))))
    #transforms.ToPILImage()(pred).save(args.output)

    if args.rmse : 
        rmses = []
        for img in npz_data : 
            pred, bicubic, img_hr, img_down = predict(img)
            rmses.append(math.sqrt(mean_squared_error(torch.flatten(pred), torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1)))))

        print(rmses)
        print('MRMSE : ', np.mean(rmses))