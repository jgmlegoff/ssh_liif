import argparse
import os
import yaml

import xarray as xr

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from sklearn.metrics import mean_squared_error

import models
from utils import make_coord
from test import batched_predict

def resize_fn(img, size):
    return T.Resize(size, T.InterpolationMode.BICUBIC)(img)

def predict(ssh, norm_sub, norm_div, sst = None) : 
    gt = T.ToTensor()(ssh)

    if sst is None :
        input = T.ToTensor()(ssh)
    else : 
        input = T.ToTensor()(sst)


    if args.inputsize is None : 
        h_lr = math.floor(gt.shape[-2] / int(args.scale) + 1e-9)
        w_lr = math.floor(gt.shape[-1] / int(args.scale) + 1e-9)
        gt = gt[:, :round(h_lr * int(args.scale)), :round(w_lr * int(args.scale))] # assume round int
    else : 
        w_lr = int(args.inputsize)
        h_lr = w_lr
        w_hr = round(w_lr * int(args.scale))
        #x0 = random.randint(0, img.shape[-2] - w_hr)
        #y0 = random.randint(0, img.shape[-1] - w_hr)
        #x0, y0 =0,0
        #crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        gt = resize_fn(gt, w_hr)
    input_down = resize_fn(input, (h_lr, w_lr))
    input_down = (input_down-norm_sub)/norm_div
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = gt.shape[-2],gt.shape[-1]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, input_down.cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred*norm_div + norm_sub).view(h, w, 1).permute(2, 0, 1).cpu()
    input_down = (input_down * norm_div) + norm_sub

    bicubic = resize_fn(input_down, (h,w))

    return(pred, bicubic, gt, input_down)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--savefig',default=False)
    parser.add_argument('--rmse',default=False)
    parser.add_argument('--inputsize')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    timestep = 5

    configpath = os.path.dirname(args.model)
    with open(os.path.join(configpath,'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    norm_sub, norm_div = config['data_norm']['inp']['sub'][0], config['data_norm']['inp']['div'][0]
    print('Normalisation coeff : ',config['data_norm']['inp']['sub'][0])

    if args.input == 'natl':
        input_path = '/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz'
        npz_data = np.load(input_path)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
        npz_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]

        pred, bicubic, img_hr, img_down = predict(npz_data[timestep], norm_sub, norm_div)


    elif args.input == 'mercator':
        init_year, last_year = 2015,2019
        datapath = '/data/jean.legoff/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/'
        mercapath = 'GLORYS12V1_PRODUCT_001_030/'
        ds_list = []
        for iyear,year in enumerate(np.arange(init_year,last_year+1)) :
            file_regexpr = f'glorys12v1_mod_product_001_030_{year}-*.nc'
            mult_datafile = os.path.join(os.path.join(datapath,mercapath),file_regexpr)
            tmp_ds = xr.open_mfdataset(mult_datafile)
            ds_list.append(tmp_ds)

        #Loading MDT (Mean Dynamic Topology)
        extrapath = 'GLORYS12V1_PRODUCT_001_030-extra'
        extra_datafile = os.path.join(datapath,extrapath,'glorys12v1_mdt_mod_product_001_030.nc')
        mdt_ds = xr.open_mfdataset(extra_datafile)

        # Concatenation des Datasets en un seul
        ds = xr.concat(ds_list, dim='time')
        n_time, n_lat, n_lon = ds.dims['time'], ds.dims['latitude'], ds.dims['longitude']
        npz_data = ds['sla'][:,:min(n_lat,n_lon),:min(n_lat,n_lon)].to_numpy()+mdt_ds['mdt'][:min(n_lat,n_lon),:min(n_lat,n_lon)].to_numpy()

        day = ds['time'][timestep]

        pred, bicubic, img_hr, img_down = predict(npz_data[timestep], norm_sub, norm_div)

    elif args.input == 'natl-sst':
        input_path = '/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz'
        npz_data = np.load(input_path)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
        ssh_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]
        sst_data = npz_data["FdataAllVar"][1,:,:,0:n_lat]

        print(sst_data[timestep])

        pred, bicubic, img_hr, img_down = predict(ssh_data[timestep], norm_sub, norm_div, sst = sst_data[timestep])

    else : 
        print("Please choose a valid input configuration : 'natl', 'mercator' and 'natl-sst'.")
        exit()

    ########################
    ### Plotting results ###
    ########################

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
        plt.colorbar(pos0, ax=axs[0,0], location = 'left', fraction=0.05)
        pos1 = axs[0,1].imshow((img_down).permute(1, 2, 0))
        plt.colorbar(pos1, ax=axs[0,1])
        pos2 = axs[1,0].imshow(pred.permute(1, 2, 0))
        plt.colorbar(pos2, ax=axs[1,0])
        
        pos3 = axs[1,1].imshow(pred.permute(1, 2, 0) - img_hr.permute(1, 2, 0), vmin = -0.35, vmax = 0.35, cmap=mpl.cm.bwr)
        plt.colorbar(pos3, ax=axs[1,1], fraction=0.05)
        fontsize = 30
        axs[0,0].title.set_text('Ground Truth')
        axs[0,1].title.set_text('Input')
        axs[1,0].title.set_text('Prediction')
        axs[1,1].title.set_text('Error map')
        for i in axs:
            for k in i:
                k.title.set_size(30)
        if args.input =='mercator' : 
            fig.suptitle(f"Super-Resolution scale : {args.scale}, Input size : {args.inputsize}, Dataset : {args.input}, Config : {args.model.split('/')[-2]}, Day : {str(day.values)[:10]}")
        else :
            fig.suptitle(f"Super-Resolution scale : {args.scale}, Input size : {args.inputsize}, Dataset : {args.input}, Config : {args.model.split('/')[-2]}")
        plt.savefig(f"/data/jean.legoff/ssh_liif/results_plot/results{args.model.split('/')[-2]}_{args.input}_i{args.inputsize}_x{args.scale}.png")

    rmsepred = torch.flatten(pred)
    rmsegt = torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1))

    print("Network prediction : ", math.sqrt(mean_squared_error(np.nan_to_num(rmsepred, nan=np.nanmean(rmsepred)),np.nan_to_num(rmsegt, nan=np.nanmean(rmsegt)))))
    print("Bicubic prediction : ", math.sqrt(mean_squared_error(torch.flatten((bicubic).view(bicubic.shape[-2],bicubic.shape[-1],1)), torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1)))))
    #T.ToPILImage()(pred).save(args.output)

    if args.rmse : 
        rmses = []
        bic_rmses = []
        for img in npz_data : 
            pred, bicubic, img_hr, img_down = predict(img, norm_sub, norm_div)
            rmsepred = torch.flatten(pred)
            rmsegt = torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1))
            rmses.append(math.sqrt(mean_squared_error(np.nan_to_num(rmsepred, nan=np.nanmean(rmsepred)),np.nan_to_num(rmsegt, nan=np.nanmean(rmsegt)))))

            bic_rmses.append(math.sqrt(mean_squared_error(torch.flatten(bicubic.view(bicubic.shape[-2],bicubic.shape[-1],1)),np.nan_to_num(rmsegt, nan=np.nanmean(rmsegt)))))

        print('Model MRMSE : ', np.mean(rmses))
        print('Bicubic MRMSE : ', np.mean(bic_rmses))

    
        print(rmses.index(min(rmses)))