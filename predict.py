import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt
import cv2

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def generate_img_from_points(txtpath, name):
    coords = []
    with open(os.path.join(txtpath,name), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            coords.append([float(line[0]),float(line[1])])
    coords = np.array(coords)
    y,x = coords.T
    y = abs(y-max(y))
    coords[:,0] = y
    print(np.max(x), np.max(y))
    plt.figure(figsize=(15, 15*(np.max(y)/np.max(x))))
    plt.xlim(0, np.max(x))
    plt.ylim(0, np.max(y))
    plt.axis('off')
    plt.scatter(x,y, s=5/(2**2))#, c=atoms_class_alline)
    plt.savefig(os.path.join(txtpath,'test_ml_4cls.png'), \
         bbox_inches='tight', pad_inches=0, dpi=int(1*60))
    #plt.show()
    plt.close()

    img = cv2.imread(os.path.join(txtpath,'test_ml_4cls.png'), cv2.IMREAD_GRAYSCALE)
    img = abs(255-img)
    cv2.imwrite(os.path.join(txtpath,'test_ml_4cls.png'), img)
    ishape = img.shape
    print(ishape)
    # cv2.imshow('black_white',img)
    # cv2.waitKey(2000)
    x_1 = np.round(x*ishape[1]/np.max(x)); y_1 = ishape[0]-np.round(y*ishape[0]/np.max(y))
    x_1 = np.where(x_1>=ishape[1], ishape[1]-1, x_1)
    y_1 = np.where(y_1>=ishape[0], ishape[0]-1, y_1)
    coords = np.concatenate((y_1.reshape(-1,1), x_1.reshape(-1,1)), axis=1)
    return torch.Tensor(img/np.max(img)), torch.Tensor(coords)

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # in_files = args.input
    # out_files = get_output_filenames(args)

    #generate_img_from_points and add path
    # txtpath = '/hardisk/image_process/real_images/grain+boundry+disloaction'
    # name = 'grain boundary dislocation_pos_circularMask.txt'
    # txtpath = r'/hardisk/image_process/model_implication/811 LPSCl data needs phase segmentation/0067/crop'
    # name = r'5MX BF  0067.s_pos_guassianMask (2).txt'
    txtpath = r'/hardisk/image_process/RUnet/Pytorch-UNet/data/5samples/0070 crop/gaussion no sample iter2/0070 crop'
    name = r'0070 crop_pos_guassianMask.txt'
    images, coors = generate_img_from_points(txtpath, name)
    images = images.unsqueeze(dim=0).unsqueeze(dim=0)
    images = images.to(device=device, dtype=torch.float32)
    coors = coors.to(device=device, dtype=torch.float32).long()
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    net.eval()
    with torch.cuda.amp.autocast(enabled=args.amp):
        masks_pred = net(images)    
        print(masks_pred.shape)                 
        h_idx = coors[:,0].clone().detach()
        w_idx = coors[:,1].clone().detach()
        labels_pred= masks_pred[0].clone().permute(1,2,0)[h_idx, w_idx]
        length_points = len(labels_pred[0])
        labels_pred = torch.nn.functional.softmax(labels_pred, dim=1)
        # print(labels_pred[110:120])
        labels_pred = torch.argmax(labels_pred, dim=1)
    coors_label = torch.cat((coors, labels_pred.unsqueeze(dim=1)), dim=1)
    np.savetxt(os.path.join(txtpath, 'predict_4classes_4w.txt') ,np.array(coors_label.cpu()))
    y = abs(coors[:,0].cpu()-max(coors[:,0].cpu())); x = coors[:,1].cpu()
    pre = coors_label[:,2].cpu()
    color_label = []
    for num in pre:
        if num == 0:
            color_label.append('#0593A2')
        elif num == 1:
            color_label.append('#E3371E')
        elif num == 2:
            color_label.append('#151F30')
        elif num == 3:
            color_label.append('#BF9000')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.scatter(x, y, s=1, c=color_label)#, cmap='Dark2')
    plt.savefig(os.path.join(txtpath, 'predict_4classes_4w'),dpi=400)
    plt.show()
