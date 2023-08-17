# HaUNet: Hard attention enhanced U-Net


## Quick start

### install requirements

```bash
pip install -r requirements.txt
```

### generate simulated images
generate images of 3 phases: rock salt, O1, and O3
```console
> python simulate_defect.py
```

generate 3 phases and dislocation
```console
> python simulate_defect_edge_dislocation.py
```


### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the HaUNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.


### Prediction
c is the classes of prediction  
Set to 3 or 4 

```console
python predict.py -m 'path_to_pretrained_model' -s 1 -c 3  
```

## Pretrained model
[pretrained models](https://figshare.com/articles/software/HaUnet_models/23896533) are available for the phase segmentaion.


### prediction example
The '0069 crop' folder

