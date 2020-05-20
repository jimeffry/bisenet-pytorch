from easydict import EasyDict

cfgs = EasyDict()
#********
cfgs.crop_height = 480
cfgs.crop_width = 480
cfgs.num_classes = 22
cfgs.netname = 'resnet101'
#**
cfgs.data_dir = '/mnt/data/LXY.data/voc2010'
cfgs.LabelFile = '../datasets/voc2010v2.csv'
cfgs.save_model_path = '/mnt/data/LXY.data/models/imgseg'
cfgs.train_file = '../datasets/voctrain.txt'
cfgs.val_file = '../datasets/vocval.txt'