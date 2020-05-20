#!/usr/bin/bash
# python train/train.py --num_epochs 1000 --learning_rate 1e-3  --cuda 0,2 --batch_size 20  --optimizer sgd --validation_step 200 --show_step 20 --use_gpu True --mulgpu True #--pretrained_model_path /mnt/data/LXY.data/models/imgseg/bs_ohem_best.pth
#***
# python utils/processdataset.py
# python utils/mat2png.py
# python preparedata/contexvoc.py
# python utils/load_label.py
#** test
# python test/demo1.py --modelpath /data/models/img_seg/bs_voc_best3.pth --file_in /data/detect/test_seg_ims --labelpath ../datasets/voc2010v2.csv  --save_dir ../datasets/voctest 
# python test/demo1.py --modelpath /data/models/img_seg/bs_voc_best3.pth --file_in /data/detect/test_seg_ims/d6.jpg --labelpath ../datasets/voc2010v2.csv  #../datasets/cityscape.csv #/data/detect/cityscape/leftImg8bit/test/berlin /data/detect/r4.jpeg 
# python test/demo1.py --modelpath /data/models/img_seg/bs_voc_b22_1.pth --file_in /data/detect/VOC/VOCdevkit/VOC2010/JPEGImages/2008_002459.jpg --labelpath ../datasets/voc2010v2.csv 
# python test/demo1.py --modelpath /data/models/img_seg/bs_voc_b22_1.pth --file_in /data/detect/test_seg_ims/h1.jpg  --labelpath ../datasets/voc2010v2.csv  --save_dir ../datasets/voctest2
# python test/demo1.py --modelpath /data/models/img_seg/bs_voc_b22_2.pth --file_in /data/videos/mframes/video2/v2_20.jpg  --labelpath ../datasets/voc2010v2.csv
# python test/demo1.py --modelpath /data/models/img_seg/bs_voc_b22_1.pth --file_in /data/videos/mframes/video3/v3_200.jpg  --labelpath ../datasets/voc2010v2.csv

# pb test
python test/demo_tf.py --modelpath ../models/bisenet_v.pb --file_in /data/videos/mframes/video2/v2_20.jpg  --labelpath ../datasets/voc2010v2.csv

#*****eval
# python test/eval.py  --cuda 3 --batch_size 1  --use_gpu True --mulgpu False --file_out voc2010_val_result.txt --pretrained_model_path /mnt/data/LXY.data/models/imgseg/bs_ohem_best.pth

#** convert model
# python utils/tr2tf.py