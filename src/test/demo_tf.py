#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import csv


def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    parser.add_argument('--labelpath', type=str,
                        default='', help='trained model')
    return parser.parse_args()


class ImgSeg(object):
    def __init__(self,args):
        self.loadtfmodel(args.modelpath)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        self.real_num = 0
        self.loadlabel(args.labelpath)
        self.save_dir = args.save_dir
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


    def loadtfmodel(self,mpath):
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(modefile.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') 
        # tf.train.write_graph(graph_def, './', 'breathtest.pbtxt', as_text=True)
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     # if 'input' in m.name or 'output' in m.name or 'confidence' in m.name:
        #     print(m.values()) #m.name,
        # print("********************end***************")
        # self.input_image = self.sess.graph.get_tensor_by_name('img_input:0') #img_input
        # self.cls_out = self.sess.graph.get_tensor_by_name('cls_out:0') #softmax_output
        # self.conf_out = self.sess.graph.get_tensor_by_name('conf_out:0')
        self.input_image = "img_input:0"
        self.conf_out = "cls_out:0"

        
    def propress(self,img):
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        # img = cv2.resize(img,(1920,1080))
        h,w = img.shape[:2]
        gth = int(np.ceil(h/8.0)*8)
        gtw = int(np.ceil(w/8.0)*8)
        gth,gtw = (560,560)
        img = cv2.resize(img,(gtw,gth))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img /= 255.0
        img -= rgb_mean
        img /= rgb_std
        return img


    def inference_img(self,imgorg):
        t1 = time.time()
        img = self.propress(imgorg.copy())
        img = np.expand_dims(img,0)
        conf_out = self.sess.run(self.conf_out,feed_dict={self.input_image:img})
        # print("***out shape:",np.shape(output))
        conf_out = conf_out[0]
        # print(conf_out[0].shape)
        # cls_out = np.argmax(conf_out,axis=-1)
        print(cls_out.shape)
        masks = self.decodeColor(cls_out)
        accs = self.calarea(cls_out)
        t2 = time.time()
        print('consuming:',t2-t1)
        return masks,accs
    
    def loadlabel(self,labelpath):
        f_in = open(labelpath,'r')
        ann = csv.DictReader(f_in)
        self.label_info = {}
        for row in ann: #ann.iterrows():
            label_name = row['name']
            r = row['r']
            g = row['g']
            b = row['b']
            class_19 = row['class_num']
            self.label_info[label_name] = [int(r), int(g), int(b), class_19]
        f_in.close()

    def decodeColor(self,predicts):
        label_values = []
        label_values.append([0,0,0])
        for key in self.label_info:
            label_values.append(self.label_info[key][:3])
        colour_codes = np.array(label_values)
        imgsout = []
        for i in range(predicts.shape[0]):
            x = colour_codes[predicts[i].astype(int)]
            x = cv2.cvtColor(np.uint8(x), cv2.COLOR_RGB2BGR)
            imgsout.append(x)
        return imgsout

    def calarea(self,predicts):
        accout = []
        id_list = [10,12,13,14,16,19]
        acc = 0
        for i in range(predicts.shape[0]):
            img = predicts[i]
            h,w = img.shape[:2]
            for tid in id_list:
                mask_tmp = np.equal(img,tid)
                acc += np.sum(mask_tmp.astype(np.float))
            acc = acc/(h*w)
            accout.append(acc)
        return accout


    def headcnts(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                frame,cnt_head = self.inference_img(img)
                print('heads >> ',cnt_head)
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                # cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir):
            #     os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                tmp_file_s = tmp_file.split('\t')
                if len(tmp_file_s)>0:
                    tmp_file = tmp_file_s[0]
                    self.real_num = int(tmp_file_s[1])
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpg'
                # tmp_path = os.path.join(self.img_dir,tmp_file) 
                tmp_path = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                frame,cnt_head = self.inference_img(img)
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                #cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            frame_width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # print(frame_width, frame_height)
            imgw = int(frame_width)
            imgh = int(frame_height)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #cv2.VideoWriter_fourcc(*"mp4v")
            # out = cv2.VideoWriter('test.mp4', fourcc, 25,(frame_width, frame_height))
            out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('I','4','2','0'), 25, (imgw, imgh))
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                frame_cnt = 0
                boxes = []
                while cap.isOpened():
                    _,frame = cap.read()
                    frame_cnt +=1
                    if frame_cnt % 10 ==0:
                        frame,boxes = self.inference_img(frame)
                    out.write(frame)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    # cv2.imwrite('test_video1.jpg',frame)
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            # img = cv2.resize(img,(1920,1080))
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,boxes = self.inference_img(img)
                # hotmaps = self.get_hotmaps(odm_maps)
                # self.display_hotmap(hotmaps)
                # keybindings for display
                cv2.imshow('result',frame[0])
                cv2.imwrite('newm1.jpg',frame[0])
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = ImgSeg(args)
    imgpath = args.file_in
    detector.headcnts(imgpath)