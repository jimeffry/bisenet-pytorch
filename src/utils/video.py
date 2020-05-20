import os
import sys
import numpy as np 
import cv2

def saveframe(vpath,vname,imgdir):
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    cap = cv2.VideoCapture(vpath)
    frame_width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cnt = 0
    while cnt < total_num and cap.isOpened():
        _,frame = cap.read()
        sys.stdout.write(">\r %d / %d" %(cnt,total_num))
        sys.stdout.flush()
        if cnt %10 ==0 :
            savename = vname +'_'+str(cnt)+'.jpg'
            savepath = os.path.join(imgdir,savename)
            cv2.imwrite(savepath,frame)
        cnt +=1
    cap.release()

if __name__ == '__main__':
    # vpath = '/data/videos/movies/20190324060005_2558.avi'
    # vname = 'v4'
    # sdir = '/data/videos/mframes/video4'
    # saveframe(vpath,vname,sdir)
    vdir = '/data/videos/movies/test'
    file_cnts = os.listdir(vdir)
    vcnt = 5
    sdir = '/data/videos/mframes'
    for tmp in file_cnts:
        vpath = os.path.join(vdir,tmp)
        vname = 'v'+str(vcnt)
        vsave = 'video'+str(vcnt)
        vsavedir = os.path.join(sdir,vsave)
        saveframe(vpath,vname,vsavedir)
        vcnt+=1