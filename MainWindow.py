
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk 
import cv2 
import datetime 
import os 
import scipy.misc 
import numpy as np 
import re 
import linecache 
import argparse
import os
from os import listdir
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm
import math
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim

from data_utils import is_image_file
from model import Net
from espcn import BaseNet
from fsrcnn import FSRCNN
 
filename = 'screenshot0.jpg' 
path = './images/' 
flag = int(0)
count = 0


img_name=''
datapath=''

def CaptureGUI():
    count = 0
    root = Toplevel()
    frame = Frame(root,bg='#3b3a4a')
    frame.pack(expand=YES,fill=BOTH,anchor='n')
    root.title("Face Detection Display")
    root.geometry("1100x800")
    root.iconbitmap('./UISource/logo.ico')

    img1 = Image.open("./UISource/picbg.png")
    photo1 = ImageTk.PhotoImage(img1)
    img_label1 = Label(frame, imag=photo1,bd=0)
    img_label1.grid(row=4,column=2,rowspan=1,columnspan=1,padx=10,pady=10)
    Button(frame,text="Camera Shot",command=lambda:shot_pic(img_label1),height=1,width=8,bg='#1ebad6',font=('Arial',14)).grid(row=6,column=2,padx=10)

    img2 = Image.open("./UISource/picbg.png")
    photo2 = ImageTk.PhotoImage(img2)
    img_label2 = Label(frame, imag=photo2,bd=0)
    img_label2.grid(row=7,column=2,rowspan=1,columnspan=1,padx=10,pady=10)
    Button(frame,text="Face Reconstruction",command=lambda:img_restore(img_label2),height=1,width=8,bg='#1ebad6',font=('Arial',14)).grid(row=9,column=2,padx=10)

    img3 = Image.open("./UISource/picbg.png")
    photo3 = ImageTk.PhotoImage(img3)
    img_label3 = Label(frame, imag=photo3,bd=0)
    img_label3.grid(row=4,column=4,rowspan=1,columnspan=1,padx=10,pady=10)
    Button(frame,text="Face Detection",command=lambda:face_detect_dnn("LR_"+filename, img_label3),height=1,width=8,bg='#1ebad6',font=('Arial',14)).grid(row=6,column=4,padx=20)
    
    img4 = Image.open("./UISource/picbg.png")
    photo4 = ImageTk.PhotoImage(img4)
    img_label4 = Label(frame, imag=photo2,bd=0)
    img_label4.grid(row=7,column=4,rowspan=1,columnspan=1,padx=10,pady=10)
    Button(frame,text="Face Detection",command=lambda:face_detect_dnn("SR_"+filename, img_label4),height=1,width=8,bg='#1ebad6',font=('Arial',14)).grid(row=9,column=4,padx=20)
    
    root.mainloop()
    
def shot_pic(img_label1):
    global flag
    global filename
    global count
    global path
    flag = 0
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

    top = Toplevel()
    top.title('Camera Shot')
    frm_top = Frame(top)
    Button(top, height=2, width=8, text='Save',command=save_pic).pack(side=TOP)
    canvas = Canvas(top, bg='black', height=480, width=640)
    canvas.pack()
    
    while (capture.isOpened()):
        ret, frame = capture.read() 
        frame = cv2.flip(frame, 1)
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=image_file, tags='c1')
            frm_top.update_idletasks()
            frm_top.update()

        if flag==1:
            filename = 'screenshot'+str(count)+'.jpg'
            cv2.imwrite(path+filename, frame)  
            img = cv2.imread(path+filename)
            img = img[90:400,150:400]  
            img = cv2.resize(img,(256,256))
            lr_name = 'LR_'+filename
            cv2.imwrite(path+lr_name, img)  
            count+=1
            break
    capture.release()
    cv2.destroyAllWindows()
    top.destroy()

    pic_name = path+lr_name
    img1 = Image.open(pic_name)  
    photo1 = ImageTk.PhotoImage(img1)  
    img_label1.config(imag=photo1)
    mainloop()
def save_pic():
    global flag
    flag = 1
    return

def img_restore(img_label2):
    global path
    global filename
    lr_name = 'LR_'+filename
    sr_name = 'SR_'+filename
    
    UPSCALE_FACTOR = 3
    MODEL_NAME = 'epoch_3_100.pt'
    model = Net(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    img = Image.open(path + lr_name).convert('YCbCr')
    width, height = img.size
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()
    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img = out_img.resize((width, height),Image.ANTIALIAS)
    out_img.save(path + sr_name)
    
    sr_img = Image.open(path+sr_name)
    photo2 = ImageTk.PhotoImage(sr_img)  
    img_label2.config(imag=photo2)
    mainloop()

'''
Haar Face Detection
'''
def face_detect_haar(img_name, img_label):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(path+img_name) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5,   minSize = (5,5))
    print('Found {0} Faces:)'.format(len(faces)))
    
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 250, 0), 1)

    cv2.imwrite(path+'detect_'+img_name, img)
    img_file = Image.open(path+'detect_'+img_name)
    photo = ImageTk.PhotoImage(img_file)
    img_label.config(imag=photo)
    mainloop()
'''
DNN Face Detection
'''
def face_detect_dnn(img_name, img_label):
    net =cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb","opencv_face_detector.pbtxt")
    image = cv2.imread(path+img_name) 
    height, width, channel = image.shape  
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))  
    net.setInput(blob)  
    detections = net.forward()  
    faces = detections[0, 0]  
    for face in faces:
        confidence = face[2]  
        if confidence > 0.5:  
            box = face[3:7] * np.array([width, height, width, height])  
            pt1 = int(box[0]), int(box[1])  
            pt2 = int(box[2]), int(box[3])  
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), thickness=2)  

            text = '{:.2f}%'.format(confidence * 100)  
            startX, startY = pt1
            y = startY - 10 if startY - 10 > 10 else startY + 10
            org = (startX, y)  
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)  
    cv2.imwrite(path+'detect_'+img_name, image)
    img_file = Image.open(path+'detect_'+img_name)
    photo = ImageTk.PhotoImage(img_file)
    img_label.config(imag=photo)
    mainloop()


def FileGUI():
    root = Toplevel()
    frame = Frame(root,bg='#3b3a4a')
    frame.pack(expand=YES,fill=BOTH,anchor='n')
    root.title("Image Reconstruction")
    root.geometry("1100x800")
    root.iconbitmap('./UISource/logo.ico')

    Button(frame,text="Select From File",command=lambda:read_pic(root, img_label1, t1),height=1,width=10,bg='#1ebad6',font=('Arial',14)).grid(row=2,column=2,padx=10,pady=20)
    img1 = Image.open("./UISource/picbg.png")
    photo1 = ImageTk.PhotoImage(img1)
    img_label1 = Label(frame, imag=photo1,bd=0)
    img_label1.grid(row=4,column=2,rowspan=1,columnspan=1,padx=10,pady=10)
    Button(frame,text="ESPCN Reconstruct",command=lambda:img_restore_base(img_label2, t2),height=1,width=10,bg='#1ebad6',font=('Arial',14)).grid(row=2,column=3,padx=10,pady=20)
    Button(frame,text="FSRCNN Reconstruct",command=lambda:img_restore_fsrcnn(img_label2, t2),height=1,width=10,bg='#1ebad6',font=('Arial',14)).grid(row=2,column=4,padx=10,pady=20)
    Button(frame,text="FSRSR Reconstruct",command=lambda:img_restore_GT(img_label2, t2),height=1,width=10,bg='#1ebad6',font=('Arial',14)).grid(row=2,column=6,padx=10,pady=20)
    
    img2 = Image.open("./UISource/picbg.png")
    photo2 = ImageTk.PhotoImage(img2)
    img_label2 = Label(frame, imag=photo2,bd=0)
    img_label2.grid(row=4,column=4,rowspan=1,columnspan=1,padx=10,pady=10)

    nfrm_L1 = Frame(frame)
    Label(nfrm_L1, text="Detection Result:",bg='#c0c0c8').pack(fill=BOTH)
    t1 = Text(nfrm_L1, width=50, height=8, undo=True, autoseparators=False)
    t1.pack()
    nfrm_L1.grid(row=10, column=2,padx=10)
    
    mfrm_L2 = Frame(frame)
    Label(mfrm_L2,text="Detection Result:",bg='#c0c0c8').pack(fill=BOTH)
    t2 = Text(mfrm_L2, width=50, height=8, undo=True, autoseparators=False)
    t2.pack()
    mfrm_L2.grid(row=10,column=4,padx=10)

    root.mainloop()
    return
    
def read_pic(root_f, img_label1, t1):
    global datapath
    global img_name
    fullpath = tkinter.filedialog.askopenfilename(parent=root_f, initialdir="F:/deep-learning/ESPCN01/images/",title='Select one picture')
    datapath, img_name = os.path.split(fullpath)
    datapath += '/'
    img = Image.open(fullpath)
    
    target_path = datapath+'target_'+img_name
    tr_img = Image.open(target_path)
    width, height = tr_img.size
    img = img.resize((width, height),Image.ANTIALIAS)
    img.save(datapath+'lr_'+img_name)
    
    lr_path = datapath+'/lr_'+img_name
    lr_img = ImageTk.PhotoImage(Image.open(lr_path))
    img_label1.config(imag=lr_img)
       
    psnr = cal_psnr(img, tr_img)
    string = '\nPSNR: '+str(psnr)

    ssim = cal_ssim(lr_path, target_path)
    string += '; SSIM: '+str(ssim)
    t1.insert(INSERT,string)
    mainloop()
    return None

def cal_psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1/1. - img2/1.) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255**2 / mse)

def cal_ssim(path1, path2):
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(img1, img2)
    return ssim_value

 
def img_restore_GT(img_label2, t2):
    global datapath
    global img_name
    data_name = img_name
    sr_name = 'sr_'+img_name
    
    UPSCALE_FACTOR = 3
    MODEL_NAME = 'epoch_3_100.pt'
    model = Net(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    img = Image.open(datapath + data_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()
    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(datapath + sr_name)
    
    sr_img = Image.open(datapath+sr_name)
    photo2 = ImageTk.PhotoImage(sr_img)  
    img_label2.config(imag=photo2)
    
    target_path = datapath+'target_'+img_name
    tr_img = Image.open(target_path)
    psnr = cal_psnr(sr_img, tr_img)
    string = '\nPSNR: '+str(psnr)

    ssim = cal_ssim(datapath+sr_name, target_path)
    string += '; SSIM: '+str(ssim)
    t2.insert(INSERT,string)
    
    mainloop()
    return

def img_restore_base(img_label2, t2):
    global datapath
    global img_name
    data_name = img_name
    sr_name = 'sr_'+img_name
    
    UPSCALE_FACTOR = 3
    MODEL_NAME = 'epoch_3_100.pt'
    model = BaseNet(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    img = Image.open(datapath + data_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()
    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(datapath + sr_name)
    
    sr_img = Image.open(datapath+sr_name)
    photo2 = ImageTk.PhotoImage(sr_img)  
    img_label2.config(imag=photo2)
    
    
    target_path = datapath+'target_'+img_name
    tr_img = Image.open(target_path)
    psnr = cal_psnr(sr_img, tr_img)
    string = '\nPSNR: '+str(psnr)
    
    ssim = cal_ssim(datapath+sr_name, target_path)
    string += '; SSIM: '+str(ssim)
    t2.insert(INSERT,string)
    
    mainloop()
    return
def img_restore_fsrcnn(img_label2, t2):
    global datapath
    global img_name
    data_name = img_name
    sr_name = 'sr_'+img_name
    
    UPSCALE_FACTOR = 3
    MODEL_NAME = 'epoch_3_100.pt'
    model = FSRCNN(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    img = Image.open(datapath + data_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()
    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(datapath + sr_name)

    sr_img = Image.open(datapath+sr_name)
    photo2 = ImageTk.PhotoImage(sr_img)  
    img_label2.config(imag=photo2)
    
    target_path = datapath+'target_'+img_name
    tr_img = Image.open(target_path)
    psnr = cal_psnr(sr_img, tr_img)
    string = '\nPSNR: '+str(psnr)
    
    ssim = cal_ssim(datapath+sr_name, target_path)
    string += '; SSIM: '+str(ssim)
    t2.insert(INSERT,string)
   
    mainloop()
    return

if __name__ == '__main__':
    root = Tk()
    frame = Frame(root,bg='#3b3a4a')
    frame.pack(expand=YES,fill=BOTH,anchor='n')
    root.title("Image Super Resolution System")
    root.geometry("1100x800")
    root.iconbitmap('./UISource/logo.ico')
    
    Button(frame,text="Face Detection Display",command=CaptureGUI,height=5,width=30,bg='#1ebad6',font=('Arial',14)).place(x=350,y=200)
    Button(frame,text="Image Reconstruction",command=FileGUI,height=5,width=30,bg='#1ebad6',font=('Arial',14)).place(x=350,y=400)
    
    root.mainloop()

