%autosave 60

from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ultralytics/yolov5 
%cd yolov5
%pip install -qr requirements.txt  

import torch
import utils
display = utils.notebook_init()

!python train.py --img 640 --batch 2 --epochs 20 --data E:/integration/yolo_detection/overall_yolotxt/data.yaml --weights yolov5s.pt --cache


# display.Image(filename='runs/detect/exp/img1.jpg', width=600)
#!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source /content/drive/MyDrive/IIITB/sakura.mp4

!python home/tcestudent/yolo_detection/yolov5/detect.py --weights home/tcestudent/yolo-detection/yolov5/runs/train/exp11/weights/epoch75.pt --img 640 --conf 0.25 --source video_path --save-crop


!zip -r /content/yolov5/runs/detect/exp10 /content/Folder_To_Zip
from google.colab import files
files.download("/content/yolov5/runs/detect/exp10")

!zip -r /content/yolov5/runs/detect.zip /content/Folder_To_Zip

!zip -r /content/down.zip /content/yolov5/runs/detect/exp11

#!python detect.py --weights runs/train/exp/weights/best.pt --source vid.mp4 --conf 0.25 --source /content/drive/MyDrive/IIITB/sakura.mp4
!python detect.py --weights runs/train/exp3/weights/best.pt --source img.jpg --conf 0.25 --source /content/drive/MyDrive/rough/test/images/railway-speed-limit-sign-uk-FB00DP.rf.175d826540c32fb7eea9053e474fc805.jpg

import cv2
import dlib
from google.colab.patches import cv2_imshow

detector = dlib.get_frontal_face_detector()
new_path ='/content/unkown/'
def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

def save(img,name, bbox, width=180,height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))#we need this line to reshape the images
    cv2.imwrite(name+".jpg", imgCrop)

def faces():
    frame =cv2.imread('/content/download.jfif')
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fit =20
    # detect the face
    for counter,face in enumerate(faces):
        print(counter)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        # save(gray,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        save(gray,new_path+str(counter),(x1,y1,x2,y2))
    frame = cv2.resize(frame,(800,800))
    cv2_imshow(frame)
    cv2.waitKey(0)
    print("done saving")
faces()