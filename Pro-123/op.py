import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
from sklearn.metrics import accuracy_score
import cv2
import csv


X = np.load('image.npz')
X = X['arr_0']
y = pd.read_csv("labels.csv")
y = y["labels"]

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n_classes = len(classes)

X,y = fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
xt,xts,yt,yts=train_test_split(X,y,random_state=9,train_size=3500,test_size=500)

xt = xt/255.0
xts = xts/255.0
clf = LogisticRegression(solver="saga",multi_class="multinomial")
clf.fit(xt,yt)

yp = clf.predict(xts)
print(yp)
print(yts)

print(accuracy_score(yts,yp))

cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    gi = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w = gi.shape
    ul = (int(w/2-56),int(h/2-56))
    br = (int(w/2+56),int(h/2+56))
    cv2.rectangle(gi,ul,br(0,0,0),2)
    roi = gi[ul[1]:br[1],ul[0]:br[0]]
    impil = Image.fromarray(roi)
    imbw = impil.convert("L")
    imbwresize = imbw.resize((28,28),Image.ANTIALIAS)
    imgin= PIL.ImageOps.invert(imbwresize)
    mp = np.percentile(imgin,20)
    ims = np.clip(imgin-mp,0,255)
    maxp=np.max(imgin)
    imss = np.asanyarray(ims)/maxp
    ti = np.array(imss).reshape(1.784)
    tp=clf.predict(ti)
    print(tp)
    cv2.imshow("frame",frame)

cap.release()    