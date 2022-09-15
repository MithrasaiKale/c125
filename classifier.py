import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x_train, x_test, y_train, y_test=train_test_split(X,y,random_state=9, train_size=7500, test_size=2500)
x_trainScale=x_train/255.0
x_testScale=x_test/255.0
clf=LogisticRegression(solver="saga", multi_class="multinomial").fit(x_trainScale,y_train)

def getPrediction(image):
    im_pil = Image.open(image)
    ibw = im_pil.convert('L')
    ibwResize = ibw.resize( (28,28) , Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(ibwResize, pixel_filter)
    iscaled = np.clip( ibwResize - min_pixel , 0, 255)
    max_pixel = np.max( ibwResize )
    finalScaleImg = np.asarray( iscaled ) / max_pixel
    test_sample = np.array( finalScaleImg ).reshape(1,784)
    test_pred = clf.predict( test_sample )
    return test_pred[0]
