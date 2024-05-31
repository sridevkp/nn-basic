import cv2 as cv
import numpy as np
from nn import NN

w, h = 28, 28
img = np.zeros(( w*12, h*12, 3 ), np.uint8 )

mnist_nn = NN.load("basic.nn")
labels = list(range(10))

def predict( input ):
    global nn 
    ip = input.reshape( w*h ) / 255.0
    result, index = mnist_nn.predict( ip )
    return labels[index] 

def handle_draw_over():
    gimg = cv.cvtColor( img, cv.COLOR_RGB2GRAY )
    input_img = cv.resize( gimg, ( w, h ), interpolation = cv.INTER_AREA)
    result = predict( input_img )
    print( result )

def draw_brush(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.circle(img, (x,y), 20, (255,255,255), -1)
    if event == cv.EVENT_LBUTTONUP:
        handle_draw_over()

cv.namedWindow('Draw')
cv.setMouseCallback('Draw', draw_brush)

while True:
    cv.imshow('Draw', img)
    
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        img = np.zeros(( w*12, h*12, 3 ), np.uint8 )


cv.destroyAllWindows()