import numpy as np
from PIL import Image
import math
from math import sin, cos, pi

Img_mat = np.zeros((200, 200, 3), dtype=np.uint8)


def draw_line( image, x0, y0, x1, y1, count, color ):
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color
        
def dotted_line(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1)**2 + (y0 -y1)**2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1,color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
        
        
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

    
        derror += dy
        if (derror > (x1-x0)):
            derror -= 2 * (x1 - x0)
            y += y_update
        
    #y = y0
    #dy = 2.0 * (x1 - x0) * abs(y1 - y0)/(x1 - x0)
    #derror = 0.0
    #y_update = 1 if y1 > y0 else -1
    
    
    #for x in range (x0, x1):
        #t = (x-x0)/(x1 - x0)
        #y = round ((1.0 - t)*y0 + t*y1)
        #image[y, x] = color
        #derror += dy
        #if (derror > 2.0 * (x1 - x0) * 0.5):
            #derror -= 2.0 * (x1 - x0) * 1.0
            #y += y_update
        #image[y, x] = color

        #if (xchange):
            #image[x, y] = color
        #else:
            #image[y, x] = color

for k in range(13):
    x0, y0 = 100, 100
    x1 = int(100 + 95 * cos(2 * pi /13 * k))
    y1 = int(100 + 95 * sin(2 * pi /13 * k))
    #draw_line(Img_mat, x0, y0, x1, y1, 100, [0, 0, 255])
    #dotted_line(Img_mat, x0, y0, x1, y1, [0, 0, 255])
    x_loop_line(Img_mat, x0, y0, x1, y1, [0, 0, 255])

img = Image.fromarray(Img_mat, mode='RGB')
img.save('img.png')
img.show()
