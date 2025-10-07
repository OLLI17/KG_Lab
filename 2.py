import numpy as np
from PIL import Image, ImageOps
import math
from math import cos, sin, pi

file_in = open('model_1.obj')
img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)

def Brezenhem(image, x0, y0, x1, y1,color):
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

def file():
    v = []
    f = []
    fs = []
    for s in file_in:
        sp = s.split()
        if (sp[0] == 'v'):
            v.append([sp[1], sp[2], sp[3]])
            img_mat[int(5000 * np.double(sp[2]) + 500), int(5000 * np.double(sp[1]) + 500)] = [100, 255, 200]
        elif (sp[0] == 'f'):
            fs = []
            for S in sp:
                fs.append(S.split('/'))
            f.append([fs[1][0], fs[2][0],fs[3][0]])
    for i in range(len(f)):
        x0 = int(5000 * np.double(v[int(f[i][0]) - 1][0]) + 500)
        y0 = int(5000 * np.double(v[int(f[i][0]) - 1][1]) + 500)
        x1 = int(5000 * np.double(v[int(f[i][1]) - 1][0]) + 500)
        y1 = int(5000 * np.double(v[int(f[i][1]) - 1][1]) + 500)
        x2 = int(5000 * np.double(v[int(f[i][2]) - 1][0]) + 500)
        y2 = int(5000 * np.double(v[int(f[i][2]) - 1][1]) + 500)
        Brezenhem(img_mat, x0, y0, x1, y1, [100, 255, 200])
        Brezenhem(img_mat, x2, y2, x1, y1, [100, 255, 200])
        Brezenhem(img_mat, x0, y0, x2, y2, [100, 255, 200])
            
        
for k in range(13):
    x0, y0 = 100, 100
    x1 = int(100 + 95 * cos(2 * pi / 13 * k))
    y1 = int(100 + 95 * sin(2 * pi / 13 * k))
file()

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('krolik.png')
