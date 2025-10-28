import numpy as np
from PIL import Image, ImageOps
import math
from math import cos, sin, pi
#from random import randint


def rotate_matrix(alf,bet,gam):
    Rx = np.array([
        [1, 0, 0],
        [0, cos(alf), sin(alf)],
        [0, -sin(alf), cos(alf)]
        ])
    Ry = np.array([
        [cos(bet), 0, sin(bet)],
        [0, 1, 0],
        [-sin(bet), 0, cos(bet)]
        ])
    Rz = np.array([
        [cos(gam), sin(gam), 0],
        [-sin(gam), cos(gam), 0],
        [0, 0, 1]
        ])
    return Rx @ Ry @ Rz


def normal_to_polygon(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1-x2, y1-y2, z1-z2])
    v2 = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(v1, v2)


def cos_our(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    
    n = normal_to_polygon(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    l = np.array([0, 0, 1])
    
    return np.dot(n, l)/np.linalg.norm(n)


def draw_polygons(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, zbuf, color):
#    if cos_our(x0, y0, z0, x1, y1, z1, x2, y2, z2) >= 0: return

    
    xmin = min(x0, x1, x2)
    xmax = max(x0, x1, x2)
    ymin = min(y0, y1, y2)
    ymax = max(y0, y1, y2)
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0

   
#    color = ( 0, -255 * cos_our(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0) #(randint(0,255), randint(0,255), randint(0, 255))#[100, 255, 200]
    
    for y in range(int(ymin), int(ymax) + 1):
        for x in range(int(xmin), int(xmax) + 1):

            

            lambda0, lambda1, lambda2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0.0 and lambda1 >= 0.0 and lambda2 >= 0.0:
                #img[y][x] = color
                 z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                 if z_buffer[x][y] >= z:
                     z_buffer[x][y] = z
                     img[y][x] = color


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = (((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda1 = (((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))

    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2



def projective_transform(X, Y, Z, a_X, a_Y, u_0, v_0):
  
    if Z <= 0:
        return None, None  # Точка находится за камерой
    
    u = (a_X * X) / Z + u_0
    v = (a_Y * Y) / Z + v_0
    return u, v


img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
file_in = open('model_1.obj')
v = []
f = []


z_buffer = np.zeros([2000, 2000])
for i in range(2000):
    for j in range(2000):
        z_buffer[i][j] = 1000000000000000.0

for s in file_in:
    sp = s.split()
    if sp[0] == 'v':
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])
    elif sp[0] == 'f':
        f.append([sp[1].split('/')[0], sp[2].split('/')[0], sp[3].split('/')[0]])


alf = 0
bet = 1
gam = 0


tx = 0.0
ty = -0.03
tz = 1

R = rotate_matrix(alf,bet,gam)

v_fix = []
for top in v:
    top_array = np.array(top)
    v_transform = R @ top_array + np.array([tx, ty, tz])
    v_fix.append(v_transform)

a_X = 8000 
a_Y = 8000
u_0 = 0 
v_0 = 0


for i in range(0,len(f)):
  
    x0 = v_fix[int(f[i][0]) - 1][0]
    y0 = v_fix[int(f[i][0]) - 1][1]
    z0 = v_fix[int(f[i][0]) - 1][2]
    
    x1 = v_fix[int(f[i][1]) - 1][0]
    y1 = v_fix[int(f[i][1]) - 1][1]
    z1 = v_fix[int(f[i][1]) - 1][2]
    
    x2 = v_fix[int(f[i][2]) - 1][0]
    y2 = v_fix[int(f[i][2]) - 1][1]
    z2 = v_fix[int(f[i][2]) - 1][2]

    scale = 8000
    WIDTH, HEIGHT = 2000, 2000
    
    x0_proj = float(scale*x0/z0  + WIDTH/2)
    y0_proj = float(scale*y0/z0  + HEIGHT/2)
    x1_proj = float(scale*x1/z1  + WIDTH/2)
    y1_proj = float(scale*y1/z1  + HEIGHT/2)
    x2_proj = float(scale*x2/z2  + WIDTH/2)
    y2_proj = float(scale * y2/z2 +HEIGHT/2)

    vec1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    vec2 = np.array([x1 - x0, y1 - y0, z1 - z0])
    
    n = np.cross(vec1, vec2)
    l = np.array([0, 0, 1])

    cos_val = (np.dot(n, l)) / (np.linalg.norm(n) * np.linalg.norm(l))
    #print (cos_val)
    
    if cos_val >= 0:
        continue

    if z0 <= 0 or z1 <= 0 or z2 <= 0:
        continue

    if not all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in [(x0_proj, y0_proj), (x1_proj, y1_proj), (x2_proj, y2_proj)]):
        continue

    color = (0, -255 * cos_val, 0)
    
    draw_polygons(img_mat, x0_proj, y0_proj, z0, x1_proj, y1_proj, z1, x2_proj, y2_proj, z2, z_buffer, color)


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('krolik2.0.png')
img.show()
