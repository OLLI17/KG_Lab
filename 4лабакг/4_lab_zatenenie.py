import numpy as np
from PIL import Image, ImageOps
import math
from math import cos, sin, pi

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


def compute_vertex_normals(v, f):
    vertex_normals = [np.array([0.0, 0.0, 0.0]) for _ in range(len(v))]
    vertex_face_count = [0 for _ in range(len(v))]
    
    for face in f:
        v0_idx = int(face[0]) - 1
        v1_idx = int(face[1]) - 1
        v2_idx = int(face[2]) - 1
        
        v0 = v[v0_idx]
        v1 = v[v1_idx]
        v2 = v[v2_idx]
        
        normal = normal_to_polygon(v0[0], v0[1], v0[2], 
                                  v1[0], v1[1], v1[2], 
                                  v2[0], v2[1], v2[2])
        
        vertex_normals[v0_idx] += normal
        vertex_normals[v1_idx] += normal
        vertex_normals[v2_idx] += normal
        
        vertex_face_count[v0_idx] += 1
        vertex_face_count[v1_idx] += 1
        vertex_face_count[v2_idx] += 1
    
    for i in range(len(vertex_normals)):
        if vertex_face_count[i] > 0:
            vertex_normals[i] = vertex_normals[i] / vertex_face_count[i]
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 0:
                vertex_normals[i] = vertex_normals[i] / norm
    
    return vertex_normals


def compute_vertex_illumination(vertex_normal, light_direction):
    n_norm = np.linalg.norm(vertex_normal)
    l_norm = np.linalg.norm(light_direction)
    
    if n_norm == 0 or l_norm == 0:
        return 0
    
    dot_product = np.dot(vertex_normal, light_direction)
    illumination = dot_product / (n_norm * l_norm)
    
    return illumination


def draw_polygons_gouraud(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, 
                         i0, i1, i2, zbuf):
    xmin = min(x0, x1, x2)
    xmax = max(x0, x1, x2)
    ymin = min(y0, y1, y2)
    ymax = max(y0, y1, y2)
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0

    for y in range(int(ymin), int(ymax) + 1):
        for x in range(int(xmin), int(xmax) + 1):
            lambda0, lambda1, lambda2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
            
            if lambda0 >= 0.0 and lambda1 >= 0.0 and lambda2 >= 0.0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                
                
                illumination = lambda0 * i0 + lambda1 * i1 + lambda2 * i2
                intensity = -225 * illumination
                intensity = max(0, min(255, intensity))
                
               
                color = (0, int(intensity), 0)
                
                if zbuf[x][y] >= z:
                    zbuf[x][y] = z
                    img[y][x] = color


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = (((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda1 = (((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


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
    if len(sp) == 0:
        continue
    if sp[0] == 'v':
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])
    elif sp[0] == 'f':
        face_data = []
        for face_part in sp[1:4]:
            parts = face_part.split('/')
            face_data.append(parts[0])
        f.append(face_data)

file_in.close()

alf = 0
bet = 4.15
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


vertex_normals = compute_vertex_normals(v_fix, f)

light_direction = np.array([0, 0, 1])

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
    
    x0_proj = float(scale*x0/z0 + WIDTH/2)
    y0_proj = float(scale*y0/z0 + HEIGHT/2)
    x1_proj = float(scale*x1/z1 + WIDTH/2)
    y1_proj = float(scale*y1/z1 + HEIGHT/2)
    x2_proj = float(scale*x2/z2 + WIDTH/2)
    y2_proj = float(scale * y2/z2 + HEIGHT/2)

    vec1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    vec2 = np.array([x1 - x0, y1 - y0, z1 - z0])
    
    n = np.cross(vec1, vec2)
    l = np.array([0, 0, 1])

    cos_val = (np.dot(n, l)) / (np.linalg.norm(n) * np.linalg.norm(l))
    
    if cos_val >= 0:
        continue

    if z0 <= 0 or z1 <= 0 or z2 <= 0:
        continue

    if not all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in [(x0_proj, y0_proj), (x1_proj, y1_proj), (x2_proj, y2_proj)]):
        continue

   
    v0_idx = int(f[i][0]) - 1
    v1_idx = int(f[i][1]) - 1
    v2_idx = int(f[i][2]) - 1
    
    i0 = compute_vertex_illumination(vertex_normals[v0_idx], light_direction)
    i1 = compute_vertex_illumination(vertex_normals[v1_idx], light_direction)
    i2 = compute_vertex_illumination(vertex_normals[v2_idx], light_direction)
    
    
    draw_polygons_gouraud(img_mat, x0_proj, y0_proj, z0, x1_proj, y1_proj, z1, x2_proj, y2_proj, z2, 
                         i0, i1, i2, z_buffer)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('krolik_zatenenie.png')
img.show()
