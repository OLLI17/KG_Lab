import numpy as np
from PIL import Image, ImageOps
import math
from math import cos, sin, pi

IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 600
PROJECTION_COEFF = 8000
CENTER_X = IMAGE_WIDTH // 2
CENTER_Y = IMAGE_HEIGHT // 2

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self):
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        n = self.norm()
        if n > 0:
            return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
        return self
    
    def __mul__(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return Quaternion(w, x, y, z)
    
    def to_rotation_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])

def euler_to_quaternion(roll, pitch, yaw, degrees=True):
    "углы Эйлера в квартениоы"
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return Quaternion(w, x, y, z)

#
def rotate_matrix(alf, bet, gam):
    "поворот в радианах"
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

def rotate_matrix_degrees(alf_deg, bet_deg, gam_deg):
    "поворот в градусах"
    alf = np.radians(alf_deg)
    bet = np.radians(bet_deg)
    gam = np.radians(gam_deg)
    return rotate_matrix(alf, bet, gam)

def normal_to_polygon(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1-x2, y1-y2, z1-z2])
    v2 = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(v1, v2)

def cos_our(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = normal_to_polygon(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    l = np.array([0, 0, 1])
    return np.dot(n, l)/np.linalg.norm(n)


def compute_vertex_normals(v, f):
    vertex_normals = [np.array([0.0, 0.0, 0.0]) for _ in range(len(v))]
    vertex_face_count = [0 for _ in range(len(v))]
    
    for face in f:
        #треугольники из полигонов с 4+ вершинами
        for i in range(1, len(face) - 1):
            v0_idx = int(face[0]) - 1
            v1_idx = int(face[i]) - 1
            v2_idx = int(face[i + 1]) - 1
            
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

def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = (((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda1 = (((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangulate_polygon(vertex_indices, tex_indices=None):
    "полигоны с 4+ вершинами делим на треугольники"
    triangles = []
    tex_triangles = []
    
    #Веером
    for i in range(1, len(vertex_indices) - 1):
        triangles.append([
            vertex_indices[0],
            vertex_indices[i],
            vertex_indices[i + 1]
        ])
        if tex_indices:
            tex_triangles.append([
                tex_indices[0],
                tex_indices[i],
                tex_indices[i + 1]
            ])
    
    return triangles, tex_triangles


def draw_polygons_gouraud(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, 
                         i0, i1, i2, zbuf, tex_coords=None):
    xmin = math.floor(min(x0, x1, x2))
    xmax = math.ceil(max(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    ymax = math.ceil(max(y0, y1, y2))
    
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax >= IMAGE_WIDTH: xmax = IMAGE_WIDTH - 1
    if ymax >= IMAGE_HEIGHT: ymax = IMAGE_HEIGHT - 1
    
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            lambda0, lambda1, lambda2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
            
            if lambda0 >= 0.0 and lambda1 >= 0.0 and lambda2 >= 0.0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                illumination = lambda0 * i0 + lambda1 * i1 + lambda2 * i2
                
                if zbuf[y][x] > z:
                    zbuf[y][x] = z
                    
                    if texture_array is not None and tex_coords is not None:
                        tex_u = lambda0 * tex_coords[0][0] + lambda1 * tex_coords[1][0] + lambda2 * tex_coords[2][0]
                        tex_v = lambda0 * tex_coords[0][1] + lambda1 * tex_coords[1][1] + lambda2 * tex_coords[2][1]
                        
                        tex_x = int(tex_u * (texture_width - 1))
                        tex_y = int(tex_v * (texture_height - 1))
                        
                        tex_x = min(max(tex_x, 0), texture_width - 1)
                        tex_y = min(max(tex_y, 0), texture_height - 1)
                        
                        tex_color = texture_array[tex_y, tex_x]
                        
                        if illumination > 0:
                            color = (0, 0, 0)
                        else:
                            intensity = -illumination
                            color = (
                                int(tex_color[0] * intensity),
                                int(tex_color[1] * intensity),
                                int(tex_color[2] * intensity)
                            )
                    else:
                        intensity = max(0, min(255, int(-illumination * 255)))
                        color = (0, intensity, 0)
                    
                    img[y][x] = color

def draw_polygons(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, zbuf, color):
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
                if zbuf[y][x] >= z:
                    zbuf[y][x] = z
                    img[y][x] = color

def projective_transform(X, Y, Z, a_X, a_Y, u_0, v_0):
    if Z <= 0:
        return None, None
    u = (a_X * X) / Z + u_0
    v = (a_Y * Y) / Z + v_0
    return u, v


def load_obj(filename):
    v = []
    f = []
    vt = []
    vt_indices = []
    

    file_in = open(filename)
    for line in file_in:
            sp = line.strip().split()
            if len(sp) == 0:
                continue
                
            if sp[0] == 'v':
                v.append([float(sp[1]), float(sp[2]), float(sp[3])])
            elif sp[0] == 'f':
                face_vertices = []
                face_textures = []
                
                for face_part in sp[1:]:
                    parts = face_part.split('/')
                    face_vertices.append(parts[0])
                    
                    if len(parts) > 1 and parts[1]:
                        face_textures.append(parts[1])
                    else:
                        face_textures.append('1')
                
                f.append(face_vertices)
                vt_indices.append(face_textures)
            elif sp[0] == 'vt':
                if len(sp) >= 3:
                    vt.append([float(sp[1]), float(sp[2])])
    
    return v, f, vt, vt_indices

def transform_vertices(v, rotation_mode='euler_deg', rotation_params=None,
                      translation=(0, 0, 0), scale=1.0, scale_xyz=(1, 1, 1)):
    "повороты"
    if scale_xyz == (1, 1, 1):
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
    else:
        scale_matrix = np.array([
            [scale_xyz[0], 0, 0],
            [0, scale_xyz[1], 0],
            [0, 0, scale_xyz[2]]
        ])
    
    #матрица поворота?
    if rotation_mode == 'euler_rad':
        alf, bet, gam = rotation_params
        R = rotate_matrix(alf, bet, gam)
    elif rotation_mode == 'euler_deg':
        alf_deg, bet_deg, gam_deg = rotation_params
        R = rotate_matrix_degrees(alf_deg, bet_deg, gam_deg)
    elif rotation_mode == 'quaternion':
        q = rotation_params
        R = q.to_rotation_matrix()
    else:
        R = np.eye(3)
    
    #матрица преобразования
    M = R @ scale_matrix
    t = np.array(translation)
    
    v_fix = []
    for vertex in v:
        vertex_array = np.array(vertex)
        v_transform = M @ vertex_array + t
        v_fix.append(v_transform)
    
    return v_fix


def render_model(img_mat, z_buffer, obj_file, texture_file, 
                rotation_mode='euler_deg', rotation_params=None,
                translation=(0, 0, 0), scale=1.0, scale_xyz=(1, 1, 1)):
    "рендеринг модели с заданными параметрами"
    global texture_array, texture_width, texture_height
    
    #загрузка текстуры
    if texture_file:
        texture_img = ImageOps.flip(Image.open(texture_file))
        texture_array = np.array(texture_img)
        texture_width, texture_height = texture_img.size
    else:
        texture_array = None
    
    #загрузка модели
    v, f, vt, vt_indices = load_obj(obj_file)
    
    #преобразование вершин
    v_fix = transform_vertices(v, rotation_mode, rotation_params, 
                              translation, scale, scale_xyz)
    
    #вычисление нормалей
    vertex_normals = compute_vertex_normals(v_fix, f)
    
    #направление света
    light_direction = np.array([0, 0, 1])
    
    #рендеринг каждого полигона
    for i in range(len(f)):
        face_vertices = f[i]
        face_tex_indices = vt_indices[i] if i < len(vt_indices) else None
        
        #разбиение полигона на треугольники
        triangles, tex_triangles = triangulate_polygon(face_vertices, face_tex_indices)
        
        for tri_idx, triangle in enumerate(triangles):
            
            x0 = v_fix[int(triangle[0]) - 1][0]
            y0 = v_fix[int(triangle[0]) - 1][1]
            z0 = v_fix[int(triangle[0]) - 1][2]
            
            x1 = v_fix[int(triangle[1]) - 1][0]
            y1 = v_fix[int(triangle[1]) - 1][1]
            z1 = v_fix[int(triangle[1]) - 1][2]
            
            x2 = v_fix[int(triangle[2]) - 1][0]
            y2 = v_fix[int(triangle[2]) - 1][1]
            z2 = v_fix[int(triangle[2]) - 1][2]
            
            #проекция
            x0_proj = (PROJECTION_COEFF * x0 / z0) + CENTER_X
            y0_proj = (PROJECTION_COEFF * y0 / z0) + CENTER_Y
            x1_proj = (PROJECTION_COEFF * x1 / z1) + CENTER_X
            y1_proj = (PROJECTION_COEFF * y1 / z1) + CENTER_Y
            x2_proj = (PROJECTION_COEFF * x2 / z2) + CENTER_X
            y2_proj = (PROJECTION_COEFF * y2 / z2) + CENTER_Y
            
            #проверка нормали для отсечения задних граней
            vec1 = np.array([x1 - x2, y1 - y2, z1 - z2])
            vec2 = np.array([x1 - x0, y1 - y0, z1 - z0])
            n = np.cross(vec1, vec2)
            l = np.array([0, 0, 1])
            
            if np.linalg.norm(n) == 0:
                continue
                
            cos_val = np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))
            
            if cos_val >= 0:
                continue
            
            if z0 <= 0 or z1 <= 0 or z2 <= 0:
                continue
            
            #границы
            if not all(0 <= x < 2000 and 0 <= y < 2000
                      for x, y in [(x0_proj, y0_proj), (x1_proj, y1_proj), (x2_proj, y2_proj)]):
                continue
            
            #освещение вершин
            v0_idx = int(triangle[0]) - 1
            v1_idx = int(triangle[1]) - 1
            v2_idx = int(triangle[2]) - 1
            
            i0 = compute_vertex_illumination(vertex_normals[v0_idx], light_direction)
            i1 = compute_vertex_illumination(vertex_normals[v1_idx], light_direction)
            i2 = compute_vertex_illumination(vertex_normals[v2_idx], light_direction)
            
            #текстурные координаты
            tex_coords = None
            if vt and tex_triangles:
                try:
                    tex_coords = [
                        vt[int(tex_triangles[tri_idx][0]) - 1] if int(tex_triangles[tri_idx][0]) - 1 < len(vt) else [0.0, 0.0],
                        vt[int(tex_triangles[tri_idx][1]) - 1] if int(tex_triangles[tri_idx][1]) - 1 < len(vt) else [1.0, 0.0],
                        vt[int(tex_triangles[tri_idx][2]) - 1] if int(tex_triangles[tri_idx][2]) - 1 < len(vt) else [0.0, 1.0]
                    ]
                except:
                    tex_coords = [

                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]
                    ]
            
            #отрисовка
            draw_polygons_gouraud(img_mat, x0_proj, y0_proj, z0, x1_proj, y1_proj, z1, 
                                 x2_proj, y2_proj, z2, i0, i1, i2, z_buffer, tex_coords)

#запуск
if __name__ == "__main__":

    img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    z_buffer = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 10000000.0)
    
    #углов Эйлера в градусах
    print("углы Эйлера")
    render_model(
        img_mat=img_mat,
        z_buffer=z_buffer,
        obj_file='model_1.obj',
        texture_file='bunny-atlas.jpg',
        rotation_mode='euler_deg',
        rotation_params=(180, 270, 180),  # alf, bet, gam в градусах
        translation=(0.009, -0.01, 0.3),
        scale=0.5
    )
    
    #вторая модель с кватернионом
    print("кватернионы")
    # кватернион поворота
    q = euler_to_quaternion(180, 90, 270, degrees=True)
    
    render_model(
        img_mat=img_mat,
        z_buffer=z_buffer,
        obj_file='12221_Cat_v1_l3 (2).obj',  
        texture_file='Cat_diffuse (2).jpg',
        rotation_mode='quaternion',
        rotation_params=q,
        translation=(-0.07, -0.01, 0.9),
        scale=0.0018
    )
    
    # с углами в радианах
   # print("углs Эйлера в радианах")
   # render_model(
      #  img_mat=img_mat,
       # z_buffer=z_buffer,
        #obj_file='model_1.obj',
        #texture_file='bunny-atlas.jpg',
        #rotation_mode='euler_rad',
        #rotation_params=(np.radians(45), np.radians(90), np.radians(135)),  #в радианах дэ...
        #translation=(0.0, 0.0, 0.4),
        #scale_xyz=(0.5, 0.7, 0.5)  #неравномерно дэ...
    #)
    
    
    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('wow.png')
    img.show()
    
    print("goodgame")

