import numpy as np
import math

# параметры камеры
focal_length_mm = 35.000
matrix_width_mm = 23.760
matrix_height_mm = 13.365

# параметры изиображения (в пикселях)
width = 1920
height = 1080

# координаты главной точки
c_x = width/2
c_y = height/2

# перевод фокусных расстояний в пиксели
f_x = (focal_length_mm * width) / matrix_width_mm
f_y = (focal_length_mm * height) / matrix_height_mm

# позиции камер
position_cam_1 = tuple[660.0, 760.0, 35.0]
position_cam_2 = tuple[810.0, 740.0, 45.0]
position_cam_3 = tuple[900.0, -400.0, 80.0]

# азимутальные углы в градусах
azimuth_deg_cam_1 = -110.0
azimuth_deg_cam_2 = -125.0
azimuth_deg_cam_3 = -225.0

# азимутальные углы в радианах
azimuth_rad_cam_1 = math.radians(azimuth_deg_cam_1)
azimuth_rad_cam_2 = math.radians(azimuth_deg_cam_2)
azimuth_rad_cam_3 = math.radians(azimuth_deg_cam_3)

# класс камеры
class Camera:
    def __init__(
        self,
        position: tuple[float, float, float],
        azimuth: float,
        f_x: float,
        f_y: float,
        c_x: float,
        c_y: float,
        width: int,
        height: int,
    ):
        
        self.position = np.asarray(position, dtype=float)
        
        self.f_x = f_x
        self.f_y = f_y
        
        self.c_x = c_x
        self.c_y = c_y
        
        self.width = width
        self.height = height
        
        c = math.cos(azimuth)
        s = math.sin(azimuth)
        
        self.R = np.array([
            [c, -s, 0],
            [0, 0,  1],
            [s, c,  0],
        ])
        
        self.K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0,  0,  1],
        ])
        
        
cam_1 = Camera(position_cam_1, azimuth_rad_cam_1, f_x, f_y, c_x, c_y, width, height)  
cam_2 = Camera(position_cam_2, azimuth_rad_cam_2, f_x, f_y, c_x, c_y, width, height)  
cam_3 = Camera(position_cam_3, azimuth_rad_cam_3, f_x, f_y, c_x, c_y, width, height)  

