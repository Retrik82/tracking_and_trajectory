import numpy as np
import math
from reconstruction.camera import Camera


# параметры камеры
focal_length_mm_cam_1 = 35.000
focal_length_mm_cam_2 = 60.000
focal_length_mm_cam_3 = 35.000
focal_length_mm_cam_4 = 50.000
focal_length_mm_cam_5 = 20.000

matrix_width_mm = 23.760
matrix_height_mm = 13.365

# параметры изиображения (в пикселях)
width = 1920
height = 1080

# координаты главной точки
c_x = width/2
c_y = height/2

# перевод фокусных расстояний в пиксели
f_x_1 = (focal_length_mm_cam_1 * width) / matrix_width_mm
f_y_1 = (focal_length_mm_cam_1 * height) / matrix_height_mm

f_x_2 = (focal_length_mm_cam_2 * width) / matrix_width_mm
f_y_2 = (focal_length_mm_cam_2 * height) / matrix_height_mm

f_x_3 = (focal_length_mm_cam_3 * width) / matrix_width_mm
f_y_3 = (focal_length_mm_cam_3 * height) / matrix_height_mm

f_x_4 = (focal_length_mm_cam_4 * width) / matrix_width_mm
f_y_4 = (focal_length_mm_cam_4 * height) / matrix_height_mm

f_x_5 = (focal_length_mm_cam_5 * width) / matrix_width_mm
f_y_5 = (focal_length_mm_cam_5 * height) / matrix_height_mm


# Передаем ваши f_x, f_y, c_x, c_y напрямую
cam_1 = Camera.from_csv_frame(
    file_path="D:\projects\tracking_and_trajectory\data\object_transform_CineCameraActor1.csv",
    frame_number=15,
    f_x=f_x_1, f_y=f_y_1,
    c_x=c_x, c_y=c_y,
    width=width, height=height
)

for current_frame in range(0, 100):
    cam_1.update_from_csv_frame(r"data\object_transform_CineCameraActor1.csv", frame_number=current_frame)
    print(cam_1.position)
    # Камера сместилась, f_x и f_y не тронуты
        
 
