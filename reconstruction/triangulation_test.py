from triangulation import triangulate, calculate_ray_error
from camera import Camera

cameras = [
    Camera(
        position=(0, 0, 5),
        azimuth=0,
        f_x = 1000,
        f_y = 1000,
        c_x = 860,
        c_y = 540,
        width=1920,
        height=1080,
    ),

    Camera(
        position=(10, 0, 5),
        azimuth=270,
        f_x = 1000,
        f_y = 1000,
        c_x = 860,
        c_y = 540,
        width=1920,
        height=1080,
    ),

    Camera(
        position=(5, 10, 5),
        azimuth=180,
        f_x = 1000,
        f_y = 1000,
        c_x = 860,
        c_y = 540,
        width=1920,
        height=1080,
    ),
]

# Пример детекций:
detections = [
    (960, 540),
    (960, 540),
    (960, 540),
]

point = triangulate(cameras, detections)

print("Координаты объекта:", point)

print("Среднее геометрическое расстояние (ошибка) от 3D-точки в пространстве до лучей зрения камер",
      calculate_ray_error(point_3d = point, cameras = cameras, detections = detections))