import csv
import math
import numpy as np


class Camera:
    def __init__(
        self,
        position: tuple[float, float, float],
        roll: float,   # Передаем углы в радианах
        pitch: float,
        yaw: float,
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
        
        # Матрица внутренних параметров камеры K
        self.K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0,  0,    1] ], dtype=float)
        
        # Строим матрицу поворота R на основе переданных углов
        self.update_rotation(roll, pitch, yaw)

    def update_rotation(self, roll: float, pitch: float, yaw: float):
        """Строит полноценную 3D-матрицу поворота R на основе углов в радианах."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        # Матрицы поворота по осям Z, Y, X
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        
        # Итоговая матрица поворота
        self.R = R_z @ R_y @ R_x

    @classmethod
    def from_csv_frame(
        cls, 
        file_path: str, 
        frame_number: int, 
        f_x: float, 
        f_y: float,
        c_x: float, 
        c_y: float, 
        width: int, 
        height: int
    ):
        """
        Фабричный метод: ищет нужный кадр в CSV-файле, забирает оттуда 
        координаты/ориентацию и создает объект Camera с вашими f_x и f_y.
        """
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Пропускаем строку заголовка
            
            for row in reader:
                # Игнорируем пустые строки и строки с метаданными матрицы
                if not row or not row[0].strip() or row[0].startswith('матрица'):
                    continue
                
                try:
                    current_frame = int(row[0])
                    if current_frame != frame_number:
                        continue
                    
                    # Извлекаем позицию XYZ
                    position = (float(row[1]), float(row[2]), float(row[3]))
                    
                    # Извлекаем углы в градусах и переводим в радианы
                    roll = math.radians(float(row[4].replace(',', '.')))
                    pitch = math.radians(float(row[5].replace(',', '.')))
                    yaw = math.radians(float(row[6].replace(',', '.')))
                    
                    # Создаем объект камеры с переданными f_x и f_y
                    return cls(
                        position=position,
                        roll=roll, pitch=pitch, yaw=yaw,
                        f_x=f_x, f_y=f_y,
                        c_x=c_x, c_y=c_y,
                        width=width, height=height
                    )
                    
                except (ValueError, IndexError):
                    continue
                    
        raise ValueError(f"Кадр {frame_number} не найден в файле {file_path}")

    def update_from_csv_frame(self, file_path: str, frame_number: int):
        """Метод для существующего объекта камеры: обновляет только позицию и вращение из файла."""
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)
            
            for row in reader:
                if not row or not row[0].strip() or row[0].startswith('матрица'):
                    continue
                
                try:
                    if int(row[0]) == frame_number:
                        # Обновляем позицию XYZ
                        self.position = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=float)
                        
                        # Обновляем углы вращения
                        roll = math.radians(float(row[4].replace(',', '.')))
                        pitch = math.radians(float(row[5].replace(',', '.')))
                        yaw = math.radians(float(row[6].replace(',', '.')))
                        self.update_rotation(roll, pitch, yaw)
                        return True
                except (ValueError, IndexError):
                    continue
        return False
