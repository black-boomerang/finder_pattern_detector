# По мотивам https://aishack.in/tutorials/scanning-qr-codes-1/

import cv2
import math
import numpy as np


class finderPatternDetector:
    def __init__(self):
        self.centers = []
        self.patternBlockSizes = []
    
    def find_patterns(self, image):
        """
        Поиск шаблонов на изображении.
        :param image: Исходное изображение.
        :return: Список, содержащий четвёрки чисел - координаты левых верхних и
                 правых нижних углов найденных шаблонов.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 305, 5)
        height, width = image.shape
        
        self.centers = []
        self.patternBlockSizes = []
        
        for row in range(height):
            stripsWidth = [0] * 5
            currentStrip = 0
            
            for column in range(width):
                if image[row][column] == 0:  # обработка чёрного пикселя
                    if currentStrip % 2 == 1:
                        currentStrip += 1
                        
                    stripsWidth[currentStrip] += 1
                else:  # обработка белого пикселя
                    if currentStrip % 2 == 1:
                        stripsWidth[currentStrip] += 1
                    elif currentStrip == 4:  # заканчиваем обработку шаблона
                            if self.check_pattern(image, stripsWidth, row, column):
                                stripsWidth = [0] * 5
                                currentStrip = 0
                            else:
                                stripsWidth[:3] = stripsWidth[2:]
                                stripsWidth[3] = stripsWidth[4] = 0
                                currentStrip = 3
                    else:
                        currentStrip += 1
                        stripsWidth[currentStrip] = 1
        
        patterns = []
        for center, patternBlockSize in zip(self.centers, self.patternBlockSizes):
            halfSide = patternBlockSize * 3.5
            patterns.append((int(center[0] - halfSide), int(center[1] - halfSide),
                             int(center[0] + halfSide), int(center[1] + halfSide)))
        
        return patterns
    
    def check_ratio(self, stripsWidth, totalWidth):
        """
        Проверка чёрно-белой полоски на соответствие пропорции 1:1:3:1:1 (с некоторой погрешностью).
        :param stripsWidth: Список размеров полосок чередующихся цветов.
        :param totalWidth: Суммарная ширина полоски.
        :return: True, если полоска соответствует пропорции 1:1:3:1:1
        """
        if totalWidth < 7 or min(stripsWidth) == 0:
            return False
        
        blockSize = math.ceil(totalWidth / 7)
        variance = blockSize // 2
        
        is_suitable = abs(blockSize - stripsWidth[0]) < variance
        is_suitable &= abs(blockSize - stripsWidth[1]) < variance
        is_suitable &= abs(3 * blockSize - stripsWidth[2]) < 3 * variance
        is_suitable &= abs(blockSize - stripsWidth[3]) < variance
        is_suitable &= abs(blockSize - stripsWidth[4]) < variance
        
        return is_suitable
    
    def find_pattern_center(self, stripsWidth, last_column):
        """
        Поиск центрального столбца шаблона по последнему столбцу.
        :param stripsWidth: Список размеров полосок чередующихся цветов.
        :param last_column: Номер последнего столбца шаблона.
        :return: Номер предположительного центрального столбца шаблона.
        """
        return int(last_column - stripsWidth[4] - stripsWidth[3] - stripsWidth[2] / 2)
    
    def check_pattern(self, image, stripsWidth, row, column):
        """
        Проверка чёрно-белой полоски на соответствие пропорции 1:1:3:1:1 (с некоторой погрешностью).
        :param image: Исходное изображение.
        :param stripsWidth: Список размеров полосок чередующихся цветов.
        :param row: Номер строки.
        :param column: Номер столбца.
        :return: True, если полоска соответствует пропорции 1:1:3:1:1, False иначе.
        """
        totalWidth = sum(stripsWidth)
        
        is_suitable = self.check_ratio(stripsWidth, totalWidth)
        if not is_suitable:
            return False
        
        centerColumn = self.find_pattern_center(stripsWidth, column)
        centerRow = self.cross_check_vertical(image, row, centerColumn, stripsWidth[2], totalWidth)
        
        if centerRow == -1:
            return False
        
        centerColumn = self.cross_check_vertical(image.T, centerColumn, centerRow, stripsWidth[2], totalWidth)
        
        if centerColumn == -1 or not self.cross_check_diagonal(image, centerRow, centerColumn, stripsWidth[2], totalWidth):
            return False
        else:
            newBlockSize = totalWidth / 7
            for i, (y, x) in enumerate(self.centers):
                distance = math.hypot(y - centerRow, x - centerColumn)
                if distance < 10:
                    self.centers[i] = ((y + centerRow) / 2, (x + centerColumn) / 2)
                    self.patternBlockSizes[i] = (self.patternBlockSizes[i] + newBlockSize) / 2
                    return is_suitable
            self.centers.append((centerRow, centerColumn))
            self.patternBlockSizes.append(newBlockSize)
        
        return is_suitable
    
    def cross_check_vertical(self, image, startRow, column, centralWidth, totalWidth):
        """
        Проверка вертикальной чёрно-белой полоски и поиск центральной строки шаблона.
        :param image: Исходное изображение.
        :param startRow: Номер текущей строки.
        :param column: Номер предполагаемого центрального столбца.
        :param centralWidth: Ширина центрального блока полоски.
        :param totalWidth: Суммарная ширина полоски.
        :return: Найденная центральная строка шаблона. Если шаблон не прошёл проверку, возвращается -1.
        """
        height = image.shape[0]
        crossStripsWidth = [0] * 5
        
        row = startRow
        while row >= 0 and image[row][column] == 0:
            crossStripsWidth[2] += 1
            row -= 1
        if row < 0:
            return -1
        
        for i in [1, 0]:
            while row >= 0 and image[row][column] == i * 255 and crossStripsWidth[i] < centralWidth:
                crossStripsWidth[i] += 1
                row -= 1
            if row < 0 or crossStripsWidth[i] >= centralWidth:
                return -1
            
        row = startRow + 1
        while row < height and image[row][column] == 0:
            crossStripsWidth[2] += 1
            row += 1
        if row >= height:
            return -1
        
        for i in [3, 4]:
            while row < height and image[row][column] == (4 - i) * 255 and crossStripsWidth[i] < centralWidth:
                crossStripsWidth[i] += 1
                row += 1
            if row >= height or crossStripsWidth[i] >= centralWidth:
                return -1
            
        crossTotalWidth = sum(crossStripsWidth)
        if 5 * abs(crossTotalWidth - totalWidth) >= 2 * totalWidth:
            return -1
        
        centerRow = self.find_pattern_center(crossStripsWidth, row)
        return centerRow if self.check_ratio(crossStripsWidth, crossTotalWidth) else -1
    
    def cross_check_diagonal(self, image, row, column, centralWidth, totalWidth):
        """
        Диагональная проверка шаблона.
        :param image: Исходное изображение.
        :param row: Номер предполагаемой центральной строки.
        :param column: Номер предполагаемого центрального столбца.
        :param centralWidth: Ширина центрального блока полоски.
        :param totalWidth: Суммарная ширина полоски.
        :return: True, если шаблон прошёл проверку, False иначе.
        """
        height, width = image.shape
        crossStripsWidth = [0] * 5
        
        shift = 0
        while row >= shift and column >= shift and image[row - shift][column - shift] == 0:
            crossStripsWidth[2] += 1
            shift += 1
        if row < shift or column < shift:
            return False
            
        for i in [1, 0]:
            while row >= shift and column >= shift and image[row - shift][column - shift] == i * 255 and \
                  crossStripsWidth[i] <= centralWidth:
                crossStripsWidth[i] += 1
                shift += 1
            if row < shift or column < shift or crossStripsWidth[i] > centralWidth:
                return False
            
        shift = 1
        while row + shift < height and column + shift < width and image[row + shift][column + shift] == 0:
            crossStripsWidth[2] += 1
            shift += 1
        if row + shift >= height or column + shift >= width:
            return False
        
        for i in [3, 4]:
            while row + shift < height and column + shift < width and image[row + shift][column + shift] == (4 - i) * 255 and \
                    crossStripsWidth[i] <= centralWidth:
                crossStripsWidth[i] += 1
                shift += 1
            if row + shift >= height or column + shift >= width or crossStripsWidth[i] > centralWidth:
                return False
            
        crossTotalWidth = sum(crossStripsWidth)
        return abs(crossTotalWidth - totalWidth) < 2 * totalWidth and self.check_ratio(crossStripsWidth, crossTotalWidth)
