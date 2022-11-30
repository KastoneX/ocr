from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np

# параметры цветового фильтра
hsv_min = np.array((1, 10, 20), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

# построить разбор аргументов и разбор аргументов
if __name__ == '__main__':
    print(__doc__)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image" )
ap.add_argument('-f')
ap.add_argument("-p", "--preprocess", type=str, default="thresh")
args = vars(ap.parse_args())

image = cv2.imread("/content/1.jpeg")

# меняем цветовую модель с BGR на HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# черно-белые
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


gray, mask = cv2.threshold(gray, 150, cv2 .ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY)

# найти контуры
contours, gray = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# выберите самый большой контур
largest_area = 0
for cnt in contours:
    if cv2.contourArea(cnt) > largest_area:
        cont = cnt
        largest_area = cv2.contourArea(cnt)

# найдите прямоугольник (и угловые точки этого прямоугольника), который окружает контуры / фото
rect = cv2.minAreaRect(cont)
box = cv2.boxPoints(rect)
box = np.int0(box)

# назначение угловых точек интересующего региона
pts1 = np.float32([box[1], box[2], box[3], box[0]])

# предоставление новых координат угловых точек
pts2 = np.float32([[0, 0], [50, 50], [50, 50], [0, 0]])

# определение и применение матрицы преобразования
M = cv2.getPerspectiveTransform(pts1, pts2)
tmp = cv2.warpPerspective(image, M, (500, 500))

# черно-белые
gray_image2 = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

# порог выполнения
gray, mask2 = cv2.threshold(gray_image2, 100, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.inRange(hsv, hsv_min, hsv_max)
# ищем контуры и складируем их в переменную contours
gray, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# отображаем контуры поверх изображения
cv2.drawContours(image, contours, -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 0)
cv2.drawContours(image, contours, -1, (50, 10, 0), 1, cv2.LINE_AA, hierarchy, 2)

 #удалить шум / закрыть зазоры
kernel = np.ones((5, 5), np.uint8)
gray = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

# нарисовать прямоугольник на исходном изображении
cv2.drawContours(image, [box], 0, (255, 0, 0), 2)


# проверьте, следует ли применять пороговое значение для предварительной обработки изображения
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# сделать проверку, чтобы увидеть, если медиана размытие должно быть сделано, чтобы удалить шум
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# запишите изображение в оттенках серого на диск как временный файл, чтобы мы могли применение OCR к нему
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, image)


text = pytesseract.image_to_string(image, lang='rus+eng')
print(text)


# показать выходные изображения
cv2.imshow('image', image)
#cv2.imshow("Output", gray)
cv2.waitKey(0)
