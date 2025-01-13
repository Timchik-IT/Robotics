import cv2
import numpy as np

image = cv2.imread("holder.png")

if image is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
adaptive_thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

num_objects = len(contours)
print(f"Количество объектов: {num_objects}")

output_image = image.copy()
areas = []
centers = []

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    areas.append(area)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

        cv2.putText(output_image, f"Area: {int(area)}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        print(f"Контур {i}: нулевая площадь.")

if areas:
    largest_idx = np.argmax(areas)
    smallest_idx = np.argmin(areas)

    largest_center = centers[largest_idx] if len(centers) > largest_idx else None
    smallest_center = centers[smallest_idx] if len(centers) > smallest_idx else None

    if largest_center:
        cv2.circle(output_image, largest_center, 5, (0, 255, 255), -1)
        cv2.putText(output_image, f"Largest: {largest_center}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        print(f"Центр самого большого объекта: {largest_center}")

    if smallest_center:
        cv2.circle(output_image, smallest_center, 5, (255, 210, 0), -1)
        cv2.putText(output_image, f"Smallest: {smallest_center}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 0), 1)
        print(f"Центр самого маленького объекта: {smallest_center}")
else:
    print("Контуры не найдены.")

cv2.drawContours(output_image, contours, -1, (0, 252, 124), 1)
cv2.imshow("Output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()