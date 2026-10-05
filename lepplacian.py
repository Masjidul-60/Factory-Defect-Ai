import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("himel4.jpeg")


imgt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


med = cv2.medianBlur(imgt, 5)


gray = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY)


lap = cv2.Laplacian(gray, cv2.CV_64F )
lap = cv2.convertScaleAbs(lap)

plt.figure(figsize=(12,6))


plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(imgt)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Median Blur")
plt.imshow(gray, cmap="gray")
plt.axis("off")


plt.subplot(1,3,3)
plt.title("Laplacian Edge")
plt.imshow(lap, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()