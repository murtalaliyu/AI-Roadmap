import matplotlib.pyplot as plt
from digits import train

image, label = train[0]

# reshape 256 values into 16x16
image_2d = [image[r*16:(r+1)*16] for r in range(16)]

plt.imshow(image_2d, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()