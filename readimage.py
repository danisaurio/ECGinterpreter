import matplotlib.pyplot as plt
from skimage import data, io, filters
from skimage.color import rgb2gray
import numpy as np

original = io.imread('ecg.jpg')
grayscale = rgb2gray(original)

print("shape of image: {}".format(grayscale.shape))
print("dtype of image: {}".format(grayscale.dtype))

plt.imshow(grayscale, cmap=plt.cm.gray)
plt.title('Grayscale ECG')

sobel = filters.sobel(grayscale)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200
plt.imshow(sobel)

blurred = filters.gaussian(sobel, sigma=2.0)
plt.imshow(blurred)
plt.savefig('image')

from skimage import data
from skimage.exposure import histogram
hist, hist_centers = histogram(grayscale, normalize=True)

from skimage.feature import canny
edges = canny(grayscale/1.)

plt.imshow(edges)

# plt.savefig('image')

# from skimage.feature import canny
# edges = canny(grayscale/255.)
# from scipy import ndimage as ndi
# fill_coins = ndi.binary_fill_holes(edges)

# label_objects, nb_labels = ndi.label(fill_coins)
# sizes = np.bincount(label_objects.ravel())
# mask_sizes = sizes > 20
# mask_sizes[0] = 0
# coins_cleaned = mask_sizes[label_objects]
# plt.savefig('image')

