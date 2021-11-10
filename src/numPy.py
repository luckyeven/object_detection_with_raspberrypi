# Author: Shifeng song
# 
#
import numpy as np
import matplotlib.pyplot as plt
import PIL
########################################################
print("#gray image from here:")

#Settings: location of image to load
image_path = "../images/5x4.bmp"

# Open image and convert to a Numpy array
img = PIL.Image.open(image_path).convert('L')
#img.show()
print(img)

# Convert to a Numpy array

img_array = np.asarray(img)
print(img_array)
print("(num rows, num columns)")
print()
print(img_array)

# Show normalized array (values between 0 and 1)
print(img_array/ 255)

# Display image from array
plt.figure()
plt.imshow(img_array, cmap= 'gray', vmin=0, vmax= 255)


img_vector = img_array.reshape(1, 20)
print(img_vector)

##########################################################
print("RGP image from here")
#Settings: location of image to load
image_colour_path = "../images/5x4_colour.bmp"

# Open image and convert to a Numpy array
img_coluor = PIL.Image.open(image_colour_path).convert("RGB")
print(img_coluor)
#img.show()


# Convert to a Numpy array

img_colour_array = np.asarray(img_coluor)

print(img_colour_array)
print("(num rows, num columns)")
print()
print(img_coluor)

# Show normalized array (values between 0 and 1)
print(img_colour_array/ 255)

# Display image from array
plt.figure()
plt.imshow(img_colour_array, cmap= 'gray', vmin=0, vmax= 255)


img_colour_vector = img_colour_array.reshape(1, 60)
print(img_colour_vector)



##########################################################
# Create 3 copies of the array
img_r = img_colour_array.copy()
img_g = img_colour_array.copy()
img_b = img_colour_array.copy()

# Fill non-red channedls with 0s
img_r[:,:,1].fill(0)
img_r[:,:,2].fill(0)

# Fill non_green channels with 0s
img_g[:,:,0].fill(0)
img_g[:,:,2].fill(0)

# Fill non_blue channels with 0s
img_b[:,:,0].fill(0)
img_b[:,:,1].fill(0)

# Plot each of the RGB layers separately
plt.figure()
plt.imshow(img_r, vmin=0, vmax= 255)
plt.figure()
plt.imshow(img_g, vmin=0, vmax=255)
plt.figure()
plt.imshow(img_b, vmin=0, vmax=255)


# Plot all layers together
plt.figure()
plt.imshow(img_colour_array, vmin=0, vmax= 255)
plt.show()