%autosave 120

from ctypes import resize
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
#!pip install opencv-python
import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

from ctypes import resize
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
#!pip install opencv-python
import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow


dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = 'E:/data_augumentation/speedsignal/'
SIZE = 128
dataset = []
imagess=[]
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] in ['jpg','jpeg','png','html']):
        image = io.imread(image_directory + image_name)
        imagess.append(cv2.imread(image_name))
        #print(str(image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))



# TYPE 1

img=cv2.imread('E:/data_augumentation/img.jpg')
img = img/255.0
fig=plt.figure(dpi=300)

gauss_noise=np.zeros((img.shape),dtype=np.uint8)
cv2.randn(gauss_noise,128,20)
gauss_noise=(gauss_noise*0.5).astype(np.uint8)
gn_img=cv2.add(img,gauss_noise)

plt.axis("off")
plt.imshow(gn_img)
plt.savefig("result.png")

uni_noise=np.zeros((img.shape),dtype=np.uint8)
cv2.randu(uni_noise,0,255)
uni_noise=(uni_noise*0.5).astype(np.uint8)
un_img=cv2.add(img,uni_noise)

plt.axis("off")
plt.imshow(un_img)
plt.savefig("result1.png")

imp_noise=np.zeros((img.shape),dtype=np.uint8)
cv2.randu(imp_noise,0,255)
imp_noise=cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]
in_img=cv2.add(img,imp_noise)

plt.axis("off")
plt.imshow(in_img)
plt.savefig("result2.png")
#plt.title("impulse")
#fig.add_subplot(2,3,2)


# TYPE 2
import cv2
import numpy as np
from skimage.util import random_noise

# Load the image
img = cv2.imread("E:/data_augumentation/img.jpg")

# Add salt-and-pepper noise to the image.
noise_img = random_noise(img, mode='gaussian',amount=0.3)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')

# Display the noise image
# cv2.imshow('blur',noise_img)
cv2.waitKey(0)


img = 'aug_0_3094.png'
from google.colab.patches import cv2_imshow
img = cv2.imread(img)
bfilter = cv2.bilateralFilter(img, 11, 17, 17)
#=img.reshape(-1,28,28,1)
type(img)
# plt.imshow(img)


datagen = ImageDataGenerator(
        rotation_range=30,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.5,
        zoom_range=0.4,
        horizontal_flip=False,
        brightness_range = (0.001, 2.5),
        fill_mode='nearest')    #Also try nearest, constant, reflect, wrap

#Loading a single image for demonstration purposes.
#Using flow method to augment the image

# Loading a sample image  
#Can use any library to read images but they need to be in an array form
#If using keras load_img convert it to an array first
x = io.imread('E:\data_augumentation\speedsignal\railway-track-number-board-548x411.jpg')
# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images
x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

i = 0
for batch in datagen.flow(x, batch_size=32,  
                          save_to_dir='augumented', 
                          save_prefix='aug', 
                          save_format='png'):
        
        # 3 blurrs
        x1=cv2.imread('E:\data_augumentation\speedsignal\railway-track-number-board-548x411.jpg')
        #cv2.imshow('blurred image',i)
        #plt.imshow(x)
        gauss = cv2.GaussianBlur(x1,(9,9) ,0)
        #plt.imshow(blurred_original_image)
        avging = cv2.blur(x1,(10,10))
        #cv2_imshow(avging)
        medBlur = cv2.medianBlur(x1,1)
        #cv2_imshow( medBlur)
        
        # hori vertical blurrs
        
        kernel_size = 30
  
        # Create the vertical kernel.
        kernel_v = np.zeros((kernel_size, kernel_size))
        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.copy(kernel_v)
        # Fill the middle row with ones.
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        # Normalize.
        kernel_v /= kernel_size
        kernel_h /= kernel_size
        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(x1, -1, kernel_v)
        # Apply the horizontal kernel.
        horizonal_mb = cv2.filter2D(x1, -1, kernel_h)
        
        # Save the outputs.
        #cv2.imwrite('car_vertical.jpg', vertical_mb)
        #cv2.imwrite('car_horizontal.jpg', horizonal_mb)

        # save files into the folder
        #for n in range(0, len(onlyfiles)):
        # other things you need to do snipped
        c=20
        cv2.imwrite(f'E:/data_augumentation/augumented/image.png',gauss)
        cv2.imwrite(f'E:/data_augumentation/augumented/image1_{c}.png',avging)
        cv2.imwrite(f'E:/data_augumentation/augumented/image2_{c}.png',medBlur)
        cv2.imwrite(f'E:/data_augumentation/augumented/image3_{c}.png',vertical_mb)
        cv2.imwrite(f'E:/data_augumentation/augumented/image4_{c}.png',horizonal_mb)
            
        
        i += 1
        if i > 15:
                break  # otherwise the generator would loop indefinitely


# tries NOISE

img = cv2.imread('E:/data_augumentation/speedsignal/aug_0_72.png')[...,::-1]/255.0
noise =  np.random.normal(loc=0, scale=1, size=img.shape)

# noise overlaid over image
noisy = np.clip((img + noise*0.2),0,1)
noisy2 = np.clip((img + noise*0.4),0,1)

# noise multiplied by image:
# whites can go to black but blacks cannot go to white
noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

# noise multiplied by bottom and top half images,
# whites stay white blacks black, noise is added to center
img2 = img*2
n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)


# norm noise for viz only
noise2 = (noise - noise.min())/(noise.max()-noise.min())
plt.figure(figsize=(5,5))
#plt.imshow(np.vstack((np.hstack((img, n2)))))
# plt.show(noise2)
# plt.show()