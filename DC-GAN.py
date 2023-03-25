

#C:\Users\shiha\Downloads\archive.zip\img_align_celeba\img_align_celeba
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


images = []
for img_path in glob.glob('C:/Users/shiha/OneDrive/Desktop/machine learning/img_align_celeba/img_align_celeba/*.jpg'):
    images.append(mpimg.imread(img_path))
images = images[:20]
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.axis('off')
    #fig.tight_layout() 
    plt.imshow(image)
plt.show()