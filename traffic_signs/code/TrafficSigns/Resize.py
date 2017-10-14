import cv2
import matplotlib.pyplot as plt
import random


for i in range(0,5):
    index=random.randint(1,1000)
    fname="/prj/selfLearningCar/LS2/GTSRB/Final_Test/Images/{0:05d}.ppm".format(index)
    print(fname)
    img=cv2.imread(fname)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(32,32))
    plt.imshow(img,cmap='gray')
    plt.show()
