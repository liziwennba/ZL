import cv2
import numpy as np
# def canny_edge_detection(image,t):
#     k = np.array(([2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]))/159
#     image_smooth = cv2.filter2D(image,-1,k)
#     kx=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
#     ky = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
#     gx = cv2.filter2D(image_smooth.astype(np.float32),-1,kx)
#     gy = cv2.filter2D(image_smooth.astype(np.float32),-1,ky)
#     print(np.min(gy))
#     cv2.imwrite('street_gx.png',gx)
#     cv2.imwrite('street_gy.png',gy)
#     gx=gx.astype(np.float64)
#     gy=gy.astype(np.float64)
#     w,h = image.shape
#     gxx = np.zeros((w,h))
#     gyy = np.zeros((w,h))
#     g= np.zeros((w,h))
#     g_theta = np.zeros((w,h))
#
#     for i in range(w):
#         for j in range(h):
#             gxx[i,j] = int(gx[i,j])*int(gx[i,j])
#             gyy[i,j] = int(gy[i,j])*int(gy[i,j])
#             res = (gxx[i,j]+gyy[i,j])**0.5
#             g[i,j]=res
#             if gx[i,j]==0:
#                 if gy[i,j]>0:
#                     g_theta[i,j] = np.pi/2
#                 elif gy[i,j]==0:
#                     g_theta[i,j] = 0
#                 else:
#                     g_theta[i,j] = np.pi*3/2
#             else:
#                 g_theta[i,j]=np.arctan(gy[i,j]/gx[i,j])
#     print(np.max(g_theta))
#     g_re = np.zeros((w,h))
#     for i in range(w):
#         for j in range(h):
#             g_theta[i,j] = round(g_theta[i,j]/(np.pi/4))
#             if g_theta[i,j] ==0 or g_theta[i,j]==4:
#                 if i-1>=0 and i+1<w:
#                     if g[i,j]<g[i-1,j] or g[i,j]<g[i+1,j]:
#                         g_re[i,j] =0
#                     else:
#                         g_re[i,j]=g[i,j]
#             elif g_theta[i,j]== 1 or g_theta[i,j]==5:
#                 if g[i,j]<g[i-1,j-1] or g[i,j]<g[i+1,j+1]:
#                     g_re[i,j] =0
#                 else:
#                     g_re[i,j]=g[i,j]
#             elif g_theta[i,j]==2 or g_theta[i,j]==6 or g_theta[i,j]==-2:
#                 if j-1>=0 and j+1<h:
#                     if g[i,j]<g[i,j-1] or g[i,j]<g[i,j+1]:
#                         g_re[i,j] =0
#                     else:
#                         g_re[i,j]=g[i,j]
#             elif g_theta[i,j]==3 or g_theta[i,j]==7 or g_theta[i,j]==-1:
#                 if g[i,j]<g[i-1,j+1] or g[i,j]<g[i+1,j-1]:
#                     g_re[i,j] =0
#                 else:
#                     g_re[i,j]=g[i,j]
#     cv2.imwrite('street_re_re.png', g_re)
#     for i in range(w):
#         for j in range(h):
#             if g_re[i,j] < t:
#                 g_re[i,j] = 0
#     cv2.imwrite('street_re_re_re.png',g_re)
#     cv2.imwrite('street_re.png',g)
#     print(g_theta)
#     print(np.min(g_theta))
#     return
# img=cv2.imread('geisel.jpg',0)
# canny_edge_detection(img,70)




'''
HW3 question 2
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Car.tif',0)
f = np.fft.fft2(img,(512,512))
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

x_axis = np.linspace(-256,255,512)
y_axis = np.linspace(-256,255,512)
[u,v] = np.meshgrid(x_axis,y_axis)
w,h = u.shape
print(u)
print(v)
uk = [-90.01,-90.01,-83.01,-83.01]
vk= [166.01,80.01,-87.01,-173.01]

def Dk(u,v,uk,vk,D0,n):
    return (D0/(((u-uk)**2+(v-vk)**2)**0.5))**(2*n)
def DKM(u,v,uk,vk,D0,n):
    return (D0/(((u+uk)**2+(v+vk)**2)**0.5))**(2*n)
def Hnr(D0,K,u,v,uk,vk,n):
    w,h = u.shape
    res = np.ones((w,h))
    for i in range(w):
        for j in range(h):
            for k in range(K):
                count1 = 1/(1+Dk(u[i,j],v[i,j],uk[k],vk[k],D0,n))
                count2 = 1/(1+DKM(u[i,j],v[i,j],uk[k],vk[k],D0,n))
                res[i,j]*=count1*count2
    return res

hnr = Hnr(5,4,u,v,uk,vk,2)
plt.subplot(133),plt.imshow(hnr)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

fshift_re = fshift*hnr

magnitude_spectrum_re = 20*np.log(np.abs(fshift_re))
# plt.subplot(133),plt.imshow(magnitude_spectrum_re, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

f_ishift = np.fft.ifftshift(fshift_re)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()
# '''
# HW3 question 3
# '''
# import torch
# import torchvision
# import torchvision.transforms as transforms
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=0)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=0)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# import matplotlib.pyplot as plt
# import numpy as np
#
# # functions to show an image
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # # get some random training images
# # dataiter = iter(trainloader)
# # images, labels = dataiter.next()
# #
# # # show images
# # imshow(torchvision.utils.make_grid(images))
# # # print labels
# # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# net = Net()
# import torch.optim as optim
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#
#
# loss_plot = []
# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             loss_plot.append(running_loss/2000)
#             running_loss = 0.0
#
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)
# print('Finished Training')
# x=[]
# mini_batch = 2000
# for i in range(len(loss_plot)):
#     x.append(mini_batch)
#     mini_batch +=2000
# plt.plot(x,loss_plot)
# plt.title('loss vs number of mini-batch')
# plt.xlabel('number of mini-batches')
# plt.ylabel('loss')
# plt.show()
#
#
#
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))
# '''
# 2.
# We are using 12500 images and 3125 batches(batch size 4) to train the network
# 3.
# Yes, we normalizer the image from [0,1] to [-1,1]. In the example, we first unnormalized the random images from training dataset to [0,1], then transform them to numpy array and use matplot to show the images.
# 4.
#
#
#
# '''