import numpy as np
import cv2
import colorsys

import numpy as np
import cv2
import colorsys

# def _get_colors(num_colors):
#     colors=[]
#     for i in np.arange(0., 360., 360. / num_colors):
#         hue = i/360.
#         lightness = (50 + np.random.rand() * 10)/100.
#         saturation = (90 + np.random.rand() * 10)/100.
#         colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
#     return colors
#
# def createDataset(im):
#     h = im.shape[0]
#     w = im.shape[1]
#     return np.reshape(im,(h*w,3))
#
# def kMeansCluster(features, centers):
#     maxiter =10
#     k = centers.shape[0]
#     n = features.shape[0]
#     idx_last = np.zeros((n))
#     for i in range(maxiter):
#         cluster = {}
#         for center in range(k):
#             cluster[center] = []
#         idx = []
#         for data_point in range(n):
#             min = ((features[data_point] - centers[0]).astype(float)**2).sum()
#             id = 0
#             for j in range(k):
#                 temp = ((features[data_point] - centers[j]).astype(float)**2).sum()
#                 if min > temp:
#                     min = temp
#                     id = j
#             idx.append(id)
#             cluster[id].append(features[data_point])
#         for kk in range(k):
#             new_cluster = np.array(cluster[kk])
#             print(new_cluster.shape)
#             centers[kk] = new_cluster.mean(axis=0)
#         idx = np.array(idx)
#         lost = (idx-idx_last).sum()
#         print(centers)
#         print(lost)
#         if lost==0:
#             return idx, centers
#             break
#     return idx, centers
#
#
# def mapValues(im, idx):
#     res = idx.reshape((im.shape[0],im.shape[1]))
#     cluster = {}
#     res_re = im.copy()
#     for i in range(7):
#         cluster[i]=[]
#     for i in range(im.shape[0]):
#         for j in range(im.shape[1]):
#             cluster[res[i,j]].append(im[i,j])
#     center = np.zeros((7,3))
#     for i in range(7):
#         temp = np.array(cluster[i])
#         center = temp.mean(axis=0)
#     for i in range(im.shape[0]):
#         for j in range(im.shape[1]):
#             res_re[i,j]=center[res[i,j],:]
#     return res_re
#
#
# img = cv2.imread('white-tower.png')
# a = createDataset(img)
# b = kMeansCluster(a,a[[50000,750000,250000,350000,450000,550000,650000],:].copy())



simple_img = np.zeros((11,11))
simple_img[0,0] =1
simple_img[10,0] = 1
simple_img[0,10]=1
simple_img[5,5]=1
simple_img[10,10] = 1

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 将关键点，也就是线上的点放入key_points列表中
key_points = []
for y_cnt in range(0,11):  # 将直线上的点取出（白点即像素值为255的点）
    for x_cnt in range(0, 11):
        if simple_img[y_cnt][x_cnt] == 1:
            key_points.append((x_cnt, y_cnt))
# 将key_points中的点转换到霍夫空间中，间隔的θ为1度，在笛卡尔坐标系中可以描述成在-90°到90°间以点为中心每隔1°画一条直线
conver_points = []
conver_points_re=[]
for key_point in key_points:  # 将点转换到霍夫空间
    conver_points_tmp = []
    x, y = key_point
    conver_points_tmp_re=[]
    for theta in range(-90, 90):  # 从-90°到90°打点，间隔1°
        rho =  x * np.cos(theta / 180 * np.pi) +   y * np.sin(theta / 180 * np.pi)  # 注意将角度转换成弧度
        print(rho)
        rho = rho *5
        conver_points_tmp.append((int(theta), int(rho)))
        conver_points_tmp_re.append((theta,rho))
    conver_points.append(conver_points_tmp)
    conver_points_re.append(conver_points_tmp_re)
# 绘制换换到霍夫空间的曲线
hough = np.zeros([15,180])
hough_img = np.ones([150, 180]).astype(np.uint8)*255  # 转换成uint8的图像，否则imshow无法显示
x =[]
y=[]
print(len(conver_points))
num=0
for conver_point in conver_points:  # 绘制霍夫空间的曲线

    for point in conver_point:

        theta, rho = point
        if num ==3:
            x.append(theta)
            y.append(rho)
        hough_img[rho-75][theta - 90] = 0
    num+=1

cv2.imshow('hough', hough_img)
cv2.waitKey()
print(x)
print(y)
plt.figure('HG')
plt.imshow(hough_img,cmap='Greys')
plt.colorbar()
xtick = np.arange(0,180)
plt.xticks([10,30,50,70,90,110,130,150,170,190],[-80,-60,-40,-20,0,20,40,60,80])
plt.yticks([25,50,75,100,125],[-10,-5,0,5,10])
plt.show()  # 显示绘图


