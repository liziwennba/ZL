from scipy.ndimage.measurements import label
from skimage import morphology
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorsys

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

# Problem 2
# ii
# 1
img=cv2.imread('lines.jpg',0)
struct=np.ones((10,2))
print(struct.shape)
vertical = morphology.opening(img,struct)
cv2.imshow('ss',vertical)
cv2.waitKey(0)
vertical[vertical<40]=0
cv2.imwrite('vertical_lines.png',vertical)
# ii
# 2
label,features=label(vertical)
label2color_dict = _get_colors(features)
view_label=np.zeros((label.shape[0],label.shape[1],3))
for i in range(view_label.shape[0]):  # i for h
    for j in range(view_label.shape[1]):
        if label[i,j]!=0:
            color = label2color_dict[label[i, j]-1]
            view_label[i, j, 0] = color[0]
            view_label[i, j, 1] = color[1]
            view_label[i, j, 2] = color[2]
cv2.imshow('ss',view_label)
cv2.waitKey(0)
cv2.imwrite('vertical_lines_label.png',view_label)

# ii
# 3
length=list(range(features))
centroid=np.zeros((features,2))
points=[[] for i in range(features)]
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        if label[i,j]!=0:
            points[label[i,j]-1].append([i,j])
for i in range(features):
    arr=np.array(points[i])
    centroid[i]=[np.mean(arr[:,0]),np.mean(arr[:,1])]
    length[i]=np.max(arr[:,0])-np.min(arr[:,0])
print(centroid)
print(length)
data_centroid=[]
print(int(centroid[i][0]))
for i in range(features):
    data_centroid.append('('+ str(int(centroid[i][0])) + ', ' + str(int(centroid[i][1])) + ')')
data={
    'features':features,
    'lenth':length,
    'centroid':data_centroid
}
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(3, 3))

ax.axis('off')
ax.axis('tight')

ax.table(cellText=df.values,
         colLabels=df.columns,
         bbox=[0, 0, 1, 1],
         )

plt.show()


