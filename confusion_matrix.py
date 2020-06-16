import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def swap(arr):
    snap = arr[0]
    arr[0] = arr[2]
    arr[2] = snap
    return arr

def normalize(arr):
    sum = 0
    for value in arr:
        sum += value
    result = np.true_divide(arr, sum)
    return result


arr1 = [[13, 1, 1],
          [3, 9, 6],
          [0, 0, 16]]
arr2 = [[13, 1, 1],
          [3, 9, 6],
          [0, 0, 16]]
arr3 = [[13, 1, 1],
          [3, 9, 6],
          [0, 0, 16]]
arr4 = [[13, 1, 1],
          [3, 9, 6],
          [0, 0, 16]]
####################33
arr_norm = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

arr_list = [arr1, arr2, arr3, arr4]
arr_norm_list = [arr_norm,arr_norm,arr_norm,arr_norm]
title_list = ['a','b','c','d']
title_size = 36



for array, norm in zip(arr_list,arr_norm_list):
    for i in range(3):
        norm[i] = normalize(array[i])
column = ['0', '1', '2']
row = ['2', '1', '0']

fig, (ax1,ax2,ax3,ax4,ax_tick) = plt.subplots(1, 5, figsize=(26.4, 6.2),gridspec_kw={'width_ratios':[1,1,1,1,0.08]})
fig.subplots_adjust(wspace = 2, hspace = 2)

df_cm1 = pd.DataFrame(arr_norm_list[0], index = row, columns = column)
sns.set(font_scale = 2.5)
g1 = sns.heatmap(df_cm1,cmap="YlOrRd",cbar=False,ax=ax1,annot=arr_list[0], annot_kws={"size": 30})
ax1.set_title(title_list[0],size = title_size)
g1.set_ylabel('True label', size = 30)
g1.set_xlabel('Predicted label', size = 30)

df_cm2 = pd.DataFrame(arr_norm_list[1], index = row, columns = column)
g2 = sns.heatmap(df_cm2,cmap="YlOrRd",cbar=False,ax=ax2,annot=arr_list[1], annot_kws={"size": 30},yticklabels=False)
ax2.set_title(title_list[1],size = title_size)
g2.set_ylabel('', size = 30)
g2.set_xlabel('Predicted label', size = 30)

df_cm3 = pd.DataFrame(arr_norm_list[2], index = row, columns = column)
g3 = sns.heatmap(df_cm3,cmap="YlOrRd",cbar=False, ax=ax3,annot=arr_list[2], annot_kws={"size": 30},yticklabels=False)
ax3.set_title(title_list[2],size = title_size)
g3.set_ylabel('', size = 30)
g3.set_xlabel('Predicted label', size = 30)

df_cm4 = pd.DataFrame(arr_norm_list[3], index = row, columns = column)
g4 = sns.heatmap(df_cm4,cmap="YlOrRd",ax=ax4, cbar_ax=ax_tick,annot=arr_list[3], annot_kws={"size": 30}, cbar_kws = {'ticks':[0,1],'shrink' : 0.95},yticklabels=False)
ax4.set_title(title_list[3],size = title_size)
g4.set_ylabel('', size = 30)
g4.set_xlabel('Predicted label', size = 30)

for ax in [g1,g2,g3,g4]:
    ax.set_xticklabels([0,1,2], size = 30)
    ax.set_yticklabels([None,None,None])
ax1.set_yticklabels([0,1,2], size = 30, rotation = 0)
ax_tick.yaxis.label.set_size(30)
ax_tick.tick_params(labelsize = 30)
#ax1.get_shared_y_axes().join(ax2,ax3,ax4)
plt.show()
plt.savefig('image/results.png',dpi = 300)

'''

fig, axes = plt.subplots(1, 4, figsize=(24.8, 6.2))
cbar_ax = fig.add_axes([.91, .3, .02, .4])

for i,(ax, arr,title) in enumerate(zip(axes,arr_list,title_list)):
    df_cm = pd.DataFrame(arr, column, row)
    sn.heatmap(df_cm,annot=True, annot_kws={"size": 20}, cmap='YlOrRd', cbar = i == 0,vmin = 0,vmax =1, ax=ax, cbar_ax = None if i else cbar_ax)
    ax.set_title(title, fontsize=36)
    ax.set_xlabel('abcabaadsfadsjfkjas;ldf', fontsize=18)
    ax.set_ylabel('adsfasdfasdfas', fontsize=18)
fig.tight_layout(rect=[0, 0, .9, 1])
//////////////////////////////////////////////////////////
ax.set_axis_off()
im = ax.imshow(arr, cmap='YlOrRd', vmin=0, vmax = 1)  # font size
ax.set_title(title, fontsize=36)
ax.set_xlabel('abcabaadsfadsjfkjas;ldf', fontsize=18)
ax.set_ylabel('adsfasdfasdfas', fontsize=18)

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.01, right=0.02,
                    wspace=1, hspace=0.02)
# set subplot title
cb_ax = fig.add_axes([0.99, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks(np.arange(0, 1.1, 0.5))
cbar.set_ticklabels(['low', 'medium', 'high'],position = (11,5))
fig.set_tight_layout(False)
'''