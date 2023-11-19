import os
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import cv2
import numpy as np
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib
import pickle


def plot_segmentation(tasks,methods,data, video_path,save_loc):

    # Video loading
    vidcap = cv2.VideoCapture(video_path)

    success = 1
    frames = []
    while success:
        success,image = vidcap.read()
        if success:
            frames.append(image[40:])

    len_data = len(data[0])


    ######### Start Figure Customization  ################
    colors = [(1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5)]
    colors = ['#A8D1D1','#FD8A8A','#F1F7B5','#9EA1D4']

    only_tx_images = False
    num_images = 6 # only for non-tx images
    img_size = 30
    bar_height = 1
    bar_font = 10
    bar_padding = 0.2
    img_padding = 0.2
    fig_width = 7.5
    ######### End Figure Customization  ################

    if not only_tx_images:
        padding = img_padding
        img_size = (len_data-1-(num_images-1)*padding)/float(num_images)

    
    fig_height = ((len(methods)*bar_height+img_size)/len_data)*fig_width

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    plt.axis('off')
    gt_txs = []
    for xx in range(len(methods)):
        for tt in range(len_data):
            if tt==0:
                curr_mode = int(data[xx][0])
                last_tx = 0
            if data[xx][tt] != curr_mode or tt == (len_data - 1):
                ax.broken_barh([(last_tx, tt-last_tx)], (xx*bar_height, bar_height), facecolors=colors[curr_mode])
                curr_mode = int(data[xx][tt])
                last_tx = tt
                if 'gt' in methods[xx] or 'ground' in methods[xx]:
                    gt_txs.append(tt)
        # line under and on top
        ax.plot([0,len_data-1], [xx*bar_height,xx*bar_height], color='black', linewidth=1, markersize=12)
        # label method
        ax.text(bar_padding,bar_height*xx+bar_height/2,methods[xx],fontsize=bar_font,verticalalignment='center')
        # ax.annotate(methods[xx], (10, 10*xx+5),
        #         xytext=(0.8, 0.9), textcoords='axes fraction',
        #         fontsize=12,
        #         horizontalalignment='left', verticalalignment='top')
    ax.plot([0,len_data-1], [len(methods)*bar_height,len(methods)*bar_height], color='black', linewidth=1, markersize=12)
    ax.plot([0,0],[0,len(methods)*bar_height],color='black', linewidth=1)
    ax.plot([len_data-1,len_data-1],[0,len(methods)*bar_height],color='black', linewidth=1)

    ax.set_ylim(0,len(methods)*10+img_size)
    ax.set_xlim(0, len_data)
    ax.grid(False)                                       
    # add the images
    if only_tx_images:
        for tt in gt_txs:
            ax.imshow(frames[tt],extent=(tt-img_size,tt,len(methods)*bar_height,len(methods)*bar_height+img_size))
    else:
        for ii in range(num_images):
            # print(img_size)
            ax.imshow(cv2.cvtColor(frames[int(len_data*ii/num_images)], cv2.COLOR_BGR2RGB),extent=(ii*(img_size+padding),(ii+1)*(img_size+padding)-padding,len(methods)*bar_height,len(methods)*bar_height+img_size))
    ax.set_ylim(0,len(methods)*bar_height+img_size)
    ax.set_xlim(0, len_data)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.tight_layout()
    # plt.show()
    # print(gt_txs)
    fig.savefig(save_loc, dpi=400)

    plt.close('all')

  



if __name__ == "__main__":
    task = 'task a'
    # save_loc = '/home/mike/Documents/generative_learning/felix_iclr_2/plot2.png'
    # video_path = '/home/mike/Downloads/pert_01_400_poo9x9zo_99000_0.mp4'
    # model_pkl_path = '/home/mike/Downloads/pert_01_400_poo9x9zo_99000_0.pkl'
    # ### can
    # root_dir = "../../to_plot_segmentation/can"
    # fname = "pert_01_400_poo9x9zo_99000_0_2"
    # ### lift
    # root_dir = "../../to_plot_segmentation/lift"
    # fname = "pert_01_1000_9o0i55q0_194000_0_2"
    ### square
    root_dir = "../../to_plot_segmentation/square"
    fname = "pert_01_2000_8r52j9pw_199000_0_2"

    video_path = f'{root_dir}/{fname}.mp4'
    model_pkl_path = f'{root_dir}/{fname}.pkl'
    save_root = os.path.join(root_dir, "segmentation_plots")
    # max_length = 100

    os.makedirs(save_root, exist_ok=True)

    methods = ['ours','ground truth'] # names on plot
    data_names = ['prediction','gt'] # names in pkl file

    file_tmp = open(model_pkl_path, 'rb') 
    pkl_data = pickle.load(file_tmp)

    print(pkl_data[0]) # see what is in there

    max_gt_mode = np.max([v['gt'].max() for v in pkl_data])
    max_pred_mode = np.max([v['prediction'].max() for v in pkl_data])
    max_mode = max(max_pred_mode, max_gt_mode - 1) # HACK

    # pkl_data = pkl_data[:max_length]

    for i, pkl_data_i in tqdm.tqdm(enumerate(pkl_data), total=len(pkl_data)):
        save_loc = os.path.join(save_root, f"{i:04d}.png")

        data = []
        for data_tmp in data_names:
            data_i = pkl_data_i[data_tmp]
            if data_tmp == "gt":
                data_i = np.clip(data_i, a_min=-999, a_max=max_mode) # ignore goal state
            data_i = np.concatenate([data_i, [-1]]) # append dummy to enable the last timestep to be visualized 
            data.append(data_i)

        # # FAKE DATA
        # modes_ours = np.zeros((150,))
        # modes_ours[50:100] = 1
        # modes_ours[100:150] = 2
        # modes_gt = np.zeros((150,))
        # modes_gt[40:90] = 1
        # modes_gt[90:150] = 2
        # data = [modes_ours,modes_ours, modes_gt]

        #data['taskid']['method_id] returns len_behavior x 1 containing the mode assignments
        plot_segmentation(task,methods,data,video_path,save_loc)
