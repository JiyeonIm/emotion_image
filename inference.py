import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Function, Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
import time
import os
import copy 
import cv2
from PIL import ImageFile, Image
from model import initialize_model
import warnings
warnings.filterwarnings(action='ignore')

class InferenceDataset(Dataset):
    '''
    torch dataset for inference
    '''
    def __init__(self, item):  
        self.item = item 
        input_size = 448
        
        self.transforms = transforms.Compose([
                transforms.Resize((input_size, input_size)), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.transforms_org = transforms.Compose([ 
                transforms.ToTensor()
            ])

    def __len__(self):
        return 1
    
    def __getitem__(self, item):    
        image = self.item.convert("RGB")  
            
        # resize image
        image_org = self.transforms_org(image) 
        image_ppr = self.transforms(image) 
         
        return {  
          'image' : image_org,  
          'image_ppr': image_ppr
        }

def denormalize(sample):
    img = sample.cpu().permute(1,2,0)
    return np.array(img*255).astype(np.uint8)

def get_heatmap(m):
    eps=1e-16
    heatmap = m
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8") 
    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.6,
    colormap=cv2.COLORMAP_VIRIDIS):
    # apply the supplied color map to the heatmap 
    heatmap = cv2.applyColorMap(heatmap, colormap)
    # overlay the heatmap on the input image
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return (heatmap, output)

def infer(input_img, model_ft):
    '''
    Arguments
    - img : 추론할 이미지
    - ground_truth : 추론할 이미지의 정답 감정 카테고리 (있다면)
    '''
    class_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

    # 1) infer용 dataset & dataloader 만들기
    inference_dataset = InferenceDataset(input_img)
    inference_dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, num_workers=0)

    # 2) infer
    softmax = nn.Softmax()
    model_ft.eval()
    with torch.set_grad_enabled(False):
        for batch in tqdm(inference_dataloader):
            org_image = batch['image']
            inputs = batch['image_ppr']

            # 2-1) get sentiment map - 8 maps for each sentiment category, 1 whole sentiment map
            outputs1, outputs2, m, m_c = model_ft(inputs)
            output_prob = softmax(outputs1[0].cpu())
            max_idx = int(torch.argmax(output_prob).item())
            pred_label = class_names[max_idx]

            # 2-2) resize   
            image = denormalize(org_image[0])
            m = cv2.resize(denormalize(m[0]), (org_image.shape[3], org_image.shape[2]))
            m_c = cv2.resize(denormalize(m_c[0]), (org_image.shape[3], org_image.shape[2]))

            # 2-3) overlap (visualization)
            # 전체 heatmap
            heatmap = get_heatmap(m)
            _, total_overlap = overlay_heatmap(heatmap, image) 
            
            # category별 heatmap
            class_map = []
            for c in range(m_c.shape[2]):
                tmp = m_c[:,:,c]
                heatmap = get_heatmap(tmp)
                _, class_overlap = overlay_heatmap(heatmap, image) 
                class_map.append(class_overlap)  
    return total_overlap, class_map, output_prob, pred_label

    
def data(img):
    ######################################
    # configuration
    ###################################### 
    model_name = "resnet"
    num_classes = 8
    num_maps = 4
    batch_size = 10
    feature_extract = False

    class_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

    ## 모델 불러오기
    # define model 
    model_ft, input_size = initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True)

    # load weight
    filename = './checkpoints/wscnet.pt'
    model_ft.load_state_dict(torch.load(filename, map_location=torch.device('cpu'))) 
    model_ft = model_ft.to('cpu')

    total_overlap, class_map, output_prob, pred_label = infer(img, model_ft)

    # 전체적인 heatmap 1개
    fig = plt.figure(figsize=(8, 5))
    plt.imshow(total_overlap)
    plt.text(10, 25, 'Predicted label : %s'%pred_label, color= 'white', weight='bold', size=12) 
    plt.xticks([])              # set no ticks on x-axis
    plt.yticks([])              # set no ticks on y-axis
    fig_html = mpld3.fig_to_html(fig)
    # plt.savefig('./test2.png')  ## !!!! image output path
    plt.close(fig)

    # 8개의 감정 카테고리에 대한 heatmap 8개
    fig_list = plt.figure(figsize=(16, 5))   
    for i in range(8):
        plt.subplot(2,4, i+1)          
        plt.xticks([])              # set no ticks on x-axis
        plt.yticks([])              # set no ticks on y-axis
        plt.imshow(class_map[i])
        plt.text(10, 35, '%s'%class_names[i], color= 'white', weight='bold', size=12)
        plt.text(10, 75, '%.2f'%output_prob[i], color= 'white', weight='bold', size=12) 
    fig_list_html = mpld3.fig_to_html(fig_list)
    # plt.savefig('./test.png')  ## !!!! image output path

    return fig_html, fig_list_html