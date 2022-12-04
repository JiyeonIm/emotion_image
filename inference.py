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
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import os
import copy 
from PIL import ImageFile, Image
import mpld3 
import cv2 

def grad_cam(model, ppr_img, org_img, num_classes):
    heatmap_with_image = []
    all_heatmap = torch.zeros(ppr_img.shape)
    
    model.eval()
    for i in tqdm(range(num_classes)):
        # final conv layer name 
        finalconv_name = 'layer4'

        # activations
        feature_blobs = []

        # gradients
        backward_feature = []

        # output으로 나오는 feature를 feature_blobs에 append하도록
        def hook_feature(module, input, output):
            feature_blobs.append(output.cpu().data.numpy())


        # Grad-CAM
        def backward_hook(module, input, output):
            backward_feature.append(output[0])


        model._modules[finalconv_name].register_forward_hook(hook_feature)
        model._modules[finalconv_name].register_backward_hook(backward_hook) 


        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().detach().numpy()) # [1000, 512]

        # Prediction
        logit = model(ppr_img)
        h_x = F.softmax(logit, dim=1).data.squeeze() # softmax 적용

        probs, idx = h_x.sort(0, True)
#         print("Predicted label : %d, Probability : %.2f" % (idx[0].item(), probs[0].item()))


        # ============================= #
        # ==== Grad-CAM main lines ==== #
        # ============================= #  
        score = logit[:, i].squeeze() # 예측값 y^c
        score.backward(retain_graph = True) # 예측값 y^c에 대해서 backprop 진행
#         print(backward_feature)
        activations = torch.Tensor(feature_blobs[-1]) # (1, 512, 7, 7), forward activations
        gradients = backward_feature[-1] # (1, 512, 7, 7), backward gradients
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2) # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
        weights = alpha.view(b, k, 1, 1) # (1, 512, 1, 1)

        grad_cam_map = (weights*activations).sum(1, keepdim = True) # alpha * A^k = (1, 512, 7, 7) => (1, 1, 7, 7)
        grad_cam_map = F.relu(grad_cam_map) # Apply R e L U
        grad_cam_map = F.interpolate(grad_cam_map, size=(256, 256), mode='bilinear', align_corners=False) # (1, 1, 224, 224)
        map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
        grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, 224, 224), min-max scaling

        # grad_cam_map.squeeze() : (224, 224)
        grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (224, 224, 3), numpy 
        grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, 244, 244)
        b, g, r = grad_heatmap.split(1)
        grad_heatmap = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.

        grad_result = grad_heatmap + org_img.cpu() # (1, 3, 244, 244)
        grad_result = grad_result.div(grad_result.max()).squeeze() # (3, 244, 244)
         
        heatmap_with_image.append(denormalize(grad_result))  
        all_heatmap += grad_heatmap
    
    all_heatmap /= num_classes
    all_heatmap_with_image = all_heatmap + org_img.cpu() # (1, 3, 244, 244)
    all_heatmap_with_image = all_heatmap_with_image.div(all_heatmap_with_image.max()).squeeze() # (3, 244, 244)
    return heatmap_with_image, denormalize(all_heatmap_with_image), h_x

input_size = 256
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def denormalize(sample):
    img = sample.cpu().permute(1,2,0)
    return np.array(img*255).astype(np.uint8)

def infer(img_path, model, num_classes):
    '''
    Arguments
    - img : 추론할 이미지
    - ground_truth : 추론할 이미지의 정답 감정 카테고리 (있다면)
    '''
    class_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
 
    
    # 1) infer용 dataset & dataloader 만들기 
    
    pil_img = Image.open(img_path).convert('RGB')
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    torch_img = F.interpolate(torch_img, size=(256, 256), mode='bilinear', align_corners=False) # (1, 3, 224, 224)
    normed_torch_img = data_transforms['test'](pil_img).unsqueeze(0)    

    
    # 2) infer
    all_results, all_heatmap, output_prob = grad_cam(model, normed_torch_img, torch_img, num_classes)
    pred_label = class_names[torch.argmax(output_prob).item()]
    return all_heatmap, all_results, output_prob, pred_label

    
def data(img):
    ######################################
    # configuration
    ######################################  

    class_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

    ## 모델 불러오기
    # define model 
    device = torch.device("cpu")  

    model = models.resnet18(pretrained = True)
    model.fc = nn.Linear(512, 8) 

    filename = './checkpoints/resnet18.pt'
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu'))) 
    model = model.to(device) 

    total_overlap, class_map, output_prob, pred_label = infer(img, model, len(class_names))

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