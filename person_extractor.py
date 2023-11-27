from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
import torch
from torch.utils.data import DataLoader
import PIL
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering

import os
import shutil
from tqdm import tqdm

from dataset.img_list import ImageListDataset
from dataset.utils import make_transform
from net.resnet import Resnet50
from net.torch_reid_models import build_model

def get_feature_extractor_model(model_name):
    if model_name == 'resnet_proxy_anchor':
        model = Resnet50(embedding_size=2048, pretrained=True, is_norm=1, bn_freeze =1)
        model.load_state_dict(torch.load('models/resnet50_2023_11_22_23_54_53_best.pth', map_location=torch.device('cpu'))['model_state_dict'])
    elif model_name == 'resnet_2048':
        model = build_model(
            name="resnet50",
            num_classes=751,
            loss="softmax",
            pretrained=True
        )
        model.load_state_dict(torch.load('../ITCS-5145-CV/learning/deep-person-reid/log/resnet50/model/model.pth.tar-50')['state_dict'])
    elif model_name == 'resnet_512_reid':
        model = torchreid.models.build_model(
            name="resnet50_fc512",
            num_classes=datamanager.num_train_pids,
            loss="softmax",
            pretrained=True
        )
        model.load_state_dict(torch.load('../ITCS-5145-CV/learning/deep-person-reid/log/resnet50')['state_dict'])
    elif model_name == 'osnet':
        model = osnet_ibn_x1_0(pretrained=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def scan_folder(folder_name):
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
    videos = os.listdir(folder_name)
    img_list = []
    index_list = []
    for cam_id, video in enumerate(videos):
        results = model(os.path.join(folder_name, video), stream=True, verbose=False)
        imgs, idxs = get_image_list(cam_id, results)
        img_list.extend(imgs)
        index_list.extend(idxs)

    return img_list, index_list

def extract_features(model_name, img_list):
    dataset = ImageListDataset(img_list, transform = make_transform(is_train = False, is_inception = False))
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size = 16,
        shuffle = False,
        num_workers = 0,
        pin_memory = True
    )

    embeddings = []
    model = get_feature_extractor_model(model_name)
    model.eval()
    iterator = tqdm(enumerate(dl))
    for i, dat in iterator:
        imgs, inds = dat
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        res = model(imgs)
        embeddings.append(res.cpu().detach().numpy())

    embeddings = torch.tensor(np.concatenate(embeddings)).numpy()

    return embeddings

def cluster(embeddings, num_clusters=15):
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', metric='cosine')
    cluster_labels = agg_clustering.fit_predict(embeddings)

    return cluster_labels

def get_image_list(cam_id, results):
    img_list = []
    index_list = []
    for i, r in enumerate(results):
        for class_label, xyxy in zip(r.boxes.cls, r.boxes.xyxy):
            if class_label == 0:
                im_crop = save_one_box(xyxy, r.orig_img, save=False)
                img_list.append(PIL.Image.fromarray(im_crop))
                xyxy_list = xyxy.cpu().numpy()
                index_list.append((cam_id, i, xyxy_list[0], xyxy_list[1], xyxy_list[2], xyxy_list[3]))
    return img_list, index_list

def draw_bbox_and_save(input_folder, bbox_df):
    videos = os.listdir(input_folder)
    print(videos)
    for cam_id, video_file in enumerate(videos):
        output_video_path = f'temp/drawn_vids/{cam_id}.avi'
        cap = cv2.VideoCapture(os.path.join(input_folder, video_file))

        if not cap.isOpened():
            print("Error: Could not open input video.")
            continue

        frame_width = int(cap.get(3))  # Width
        frame_height = int(cap.get(4))  # Height

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
        
        print(f"Writing video for cam: {cam_id}, Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_number > 900:
                break
            frame_data = bbox_df[(bbox_df['frame_number'] == frame_number) & (bbox_df['cam_id'] == cam_id)]
            
            print(f"Writing frame: {frame_number}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}", end='\r')
            
            for index, row in frame_data.iterrows():
                xmin, ymin, xmax, ymax, cluster_number = row['x1'], row['y1'], row['x2'], row['y2'], row['cluster']
                
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person#: {cluster_number}', (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            out.write(frame) 
        
        cap.release()
        out.release()
        print()

def write_people_images(bbox_df, img_list):
    results_folder = f'bbox_clustering'
    if not os.path.exists(os.path.join('temp', results_folder)):
        os.mkdir(os.path.join('temp', results_folder))
        
    for cluster in range(bbox_df['cluster'].max()+1):
        cluster_folder = str(cluster+1)
        if not os.path.exists(os.path.join('temp', results_folder, cluster_folder)):
            os.mkdir(os.path.join('temp', results_folder, cluster_folder))
            
        for i, row in bbox_df[bbox_df['cluster'] == cluster].iterrows():
            cam_id = int(row['cam_id'])
            frame_number = int(row['frame_number'])
            img_list[i].save(os.path.join('temp', results_folder, cluster_folder, f'{cam_id}_{frame_number}.jpg'))

def main(vid_folder, num_clusters):
    try:
        shutil.rmtree('temp/')
    except:
        pass
    os.mkdir('temp/')
    os.mkdir('temp/drawn_vids/')
    img_list, index_list = scan_folder(vid_folder)
    bbox_df = pd.DataFrame(np.array(index_list), columns=['cam_id', 'frame_number', 'x1', 'y1', 'x2', 'y2'])
    embeddings = extract_features('resnet_proxy_anchor', img_list)
    bbox_df['cluster'] = cluster(embeddings, num_clusters=num_clusters)
    draw_bbox_and_save(vid_folder, bbox_df)
    write_people_images(bbox_df, img_list)
    summary_df = bbox_df.groupby(['cluster', 'cam_id'], as_index=False).agg({'frame_number': ('min', 'max')})
    summary_df.columns = ['person_id', 'cam_id', 'first_frame', 'last_frame']

    bbox_df.to_csv('temp/bbox.csv', index=None)
    summary_df.to_csv('temp/summary.csv', index=None)

if __name__ == "__main__":
    main('../../data/reid_custom', 4)