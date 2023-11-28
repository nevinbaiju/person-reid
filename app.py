import streamlit as st
import os
from PIL import Image
import subprocess

from person_extractor import *

def main():
    st.title("Image Viewer App")

    tab_1, tab_2 = st.tabs(["Home", "Analyze"])
    with tab_1:
        st.write("### Scan Tab")
        scan_button = st.button("Scan Images")
        if scan_button:
            progbar = st.progress(0, text="")
            scan_images(progbar)
        if os.path.exists('temp/bbox_clustering'):
            display_tiles()
    
    with tab_2:
        try:
            with open('complete', 'r') as file:
                status = file.read()
        except:
            status = 'no'
        if status == 'yes':
            st.write("### View Tab")
            num_people = os.listdir('temp/bbox_clustering')
            folder_number = st.selectbox("Select The person you want to track", list(range(1, len(num_people)+1)))
            display_videos(folder_number-1)
        else:
            st.warning('Run the person extractor module before analyzing.')

def display_tiles():
    bbox_folder='temp/bbox_clustering'
    people_folders = sorted(os.listdir(bbox_folder))
    for person in people_folders:
        person_total = sorted(os.listdir(os.path.join(bbox_folder, person)))
        person_samples = person_total[:2] + person_total[-2:]
        cams = ", ".join(sorted(list(set([file.split('_')[0] for file in person_total]))))
        st.header(f"Identified Person {person} in cams: {cams}")
        cols = st.columns(4)
        for i, sample in enumerate(person_samples):
            with cols[i]:
                st.write('Cam '+sample.split('_')[0])
                st.image(Image.open(os.path.join(bbox_folder, person, sample)), use_column_width=True)


def display_videos(person_id):
    bbox_df = pd.read_csv('temp/bbox.csv')
    summary_df = pd.read_csv('temp/summary.csv')
    sub_df = summary_df[summary_df['person_id'] == person_id]

    st_videos = []
    st_buttons = []
    frame_values = []
    for i, cam in sub_df.iterrows():
        person_id, cam_id, first_frame, last_frame = cam.values.astype(int)
        video_file = open(f'temp/processed_vids/{cam_id}.mp4', 'rb')
        video_bytes = video_file.read()
        st.header(f"Camera: {cam_id}")
        st_videos.append(st.video(video_bytes, start_time=first_frame//20))
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Enters the camera at {first_frame//20}s")
        with col2:
            st.write(f"Exits the camera at {last_frame//20}s")
    for i, button in enumerate(st_buttons):
        if button[0]:
            print([x for x in st_videos[i]])  

def scan_images(progbar, vid_folder='./data/', num_clusters=4):
    delete_files_in_directory('temp/')
    os.mkdir('temp/drawn_vids/')
    progbar.progress(5, text="Created folders")
    img_list, index_list = scan_folder(vid_folder)
    progbar.progress(20, text="Scanned CCTV videos")
    bbox_df = pd.DataFrame(np.array(index_list), columns=['cam_id', 'frame_number', 'x1', 'y1', 'x2', 'y2'])
    embeddings = extract_features('resnet_proxy_anchor', img_list)
    progbar.progress(50, text="Extracted features from people")
    bbox_df['cluster'] = cluster(embeddings, num_clusters=num_clusters)
    draw_bbox_and_save(vid_folder, bbox_df)
    progbar.progress(80, text="Processing videos with bounding boxes")
    write_people_images(bbox_df, img_list)
    summary_df = bbox_df.groupby(['cluster', 'cam_id'], as_index=False).agg({'frame_number': ('min', 'max')})
    summary_df.columns = ['person_id', 'cam_id', 'first_frame', 'last_frame']
    progbar.progress(100, text="Finished analyzing")

    bbox_df.to_csv('temp/bbox.csv', index=None)
    summary_df.to_csv('temp/summary.csv', index=None) 
    st.success("Scan completed!")

    subprocess.call(['./convert_vids.sh'])

def display_images(folder_number):
    st.write(f"Displaying images from folder: {folder_number}")

    folder_path = f"temp/bbox_clustering/{folder_number}"
    
    cols = st.columns(8)
    if os.path.exists(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        if image_files:
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path)
                with cols[i%8]:
                    st.image(image, caption=image_file, width=80)
        else:
            st.warning("No images found in the selected folder.")
    else:
        st.warning("Selected folder does not exist.")

def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            try:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"{e}: {file_path}")
        with open('complete', 'w') as file:
            file.write('')
        print("All files deleted successfully.")
    except OSError as e:
        print("Error occurred while deleting files.", e)

if __name__ == "__main__":
    main()
