import streamlit as st
import os
from PIL import Image

from person_extractor import *

def main():
    st.title("Image Viewer App")

    tab_1, tab_2 = st.tabs(["Home", "Analyze"])
    with tab_1:
        st.write("### Scan Tab")
        scan_button = st.button("Scan Images")
        if scan_button:
            scan_images()
    
    with tab_2:
        if os.path.exists('temp'):
            st.write("### View Tab")
            num_people = os.listdir('temp/bbox_clustering')
            folder_number = st.selectbox("Select The person you want to track", list(range(1, len(num_people)+1)))
            display_videos(folder_number-1)
        else:
            st.warning('Run the person extractor module before analyzing.')

def display_videos(person_id):
    bbox_df = pd.read_csv('temp/bbox.csv')
    summary_df = pd.read_csv('temp/summary.csv')
    sub_df = summary_df[summary_df['person_id'] == person_id]

    st_videos = []
    st_buttons = []
    frame_values = []
    for i, cam in sub_df.iterrows():
        person_id, cam_id, first_frame, last_frame = cam.values.astype(int)
        video_file = open(f'temp/drawn_vids/{cam_id}.mp4', 'rb')
        video_bytes = video_file.read()
        st.header(f"Camera: {cam_id}")
        st_videos.append(st.video(video_bytes, start_time=0))
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Enters the camera at {first_frame//20}s")
        with col2:
            st.write(f"Exits the camera at {last_frame//20}s")
    for i, button in enumerate(st_buttons):
        if button[0]:
            print([x for x in st_videos[i]])

    

def scan_images():
    # Function to simulate scanning images
    st.write("Scanning images...")

    # Your scanning logic goes here
    # For example, you can create folders (1 to 10) and save images in them

    st.success("Scan completed!")

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

if __name__ == "__main__":
    main()
