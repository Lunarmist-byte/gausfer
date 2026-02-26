import streamlit as st
import os
import cv2
import tempfile

#Config
SAVE_DIR="./images"
os.makedirs(SAVE_DIR,exist_ok=True)

def is_blurry(image,threshold=100.0):
    '''
    Uses Laplacian Variance to calculate image sharpness
    If the variance is below threshold, its blurry
    '''
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    variance=cv2.Laplacian(gray,cv2.CV_64F).var()
    return variance<threshold
def extract_frames(video_path,extract_fps=2,blur_threshold=100.0):
    '''
        Automatic Extractor of frames from video
    '''
    cap=cv2.VideoCapture(video_path)
    video_fps=cap.get(cv2.CAP_PROP_FPS)
    #how many frames to skip to get target
    frame_interval=int(video_fps/extract_fps)
    count=0
    saved=0
    progress_bar=st.progress(0.0)
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        #process frames at requested interval
        if count%frame_interval==0:
            #throw away blurry frames
            if not is_blurry(frame,blur_threshold):
                filepath=os.path.join(SAVE_DIR,f"auto_frame_{saved:04d}.jpg")
                cv2.imwrite(filepath,frame)
                saved+=1
        count+=1
        progress_bar.progress(min(count/total_frames,1.0))
    cap.release()
    return saved

#UI for st
st.set_page_config(page_title="Automated Video Capture for gausfer")
st.title("Pipeline Feeder for dataset loader")
st.write("Upload a video walking around a room. The system will automatically extract the sharpest frames")

st.metric(label="Total Ready Images",value=len(os.listdir(SAVE_DIR)))
st.divider()
#Automated processor
uploaded_video=st.file_uploader("Upload Room Walkthrough Video(MP4/MOV)",type=['mp4','mov'])
col1,col2=st.columns(2)
with col1:
    fps_target=st.slider("Target Frames per second",min_value=1,max_value=5,value=2,help="How many images to extract per second of video.2 is usually perfect")
with col2:
    blur_strictness=st.slider("Blur Filter Strictness",min_value=50.0,max_value=300.0,value=100.0,help="Higher=Strict sharp images. Lower=allows some motion blur")
if uploaded_video:
    if st.button("Start Extraction",type="primary"):
        st.info("Analyzing video geometry and filtering motion blur")
        #Save as temporary 
        tfile=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
        tfile.write(uploaded_video.read())
        #Run automatic extraction
        frames_saved=extract_frames(tfile.name,extract_fps=fps_target,blur_threshold=blur_strictness)

        st.success("Automatically extracted {frames_saved} sharp frames")
        st.ballons()

