import streamlit as st
import os
import numpy as np
from moviepy import VideoFileClip
import chromadb
import uuid
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --- 1. THE REPLACEABLE MODEL ARCHITECTURE ---
class BaseAIModel:
    def get_video_embedding(self, video_path):
        raise NotImplementedError

    def get_text_embedding(self, text):
        raise NotImplementedError

class DummyModel(BaseAIModel):
    """A fake model for fast plumbing tests."""
    def get_video_embedding(self, video_path):
        return np.random.rand(512).tolist()

    def get_text_embedding(self, text):
        return np.random.rand(512).tolist()

class RealCLIPModel(BaseAIModel):
    """The real AI engine using OpenAI's CLIP model (CLIP4Clip Baseline)."""
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)

    def get_video_embedding(self, video_path):
        clip = VideoFileClip(video_path)
        frames = []
        
        # Extract 1 frame for every single second of the clip
        for t in range(int(clip.duration)):
            frame = clip.get_frame(t)
            frames.append(Image.fromarray(frame))
        clip.close()
        
        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
        
        video_embedding = image_features.mean(dim=0)
        video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
        return video_embedding.tolist()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()

class CLIP4ClipModel(BaseAIModel):
    """A fine-tuned model specifically trained for video retrieval."""
    def __init__(self):
        # This points to the fine-tuned video model on Hugging Face
        self.model_id = "Searchium-ai/clip4clip-webvid150k"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)

    def get_video_embedding(self, video_path):
        clip = VideoFileClip(video_path)
        frames = []
        
        # 1 frame per second for high accuracy
        for t in range(int(clip.duration)):
            frame = clip.get_frame(t)
            frames.append(Image.fromarray(frame))
        clip.close()
        
        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
        
        video_embedding = image_features.mean(dim=0)
        video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
        return video_embedding.tolist()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()

    
# --- 2. THE DATABASE SETUP ---
client = chromadb.Client()
# We use get_or_create so Streamlit doesn't give our database amnesia!
collection = client.get_or_create_collection("clipflow_db")

# --- 3. VIDEO PROCESSING ENGINE ---
def chop_video(video_path, chunk_length=5):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    chunks = []
    
    if not os.path.exists("chunks"):
        os.makedirs("chunks")

    for i in range(0, int(duration), chunk_length):
        start = i
        end = min(i + chunk_length, duration)
        chunk_path = f"chunks/clip_{start}_{end}.mp4"
        
        subclip = clip.subclipped(start, end)
        subclip.write_videofile(chunk_path, codec="libx264", audio_codec="aac", logger=None)
        chunks.append(chunk_path)
        
    return chunks

# --- 4. FRONT END (STREAMLIT) ---
st.title("🌊 ClipFlow: AI Video Search")
st.write("Upload a video, and we will chop it up, embed it, and make it searchable!")

# >>> THE UI SWITCHER <<<
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox(
    "Choose AI Engine", 
    ["Original CLIP (Baseline)", "Searchium CLIP4Clip (High Accuracy)", "Dummy Model (Fast Testing)"]
)

# Cache both models separately so switching is instant after the first load
@st.cache_resource
def load_baseline():
    return RealCLIPModel()

@st.cache_resource
def load_clip4clip():
    return CLIP4ClipModel()

# Route traffic to the selected model
if selected_model == "Original CLIP (Baseline)":
    model = load_baseline()
elif selected_model == "Searchium CLIP4Clip (High Accuracy)":
    model = load_clip4clip()
else:
    model = DummyModel()
    
uploaded_file = st.file_uploader("Upload a Video (mp4)", type=["mp4"])

if uploaded_file is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(temp_video_path)
    
    if st.button("Process Video"):
        # We ONLY wipe the database when you process a brand new video
        try:
            client.delete_collection("clipflow_db")
        except:
            pass
        collection = client.create_collection("clipflow_db")

        with st.spinner("Chopping video into 5-second clips..."):
            chunk_paths = chop_video(temp_video_path)
            st.success(f"Successfully created {len(chunk_paths)} short clips!")
            
        with st.spinner("Running AI Model (This will take a moment...)"):
            for chunk in chunk_paths:
                embedding = model.get_video_embedding(chunk)
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk], 
                    ids=[str(uuid.uuid4())]
                )
            st.success("All clips embedded and stored securely!")

    st.markdown("---")
    st.subheader("🔍 Search Your Video")
    search_query = st.text_input("What action are you looking for?")
    
    # Search Button
    if st.button("Search"):
        if search_query:
            # Safety check to ensure the database actually has clips in it
            if collection.count() == 0:
                st.warning("The database is empty! Please click 'Process Video' first.")
            else:
                with st.spinner(f"Searching for the best 5 matches..."):
                    query_embedding = model.get_text_embedding(search_query)
                    
                    # Update: Ask ChromaDB for the top 5 results and include their mathematical distance
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        include=["documents", "distances"]
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        st.subheader(f"🎯 Top 5 Matches for: '{search_query}'")
                        
                        # Extract the lists of video files and their scores
                        docs = results['documents'][0]
                        distances = results['distances'][0]
                        
                        # Loop through all 5 results and display them
                        for i, (doc, dist) in enumerate(zip(docs, distances)):
                            # Convert Chroma's raw distance metric to a clean 0-100% similarity score
                            similarity_score = (1 - (dist / 2)) * 100
                            
                            st.write(f"**Match #{i+1}** | AI Confidence Score: `{similarity_score:.2f}%`")
                            st.video(doc)
                            st.divider() # Adds a clean visual line between videos
                    else:
                        st.warning("Could not find a match.")