#!/usr/bin/env python3
"""
Large Dataset Loader for UTKFace (>100MB)
Handles datasets that exceed Replit's upload limits
"""

import os
import requests
import zipfile
import streamlit as st
from urllib.parse import urlparse
import tempfile

def download_from_url(url, chunk_size=8192):
    """Download large file from URL with progress tracking"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded / total_size
                        st.progress(progress)
                        st.text(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
            
            return tmp_file.name
            
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

def extract_and_organize(zip_path):
    """Extract zip file and organize images"""
    try:
        os.makedirs('./UTKFace', exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            extracted_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file_info in enumerate(zip_ref.infolist()):
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract file
                    zip_ref.extract(file_info, './temp_extract')
                    
                    # Move to UTKFace directory
                    source_path = f"./temp_extract/{file_info.filename}"
                    if os.path.exists(source_path):
                        filename = os.path.basename(file_info.filename)
                        dest_path = f"./UTKFace/{filename}"
                        
                        # Avoid duplicates
                        if not os.path.exists(dest_path):
                            os.rename(source_path, dest_path)
                            extracted_count += 1
                
                # Update progress
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing: {idx + 1}/{total_files} files | Extracted: {extracted_count} images")
            
            progress_bar.empty()
            status_text.empty()
            
            # Cleanup temp directory
            os.system("rm -rf ./temp_extract")
            
            return extracted_count
            
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        return 0

def get_dataset_from_cloud():
    """Interactive cloud download interface"""
    st.markdown("### 🌐 Download from Cloud Storage")
    
    st.markdown("""
    **Steps to use cloud storage:**
    1. Upload your 134.6MB UTKFace zip to Google Drive, Dropbox, or similar
    2. Get a direct download link
    3. Paste the link below
    """)
    
    # Google Drive instructions
    with st.expander("📱 Google Drive Instructions"):
        st.markdown("""
        1. Upload your UTKFace zip to Google Drive
        2. Right-click → Share → Change to "Anyone with the link"
        3. Copy the sharing link (looks like: https://drive.google.com/file/d/FILE_ID/view)
        4. Convert to direct download: Replace `/view` with `/uc?export=download`
        5. Final link: `https://drive.google.com/uc?export=download&id=FILE_ID`
        """)
    
    # Dropbox instructions  
    with st.expander("📦 Dropbox Instructions"):
        st.markdown("""
        1. Upload your UTKFace zip to Dropbox
        2. Right-click → Share → Create link
        3. Replace `?dl=0` with `?dl=1` at the end of the URL
        4. Example: `https://dropbox.com/s/abc123/utkface.zip?dl=1`
        """)
    
    # URL input
    download_url = st.text_input("📎 Paste direct download URL here:")
    
    if download_url and st.button("🚀 Download and Extract Dataset", type="primary"):
        with st.spinner("Downloading large dataset..."):
            zip_path = download_from_url(download_url)
            
            if zip_path:
                st.success("Download completed! Extracting images...")
                
                with st.spinner("Extracting and organizing images..."):
                    extracted_count = extract_and_organize(zip_path)
                
                # Cleanup
                os.unlink(zip_path)
                
                if extracted_count > 0:
                    st.success(f"🎉 Successfully extracted {extracted_count} images!")
                    st.balloons()
                    return True
                else:
                    st.error("No images found in the archive")
                    return False
            else:
                st.error("Download failed. Please check the URL and try again.")
                return False
    
    return False

def split_dataset_instructions():
    """Instructions for splitting large dataset"""
    st.markdown("### ✂️ Split Dataset Method")
    
    st.markdown("""
    **If cloud storage isn't available, split your dataset:**
    
    1. **On your computer**, divide the 23K+ images into smaller folders:
       - Folder 1: First 5,000 images (~25MB)
       - Folder 2: Next 5,000 images (~25MB)
       - Continue until all images are split
    
    2. **Compress each folder** separately (each will be <100MB)
    
    3. **Upload and extract** each zip file one by one using the buttons below
    """)
    
    # Check for multiple zip files
    zip_files = [f for f in os.listdir('.') if f.lower().endswith('.zip') and os.path.getsize(f) < 100*1024*1024]
    
    if zip_files:
        st.success(f"Found {len(zip_files)} zip files ready for extraction:")
        
        for zip_file in zip_files:
            size_mb = os.path.getsize(zip_file) / (1024*1024)
            st.text(f"📦 {zip_file} ({size_mb:.1f}MB)")
        
        if st.button("📂 Extract All Zip Files", type="primary"):
            total_extracted = 0
            
            for zip_file in zip_files:
                st.info(f"Extracting {zip_file}...")
                extracted = extract_and_organize(zip_file)
                total_extracted += extracted
                
                # Remove processed zip file
                os.unlink(zip_file)
            
            if total_extracted > 0:
                st.success(f"🎉 Total extracted: {total_extracted} images!")
                return True
    
    return False

if __name__ == "__main__":
    print("Large Dataset Loader")
    print("Handles UTKFace datasets >100MB")
    
    # Check current dataset
    current_count = 0
    if os.path.exists('./UTKFace'):
        current_count = len([f for f in os.listdir('./UTKFace') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Current images in UTKFace: {current_count}")
    
    if current_count < 1000:
        print("Small dataset detected. Use Streamlit interface for guided loading.")
    else:
        print(f"Dataset ready with {current_count} images!")