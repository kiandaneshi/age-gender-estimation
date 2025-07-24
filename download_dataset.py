#!/usr/bin/env python3
"""
Download and extract full UTKFace dataset
This script helps download the complete 23K+ UTKFace dataset
"""

import os
import requests
import zipfile
import tarfile
from urllib.parse import urlparse
import streamlit as st

def download_large_file(url, local_filename):
    """Download large file with progress tracking"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        print(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({progress*100:.1f}%)")
    
    return local_filename

def extract_dataset(archive_path, extract_to="./"):
    """Extract compressed dataset"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"Unsupported archive format: {archive_path}")
        return False
    
    print("Extraction completed!")
    return True

def organize_dataset():
    """Organize extracted files into proper structure"""
    # Look for extracted UTKFace images
    possible_paths = [
        './UTKFace',
        './utkface', 
        './UTK_Face',
        './crop_part1',
        './part1',
        './images'
    ]
    
    found_images = []
    for path in possible_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        found_images.append(os.path.join(root, file))
    
    if not found_images:
        print("No images found in extracted files")
        return False
    
    # Create UTKFace directory
    os.makedirs('./UTKFace', exist_ok=True)
    
    # Move all images to UTKFace directory
    moved_count = 0
    for img_path in found_images:
        img_name = os.path.basename(img_path)
        dest_path = f'./UTKFace/{img_name}'
        
        if not os.path.exists(dest_path):
            os.rename(img_path, dest_path)
            moved_count += 1
    
    print(f"Organized {moved_count} images into ./UTKFace/")
    return moved_count > 0

if __name__ == "__main__":
    print("UTKFace Dataset Downloader")
    print("=" * 40)
    
    # Option 1: Direct download (if you have a direct link)
    dataset_url = input("Enter UTKFace dataset URL (or press Enter to skip): ").strip()
    
    if dataset_url:
        try:
            parsed_url = urlparse(dataset_url)
            filename = os.path.basename(parsed_url.path) or "utkface_dataset.zip"
            
            print(f"Downloading from: {dataset_url}")
            download_large_file(dataset_url, filename)
            
            print("Extracting dataset...")
            extract_dataset(filename)
            
            print("Organizing files...")
            organize_dataset()
            
            # Clean up
            os.remove(filename)
            print("Cleanup completed!")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try manual upload or check the URL")
    
    else:
        print("\nAlternative methods to get full dataset:")
        print("1. Upload a .zip/.tar.gz file containing all UTKFace images")
        print("2. Use cloud storage (Google Drive, Dropbox) and provide link") 
        print("3. Upload in batches of 5K-10K images at a time")
        
        # Check for uploaded archives
        archives = []
        for file in os.listdir('.'):
            if file.lower().endswith(('.zip', '.tar.gz', '.tgz', '.tar', '.rar')):
                archives.append(file)
        
        if archives:
            print(f"\nFound archives: {archives}")
            for archive in archives:
                try:
                    print(f"Extracting {archive}...")
                    extract_dataset(archive)
                    organize_dataset()
                    print(f"Successfully processed {archive}")
                except Exception as e:
                    print(f"Error processing {archive}: {e}")
        
        # Final count
        if os.path.exists('./UTKFace'):
            count = len([f for f in os.listdir('./UTKFace') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"\nFinal result: {count} images in UTKFace directory")
            
            if count > 20000:
                print("✅ Successfully loaded full UTKFace dataset!")
            elif count > 1000:
                print("✅ Loaded substantial dataset - ready for training!")
            else:
                print("⚠️  Small dataset - consider uploading more images")