"""
Model utilities for ResNet50-based age and gender estimation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os

def create_resnet50_model(freeze_base=True, learning_rate=0.0001):
    """
    Create a ResNet50-based model for age and gender estimation
    
    Args:
        freeze_base (bool): Whether to freeze the ResNet50 base layers
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        tensorflow.keras.Model: Compiled model with dual outputs
    """
    
    # Input layer for 224x224 RGB images (ResNet50 requirement)
    input_layer = Input(shape=(224, 224, 3), name='input_layer')
    
    # Load ResNet50 base model pre-trained on ImageNet
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,  # Exclude the final classification layer
        input_tensor=input_layer
    )
    
    # Freeze base model layers if specified
    if freeze_base:
        base_model.trainable = False
        print("ResNet50 base layers frozen for transfer learning")
    else:
        print("ResNet50 base layers unfrozen for fine-tuning")
    
    # Add custom layers on top of ResNet50
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(128, activation='relu', name='dense_features')(x)
    x = Dropout(0.5, name='dropout')(x)
    
    # Two separate output branches
    # Age regression output
    age_output = Dense(1, name='age_output', activation='linear')(x)
    
    # Gender classification output  
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    
    # Create the complete model
    model = Model(inputs=input_layer, outputs=[age_output, gender_output], name='resnet50_age_gender')
    
    # Compile model with appropriate losses and metrics
    model.compile(
        loss={
            'age_output': 'mse',  # Mean Squared Error for age regression
            'gender_output': 'categorical_crossentropy'  # Cross-entropy for gender classification
        },
        optimizer=Adam(learning_rate=learning_rate),
        metrics={
            'age_output': ['mae'],  # Mean Absolute Error for age
            'gender_output': ['accuracy']  # Accuracy for gender
        },
        loss_weights={
            'age_output': 1.0,
            'gender_output': 1.0
        }
    )
    
    print(f"Model compiled with learning rate: {learning_rate}")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

def load_and_preprocess_data(data_dir, target_size=(224, 224)):
    """
    Load and preprocess UTKFace dataset for ResNet50
    
    Args:
        data_dir (str): Path to the UTKFace dataset directory
        target_size (tuple): Target image size (height, width)
    
    Returns:
        tuple: (images, ages, genders) as numpy arrays
    """
    
    if not os.path.exists(data_dir):
        print(f"Warning: Dataset directory {data_dir} not found")
        return None, None, None
    
    images, ages, genders = [], [], []
    
    print(f"Loading images from {data_dir}...")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    processed_count = 0
    
    for img_name in os.listdir(data_dir):
        if not img_name.lower().endswith(valid_extensions):
            continue
            
        try:
            # Parse filename: age_gender_race_date&time.jpg
            parts = img_name.split('_')
            if len(parts) < 3:
                continue
                
            age = int(parts[0])
            gender = int(parts[1])
            
            # Skip invalid entries
            if age < 0 or age > 120 or gender not in [0, 1]:
                continue
            
            # Load and preprocess image
            img_path = os.path.join(data_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size for ResNet50
            img = cv2.resize(img, target_size)
            
            # Convert to array and normalize to [0, 1]
            img = img_to_array(img) / 255.0
            
            images.append(img)
            ages.append(age)
            genders.append(gender)
            
            processed_count += 1
            
            # Progress indicator
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} images...")
                
        except (ValueError, IndexError, cv2.error) as e:
            # Skip problematic files
            continue
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    ages = np.array(ages, dtype=np.int32)
    genders = np.array(genders, dtype=np.int32)
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Age range: {ages.min()} - {ages.max()}")
    print(f"Gender distribution: Male={np.sum(genders == 0)}, Female={np.sum(genders == 1)}")
    
    return images, ages, genders

def preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for model prediction
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size (height, width)
    
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img_to_array(img) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def resize_images_for_resnet50(images, target_size=(224, 224)):
    """
    Resize existing images from 64x64 to 224x224 for ResNet50 compatibility
    
    Args:
        images (numpy.ndarray): Array of images with shape (n, 64, 64, 3)
        target_size (tuple): Target size (height, width)
    
    Returns:
        numpy.ndarray: Resized images with shape (n, 224, 224, 3)
    """
    
    print(f"Resizing {len(images)} images from {images.shape[1:3]} to {target_size}")
    
    resized_images = []
    
    for i, img in enumerate(images):
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        # Resize image
        resized = cv2.resize(img_uint8, target_size)
        
        # Convert back to float32 and normalize
        resized = resized.astype(np.float32) / 255.0
        
        resized_images.append(resized)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Resized {i + 1}/{len(images)} images")
    
    resized_images = np.array(resized_images, dtype=np.float32)
    
    print(f"Resizing completed. New shape: {resized_images.shape}")
    
    return resized_images

def get_model_summary(model):
    """
    Get a formatted model summary as string
    
    Args:
        model (tensorflow.keras.Model): The model to summarize
    
    Returns:
        str: Formatted model summary
    """
    
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)

def calculate_model_size(model):
    """
    Calculate the size of the model in MB
    
    Args:
        model (tensorflow.keras.Model): The model to analyze
    
    Returns:
        float: Model size in MB
    """
    
    param_count = model.count_params()
    # Assume 4 bytes per parameter (float32)
    size_mb = (param_count * 4) / (1024 * 1024)
    
    return size_mb
