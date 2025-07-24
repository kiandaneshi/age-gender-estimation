#!/usr/bin/env python3
"""
ResNet50 Age & Gender Estimation - Results Generator
Generates comprehensive results for manuscript/proof of concept paper
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

def load_sample_data(max_samples=500):
    """Load sample UTKFace data for demonstration"""
    print("📚 Loading UTKFace Dataset Sample...")
    
    data_dir = './UTKFace'
    images, ages, genders = [], [], []
    
    # Get image files
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
    print(f"Found {len(image_files)} total image files")
    
    # Process subset for quick demo
    processed = 0
    for img_name in image_files[:max_samples]:
        try:
            base_name = img_name.replace('.chip.jpg', '').replace('.jpg', '')
            parts = base_name.split('_')
            
            if len(parts) >= 3:
                age, gender = int(parts[0]), int(parts[1])
                if 0 <= age <= 116 and gender in [0, 1]:
                    img_path = os.path.join(data_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        images.append(img)
                        ages.append(age)
                        genders.append(gender)
                        processed += 1
        except:
            continue
    
    # Convert to arrays
    images = np.array(images, dtype=np.float32) / 255.0
    ages = np.array(ages)
    genders = to_categorical(genders, 2)
    
    print(f"✅ Processed {processed} images")
    return images, ages, genders

def create_model():
    """Create ResNet50 transfer learning model"""
    print("🧠 Creating ResNet50 Model...")
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    age_output = Dense(1, name='age_output')(x)
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    
    model = Model(inputs=inputs, outputs=[age_output, gender_output])
    model.compile(
        loss={'age_output': 'mse', 'gender_output': 'categorical_crossentropy'},
        optimizer=Adam(learning_rate=0.001),
        metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
    )
    
    return model

def generate_manuscript_results():
    """Generate comprehensive results for manuscript"""
    
    print("🧠 CNN-Based Age & Gender Estimation with ResNet50")
    print("=" * 70)
    
    # Load data
    images, ages, genders = load_sample_data(500)
    
    # Dataset statistics
    print(f"\n📊 DATASET STATISTICS:")
    print(f"Total samples: {len(images)}")
    print(f"Age range: {ages.min():.0f} - {ages.max():.0f} years")
    print(f"Age mean: {ages.mean():.1f} ± {ages.std():.1f} years")
    
    male_count = np.sum(np.argmax(genders, axis=1) == 0)
    female_count = len(genders) - male_count
    print(f"Gender distribution: {male_count} Male ({male_count/len(genders)*100:.1f}%), {female_count} Female ({female_count/len(genders)*100:.1f}%)")
    
    # Age distribution by decade
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_hist = np.histogram(ages, bins=age_bins)[0]
    print(f"\nAge distribution by decade:")
    for i, count in enumerate(age_hist):
        if i < len(age_bins)-1:
            print(f"  {age_bins[i]:2d}-{age_bins[i+1]:2d} years: {count:3d} samples ({count/len(ages)*100:4.1f}%)")
    
    # Train-test split
    X_train, X_test, age_train, age_test, gen_train, gen_test = train_test_split(
        images, ages, genders, test_size=0.2, random_state=42, stratify=np.argmax(genders, axis=1)
    )
    
    print(f"\n🔀 DATA SPLIT:")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(images)*100:.1f}%)")
    print(f"Testing samples: {len(X_test)} ({len(X_test)/len(images)*100:.1f}%)")
    
    # Create model
    model = create_model()
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"Base model: ResNet50 (ImageNet pre-trained)")
    print(f"Input shape: (224, 224, 3)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Outputs: Age (regression) + Gender (binary classification)")
    
    # Quick training (2 epochs for demo)
    print(f"\n🏋️ TRAINING (Quick Demo - 2 epochs):")
    history = model.fit(
        X_train,
        {'age_output': age_train, 'gender_output': gen_train},
        validation_data=(X_test, {'age_output': age_test, 'gender_output': gen_test}),
        epochs=2,
        batch_size=16,
        verbose=1
    )
    
    # Evaluate
    print(f"\n📊 EVALUATION RESULTS:")
    
    # Predictions
    age_pred, gender_pred = model.predict(X_test, verbose=0)
    age_pred = age_pred.flatten()
    gender_pred_classes = np.argmax(gender_pred, axis=1)
    gender_true_classes = np.argmax(gen_test, axis=1)
    
    # Age metrics
    age_mae = mean_absolute_error(age_test, age_pred)
    age_rmse = np.sqrt(np.mean((age_test - age_pred)**2))
    age_mape = np.mean(np.abs((age_test - age_pred) / age_test)) * 100
    
    print(f"\n🎯 AGE PREDICTION PERFORMANCE:")
    print(f"Mean Absolute Error (MAE): {age_mae:.2f} years")
    print(f"Root Mean Square Error (RMSE): {age_rmse:.2f} years")
    print(f"Mean Absolute Percentage Error (MAPE): {age_mape:.2f}%")
    print(f"Prediction bias: {np.mean(age_pred - age_test):.2f} years")
    print(f"Error standard deviation: {np.std(age_test - age_pred):.2f} years")
    
    # Gender metrics
    gender_accuracy = accuracy_score(gender_true_classes, gender_pred_classes)
    cm = confusion_matrix(gender_true_classes, gender_pred_classes)
    
    print(f"\n🎯 GENDER CLASSIFICATION PERFORMANCE:")
    print(f"Overall Accuracy: {gender_accuracy:.4f} ({gender_accuracy*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"          Predicted")
    print(f"Actual    Male  Female")
    print(f"Male      {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Female    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Per-class metrics
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        male_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        female_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        male_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        female_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nPer-class metrics:")
        print(f"Male   - Precision: {male_precision:.4f}, Recall: {male_recall:.4f}")
        print(f"Female - Precision: {female_precision:.4f}, Recall: {female_recall:.4f}")
    
    # Sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    print(f"{'#':>2} {'True Age':>8} {'Pred Age':>8} {'Age Err':>8} {'True Gender':>11} {'Pred Gender':>11} {'Confidence':>10}")
    print("-" * 70)
    
    for i in range(min(15, len(X_test))):
        true_age = age_test[i]
        pred_age = age_pred[i]
        age_error = abs(true_age - pred_age)
        true_gender = 'Male' if gender_true_classes[i] == 0 else 'Female'
        pred_gender = 'Male' if gender_pred_classes[i] == 0 else 'Female'
        confidence = np.max(gender_pred[i])
        
        print(f"{i+1:2d} {true_age:8.0f} {pred_age:8.1f} {age_error:8.1f} {true_gender:>11} {pred_gender:>11} {confidence:10.3f}")
    
    # Training summary
    print(f"\n📈 TRAINING SUMMARY:")
    print(f"Epochs trained: {len(history.history['loss'])}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final age MAE: {history.history['age_output_mae'][-1]:.2f} years")
    print(f"Final gender accuracy: {history.history['gender_output_accuracy'][-1]:.4f}")
    
    # MANUSCRIPT SUMMARY
    print(f"\n" + "="*70)
    print(f"📄 MANUSCRIPT/PROOF OF CONCEPT SUMMARY")
    print(f"="*70)
    print(f"Project: CNN-based Age and Gender Estimation using ResNet50 Transfer Learning")
    print(f"Dataset: UTKFace - {len(images)} processed samples from 23,708+ total images")
    print(f"Architecture: ResNet50 (ImageNet) + Custom regression/classification heads")
    print(f"")
    print(f"KEY RESULTS:")
    print(f"• Age Estimation MAE: {age_mae:.2f} ± {np.std(age_test - age_pred):.2f} years")
    print(f"• Gender Classification Accuracy: {gender_accuracy*100:.2f}%")
    print(f"• Model Parameters: {total_params:,} total ({trainable_params:,} trainable)")
    print(f"• Training Efficiency: {len(history.history['loss'])} epochs on CPU")
    print(f"")
    print(f"TECHNICAL CONTRIBUTIONS:")
    print(f"• Transfer learning with frozen ResNet50 base")
    print(f"• Multi-task learning (age regression + gender classification)")
    print(f"• Real-world dataset evaluation (UTKFace)")
    print(f"• Comprehensive evaluation metrics (MAE, RMSE, Accuracy, Confusion Matrix)")
    print(f"")
    print(f"POTENTIAL APPLICATIONS:")
    print(f"• Demographic analysis systems")
    print(f"• Age verification systems")
    print(f"• Human-computer interaction")
    print(f"• Social media content filtering")
    print(f"• Market research and advertising")
    
    return {
        'dataset_size': len(images),
        'age_mae': age_mae,
        'age_rmse': age_rmse,
        'gender_accuracy': gender_accuracy,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'epochs': len(history.history['loss']),
        'confusion_matrix': cm.tolist()
    }

if __name__ == "__main__":
    results = generate_manuscript_results()
    print(f"\n✅ Results generation completed successfully!")