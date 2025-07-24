"""
Grad-CAM implementation for visualizing CNN model attention
Gradient-weighted Class Activation Mapping for age and gender prediction
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import Model

class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM with a trained model
        
        Args:
            model (tensorflow.keras.Model): Trained model
            layer_name (str): Name of the convolutional layer to visualize
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        # Create a model that outputs both predictions and intermediate activations
        self.grad_model = self._create_grad_model()
        
    def _find_target_layer(self):
        """
        Automatically find the best convolutional layer for visualization
        Usually the last convolutional layer before global pooling
        
        Returns:
            str: Name of the target layer
        """
        
        # Look for the last convolutional layer in ResNet50 base
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'activation'):
                return layer.name
        
        # Fallback to a common ResNet50 layer
        return 'conv5_block3_out'
    
    def _create_grad_model(self):
        """
        Create a model that outputs predictions and intermediate activations
        
        Returns:
            tensorflow.keras.Model: Gradient model
        """
        
        try:
            # Get the target layer
            target_layer = self.model.get_layer(self.layer_name)
            
            # Create gradient model
            grad_model = Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
            
            return grad_model
            
        except ValueError as e:
            print(f"Layer '{self.layer_name}' not found. Available layers:")
            for layer in self.model.layers:
                print(f"  - {layer.name}")
            raise e
    
    def generate_heatmap(self, image, output_index, task='age'):
        """
        Generate Grad-CAM heatmap for a specific prediction
        
        Args:
            image (numpy.ndarray): Input image with shape (1, 224, 224, 3)
            output_index (int): Index of the output to analyze (0 for age, 0/1 for gender)
            task (str): 'age' or 'gender' to specify which output to analyze
        
        Returns:
            numpy.ndarray: Heatmap with same spatial dimensions as input
        """
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get model predictions and intermediate activations
            conv_outputs, predictions = self.grad_model(image)
            
            # Select the appropriate output based on task
            if task == 'age':
                # Age output is the first output
                target_output = predictions[0][:, 0]  # Single neuron for age
            elif task == 'gender':
                # Gender output is the second output
                target_output = predictions[1][:, output_index]  # Specific class
            else:
                raise ValueError("Task must be 'age' or 'gender'")
        
        # Calculate gradients of target output with respect to feature maps
        grads = tape.gradient(target_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by corresponding gradients
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Resize heatmap to input image size
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        
        return heatmap
    
    def generate_multiple_heatmaps(self, image):
        """
        Generate heatmaps for both age and gender predictions
        
        Args:
            image (numpy.ndarray): Input image
        
        Returns:
            dict: Dictionary containing heatmaps for different tasks
        """
        
        heatmaps = {}
        
        # Age heatmap
        try:
            heatmaps['age'] = self.generate_heatmap(image, 0, task='age')
        except Exception as e:
            print(f"Error generating age heatmap: {e}")
            heatmaps['age'] = np.zeros((224, 224))
        
        # Gender heatmaps (for both classes)
        try:
            heatmaps['gender_male'] = self.generate_heatmap(image, 0, task='gender')
            heatmaps['gender_female'] = self.generate_heatmap(image, 1, task='gender')
        except Exception as e:
            print(f"Error generating gender heatmaps: {e}")
            heatmaps['gender_male'] = np.zeros((224, 224))
            heatmaps['gender_female'] = np.zeros((224, 224))
        
        return heatmaps

def visualize_gradcam(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        image (numpy.ndarray): Original image (224, 224, 3)
        heatmap (numpy.ndarray): Grad-CAM heatmap (224, 224)
        alpha (float): Transparency factor for overlay
        colormap (int): OpenCV colormap for heatmap visualization
    
    Returns:
        numpy.ndarray: Image with heatmap overlay
    """
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), 
        colormap
    )
    
    # Convert colormap from BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay

def create_gradcam_visualization_grid(image, heatmaps, predictions=None):
    """
    Create a grid visualization showing original image and multiple Grad-CAM heatmaps
    
    Args:
        image (numpy.ndarray): Original image
        heatmaps (dict): Dictionary of heatmaps for different tasks
        predictions (dict): Dictionary of predictions (optional)
    
    Returns:
        matplotlib.figure.Figure: Figure with grid visualization
    """
    
    # Prepare the figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Grad-CAM Visualization Analysis', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Age heatmap
    if 'age' in heatmaps:
        age_overlay = visualize_gradcam(image, heatmaps['age'], colormap=cv2.COLORMAP_JET)
        axes[0, 1].imshow(age_overlay)
        title = 'Age Prediction Focus'
        if predictions and 'age' in predictions:
            title += f"\nPredicted: {predictions['age']:.1f} years"
        axes[0, 1].set_title(title, fontweight='bold')
        axes[0, 1].axis('off')
    
    # Gender (Male) heatmap
    if 'gender_male' in heatmaps:
        male_overlay = visualize_gradcam(image, heatmaps['gender_male'], colormap=cv2.COLORMAP_COOL)
        axes[0, 2].imshow(male_overlay)
        title = 'Male Classification Focus'
        if predictions and 'gender_probs' in predictions:
            title += f"\nMale Prob: {predictions['gender_probs'][0]:.3f}"
        axes[0, 2].set_title(title, fontweight='bold')
        axes[0, 2].axis('off')
    
    # Age heatmap only
    if 'age' in heatmaps:
        im1 = axes[1, 0].imshow(heatmaps['age'], cmap='jet')
        axes[1, 0].set_title('Age Heatmap', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Gender (Female) heatmap
    if 'gender_female' in heatmaps:
        female_overlay = visualize_gradcam(image, heatmaps['gender_female'], colormap=cv2.COLORMAP_HOT)
        axes[1, 1].imshow(female_overlay)
        title = 'Female Classification Focus'
        if predictions and 'gender_probs' in predictions:
            title += f"\nFemale Prob: {predictions['gender_probs'][1]:.3f}"
        axes[1, 1].set_title(title, fontweight='bold')
        axes[1, 1].axis('off')
    
    # Combined heatmap (average of all)
    if len(heatmaps) > 1:
        combined_heatmap = np.mean(list(heatmaps.values()), axis=0)
        combined_overlay = visualize_gradcam(image, combined_heatmap, colormap=cv2.COLORMAP_VIRIDIS)
        axes[1, 2].imshow(combined_overlay)
        axes[1, 2].set_title('Combined Attention', fontweight='bold')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_attention_regions(heatmap, threshold=0.5):
    """
    Analyze which regions of the face receive the most attention
    
    Args:
        heatmap (numpy.ndarray): Grad-CAM heatmap
        threshold (float): Threshold for considering high attention
    
    Returns:
        dict: Analysis results with region importance
    """
    
    h, w = heatmap.shape
    
    # Define facial regions (approximate)
    regions = {
        'forehead': heatmap[0:h//3, w//4:3*w//4],
        'eyes': heatmap[h//4:h//2, w//6:5*w//6],
        'nose': heatmap[h//3:2*h//3, 2*w//5:3*w//5],
        'mouth': heatmap[2*h//3:h, w//4:3*w//4],
        'left_cheek': heatmap[h//3:2*h//3, 0:w//3],
        'right_cheek': heatmap[h//3:2*h//3, 2*w//3:w],
        'chin': heatmap[3*h//4:h, w//3:2*w//3]
    }
    
    analysis = {}
    
    for region_name, region_data in regions.items():
        avg_attention = np.mean(region_data)
        max_attention = np.max(region_data)
        high_attention_ratio = np.sum(region_data > threshold) / region_data.size
        
        analysis[region_name] = {
            'average_attention': avg_attention,
            'max_attention': max_attention,
            'high_attention_ratio': high_attention_ratio,
            'importance_rank': 0  # Will be filled later
        }
    
    # Rank regions by average attention
    sorted_regions = sorted(analysis.items(), key=lambda x: x[1]['average_attention'], reverse=True)
    
    for rank, (region_name, _) in enumerate(sorted_regions, 1):
        analysis[region_name]['importance_rank'] = rank
    
    return analysis

def create_attention_analysis_report(image, heatmaps, predictions=None):
    """
    Create a comprehensive analysis report of model attention
    
    Args:
        image (numpy.ndarray): Original image
        heatmaps (dict): Dictionary of Grad-CAM heatmaps
        predictions (dict): Model predictions (optional)
    
    Returns:
        dict: Comprehensive analysis report
    """
    
    report = {
        'image_shape': image.shape,
        'predictions': predictions or {},
        'attention_analysis': {}
    }
    
    for task, heatmap in heatmaps.items():
        task_analysis = analyze_attention_regions(heatmap)
        
        # Add summary statistics
        task_analysis['summary'] = {
            'overall_attention_strength': np.mean(heatmap),
            'attention_concentration': np.std(heatmap),
            'max_attention_value': np.max(heatmap),
            'high_attention_percentage': np.sum(heatmap > 0.5) / heatmap.size * 100
        }
        
        # Find the most important region
        most_important = min(task_analysis.items(), 
                           key=lambda x: x[1]['importance_rank'] if isinstance(x[1], dict) else float('inf'))
        
        if isinstance(most_important[1], dict):
            task_analysis['most_important_region'] = most_important[0]
        
        report['attention_analysis'][task] = task_analysis
    
    return report

# Utility functions for batch processing
def process_batch_gradcam(model, images, max_samples=10):
    """
    Process multiple images with Grad-CAM analysis
    
    Args:
        model: Trained model
        images (numpy.ndarray): Batch of images
        max_samples (int): Maximum number of samples to process
    
    Returns:
        list: List of analysis results for each image
    """
    
    gradcam = GradCAM(model)
    results = []
    
    n_samples = min(len(images), max_samples)
    sample_indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx in sample_indices:
        image = images[idx]
        
        try:
            # Generate heatmaps
            heatmaps = gradcam.generate_multiple_heatmaps(image)
            
            # Create analysis report
            report = create_attention_analysis_report(image, heatmaps)
            
            results.append({
                'index': idx,
                'image': image,
                'heatmaps': heatmaps,
                'analysis': report
            })
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    return results
