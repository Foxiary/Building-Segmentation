**Project: Building Segmentation from Satellite Images using U-Net**  

- **Objective**: Developed a deep learning model to automatically segment buildings from satellite images, aiding urban planning and management.  
- **Approach**:  
  - Utilized the U-Net architecture with encoder-decoder structure and skip connections for precise segmentation.  
  - Trained the model on a dataset from Kaggle, applying data augmentation (rotation, brightness adjustment, etc.) to enhance robustness.  
  - Optimized using Binary Cross-Entropy and Dice loss functions, achieving high accuracy (IoU ≈ 0.89, Dice ≈ 0.94).  
- **Results**:  
  - Model demonstrated strong performance in distinguishing buildings from complex backgrounds, including noise from trees and roads.  
  - Outperformed pretrained models (ResNet50, VGG16) and variants with modified blocks (LSTM, attention mechanisms).  
- **Tools & Technologies**: Python, TensorFlow/Keras, OpenCV, Kaggle (NVIDIA Tesla P100 GPU).  
- **Impact**: Potential applications in urban development, reducing manual effort in map updating and city management.  
