## EEG-Based Emotion Recognition Using Neural Networks
Sarshar Dorosti (ID: s5639776)

### Files
Please download the GitHub files from the link below. Also, put the codes in a file to use the new classification.

#### Article
https://www.sciencedirect.com/science/article/pii/S2352340921010465

#### Github
https://github.com/wavesresearch/eeg_stress_detection

#### Figshare
https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090/1


## Abstract
In this study, the SAM 40 dataset is specially used to train neural network models to identify emotions from EEG data. This dataset records different emotional states experienced during cognitive activities such as mirror image identification, the Stroop test, and arithmetic tests. The dataset is useful for researching stress and cognitive load since these tasks cause varying degrees of cognitive stress. Python is used for the analysis, which focuses on intricate EEG patterns connected to these mental processes. The dataset allows for a variety of study in signal processing and artifact removal because it contains both raw and modified EEG data. This work advances the development of BCI and EEG-based cognitive state analysis. 


## Introduction 
This study merges neuroscience and machine learning to gauge cognitive stress levels using 32-channel EEG data from 40 participants (average age: 21.5). The dataset comprises EEG recordings during stress-inducing tasks (e.g., Stroop test, arithmetic, symmetry recognition, and relaxation phases). Each task lasts 25 seconds, with three trials per task, yielding a comprehensive dataset.
![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/082ea8a3-8c3f-49f9-8acc-18934b5a6a08)


![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/a46e796a-75f4-442b-bc5b-1f12a83bc275)
![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/ec56bf14-a2da-4ac5-b1ef-532541dbdd7a)
![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/aab83c24-2d2f-4543-9cae-ffa42f9d85f2)

Our approach involves EEG data import with `load_dataset` and `load_labels`, efficient handling via the `EEGDataset` class in PyTorch for batch processing during training. We have two model classes: `SimpleNN` (multilayer perceptron with ReLU activation and dropout) for foundational analysis and `EEG_CNN` (CNN model) to extract spatial features from complex EEG data.
Thorough training and validation with `train_one_epoch` and `validate` ensure precision and reliability. We employ `EarlyStopping` to combat overfitting and enhance model generalization. Additionally, we incorporate functionalities for model storage, retrieval, and unit tests for data loading and model initialization reliability.

![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/d886b925-6973-4bb1-80a7-b6755a879fad)

# Literature Review
The topic of emotion recognition utilizing EEG data shows great potential, but it encounters difficulties stemming from the complexity of high-dimensional data and the presence of noise. ICA and wavelet transform are essential techniques for removing artifacts and reducing noise (Makeig, et al., 1996).
EEG analysis has been revolutionized by the use of deep learning techniques, such as Convolutional Neural Networks (CNNs) for extracting features and Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, for capturing temporal dynamics (Lawhern, et al., 2018; Bashivan, et al., 2016).
In their study, Dhake and Angal (2023) conducted a comparison of stress detection approaches based on EEG signals. They discovered that deep learning classifiers exhibited superior performance compared to typical machine learning techniques. With feature extractors like PCA, ICA, and DCT, deep learning outperformed classical techniques by 76% in accuracy (Dhake & Angal, 2023).
The results emphasize the capacity of deep learning to effectively manage the intricate nature of EEG data for the purpose of stress and emotion recognition. Future research endeavors to tackle obstacles such as differences between individuals and disturbances, by including many types of data and investigating unsupervised and semi-supervised learning techniques (Jirayucharoensak, et al., 2014).

# Dataset Description
## Specifications Table
Subject Area: Neuroscience and Psychology.
More Specific Subject Area: Brain-Computer Interface, Experimental and Cognitive Psychology, Neuroimaging.
Type of Data: 32 Channel EEG time-series data.
Data Acquisition: Collected using visual stimuli across diverse cognitive tasks, recorded with a 32-channel Emotiv Epoc Flex gel kit.
Data Format: Both raw and processed EEG time series.
Parameters for Data Collection: EEG signals recorded from 32 channels, sampled at 128 SPS (1024 Hz internal).
Description of Data Collection: Data gathered from 40 subjects during various cognitive tasks, aiming to monitor short-term stress responses.
EEG Data Plotting: A .locs file, Coordinates.locs, facilitates the plotting of EEG data.
Data Source Location: Department of Information Technology, Gauhati University, Guwahati, India.
Data Accessibility: Publicly available on Figshare ([DOI](https://doi.org/10.6084/m9.figshare.14562090.v1)).

![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/f3696b33-7b04-45ba-a414-f73e7824ccac)

### Data Structure
Raw and Filtered Data: The dataset is divided into two main folders: /raw_data and /filtered_data. The /raw_data folder includes EEG time-series segmented according to experimental trials, containing noise and artifacts. The /filtered_data folder, on the other hand, presents clean EEG data, free from artifacts.
Artifacts and Filtering: To cater to different research methodologies, both raw and processed data are provided. This allows researchers to apply various filtering methods for artifact removal.
Subject Feedback: Included is a .xls file named scales.xls, which contains feedback from subjects rating their stress levels on a scale of 1–10 during each task and trial.


![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/c4434579-eb1f-497d-b180-7486ec164801)


### Additional Resources
Artifact Removal Code: The /artifact_removal folder includes Matlab code (correct_EEG.m) for artifact removal, along with two .mat files (Corrupted_EEG.mat and Cleaned_EEG.mat) demonstrating the process on a sample EEG recording.
Data Segmentation: EEG data is segmented according to specific tasks: Stroop color-word test, arithmetic task, mirror image recognition task, and a relaxation state. The data is provided in EEGLAB format and can be visualized through the EEGLAB interface.


![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/640b49de-d826-4e4f-afbf-bf1c4b94dcbc)



# Model Description
### Code Overview
The framework for EEG-Based Emotion Recognition includes several critical components for data processing, handling, and neural network implementation, leveraging the power of PyTorch for efficient computation.

#### Data Loading and Preparation
- **Functions**: `load_dataset`, `load_labels`, and `split_data` form the backbone of data interaction, efficiently loading and processing EEG data for analysis.
- **Data Types and Tests**: Supports various data types (e.g., `ica_filtered`) and test conditions (e.g., `Arithmetic`) with assertive checks for compatibility and validity.
- **Data Structure Adaptation**: Data is reshaped into epochs, transforming the raw EEG signals into a format suitable for neural network training.

#### Data Handling with Custom Class
- **`EEGDataset` Class**: A custom PyTorch `Dataset` class, it encapsulates EEG data and labels, facilitating streamlined batch processing during model training. This class converts data into `FloatTensors` for compatibility with PyTorch's computation requirements.

#### Neural Network Models
1. **`SimpleNN`: A Multilayer Perceptron for EEG Classification**
   - **Architecture**: Consists of multiple linear layers with ReLU activations and dropout for non-linear EEG pattern modeling and overfitting prevention.
   - **Customizability**: Parameters like number of layers and neurons per layer are configurable, allowing for flexible model architecture.

2. **`EEG_CNN`: A Convolutional Neural Network for Advanced EEG Analysis**
   - **Convolutional Layers**: Utilizes multiple convolutional layers with ReLU activation and max pooling, designed to extract spatial features from EEG data. The architecture starts with 64 filters in the first layer and expands to 128 in subsequent layers.
   - **Fully Connected Layers**: Four linear layers with dropout are used to prevent overfitting, each layer gradually reducing the dimensionality towards the output size.
   - **Output Layer**: The final layer employs a sigmoid function, making the model suitable for binary classification tasks.


![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/a6f73c54-151c-449e-9e15-cdcf57a72c21)


## Results and Discussion

### Challenges Encountered
- **Complexity and Noise**: The high dimensionality and inherent noise in EEG data present substantial challenges, particularly for the CNN model.
- **Subject Variability**: The variability in EEG signals across different subjects and sessions posed hurdles for consistent model performance.

### Model Training and Outcome Analysis
- **Data Reshaping**: Input data is specifically reshaped to meet the CNN's requirements, ensuring compatibility with the model's expected input format.
- **CNN Performance**: The CNN model, despite its sophisticated architecture, did not achieve the anticipated accuracy, indicating potential overfitting or the need for further hyperparameter tuning.
- **`SimpleNN` as an Alternative**: The simpler `SimpleNN` model, with its multilayer perceptron structure, might be more suitable for certain datasets due to its straightforward architecture.

### Training, Validation, and Utility Processes
- **Comprehensive Training Procedures**: The training process involves iterating over epochs, with functions `train_one_epoch` for model training and `validate` for performance evaluation.
- **Early Stopping Mechanism**: `EarlyStopping` is employed to halt training when no significant improvement in validation loss is observed, aiding in preventing overfitting.
- **Utilities and Testing**: The codebase includes utility functions for saving and loading models, ensuring reproducibility and ease of model deployment. Unit tests are integrated to ensure the reliability of data loading and model initialization.
![image](https://github.com/sarshardorosti/eeg-stress-classification/assets/50841748/6959ff3b-75fd-4327-bdc3-6c3fb5c7b439)

## Visualization and Analysis

### Extended Data Preprocessing Requirements
- **Additional Time for EEG Data Preprocessing**: A significant amount of additional time was required for preprocessing EEG data for this activity. The preprocessing method implemented in the dataset was distinct from the approach we adopted. This difference necessitated extra effort to align the data processing methods with our model requirements.
- **Future Considerations**: Given that the primary objective of this project was not solely to maximize accuracy, we have deferred further optimization of the preprocessing steps to subsequent phases. This decision allows for a more focused approach on model development and performance evaluation in the current stage.

### Insights from Model Implementation and Comparison
- **Challenges with Random State in Data Selection**: During the implementation and comparative analysis of the models, we encountered an issue where the random state selection for test data was ineffective. This limitation hindered the learning process by restricting it to a single epoch.
- **Impact on Accuracy Calculation**: The aforementioned issue with the random state also affected the accuracy measurement, preventing an accurate assessment of the model's performance. This observation highlights the need for a more robust method of data partitioning to ensure consistent and reliable training and validation processes.

These additional insights into the visualization and analysis phase underscore the complexities and challenges encountered in the EEG data preprocessing and model training stages. The experience gained from these challenges provides valuable lessons for future iterations and enhancements of the project.

# References
Makeig, S., et al. (1996). "Independent component analysis of electroencephalographic data". Advances in neural information processing systems.

Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces". Journal of Neural Engineering.

Bashivan, P., et al. (2016). "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks". arXiv preprint arXiv:1511.06448.

Jirayucharoensak, S., et al. (2014). "EEG-based emotion recognition using Deep Learning Network with principal component based covariate shift adaptation". The Scientific World Journal.

Dhake, D., & Angal, Y. (2023). "A Comparative Analysis of EEG-based Stress Detection Utilizing Machine Learning and Deep Learning Classifiers with a Critical Literature Review". International Journal on Recent and Innovation Trends in Computing and Communication, 11(8s), 61–73. https://doi.org/10.17762/ijritcc.v11i8s.7175


