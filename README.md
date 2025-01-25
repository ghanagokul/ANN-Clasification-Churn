# ANN Classification for Churn Prediction

This repository contains the implementation of an Artificial Neural Network (ANN) model to predict customer churn. The model classifies whether a customer is likely to churn or remain based on their attributes and behaviors.

## Features
- **Deep Learning Architecture**: Implements a fully connected ANN for binary classification.
- **Dataset**: Processes a customer dataset with features like demographics, account information, and behavior metrics.
- **Preprocessing**: Handles data cleaning, normalization, and encoding for optimal model performance.
- **Training**: Trains the ANN model using optimized hyperparameters for improved accuracy and reduced loss.

## Model Architecture
The ANN model includes:
1. **Input Layer**: Accepts preprocessed feature vectors.
2. **Hidden Layers**: Fully connected dense layers with ReLU activation.
3. **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

## Requirements
To run this project, the following dependencies are required:
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ghanagokul/ANN-Clasification-Churn.git
   cd ANN-Clasification-Churn
   ```

2. **Run the Training Script**:
   Execute the Python script to train the model:
   ```bash
   python train_ann.py
   ```

3. **Evaluate the Model**:
   The script will output metrics such as accuracy, precision, recall, and the confusion matrix for model evaluation.

## Dataset
The dataset includes customer data with features such as:
- Demographics: Age, Gender, etc.
- Account Details: Tenure, Balance, etc.
- Behavioral Metrics: Transaction frequency, product usage, etc.

For more details about the dataset, refer to the source or accompanying documentation.

## Results
- **Accuracy**: Achieves high accuracy in predicting customer churn.
- **Loss**: Optimized during training and validation for better predictions.
- **Insights**: Generates actionable insights into the factors influencing churn.

## Deployment
- The trained model can be exported and deployed for real-time churn predictions.
- Integration with applications like customer retention dashboards is supported.

## Future Enhancements
- Experiment with different activation functions and architectures for improved performance.
- Add feature importance analysis to identify key churn indicators.
- Explore deployment on cloud platforms like AWS or GCP for scalability.

## Repository Structure
```plaintext
ANN-Clasification-Churn/
│
├── train_ann.py          # Script to train the ANN model
├── preprocess.py         # Data preprocessing utilities
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── dataset/              # Dataset folder (if applicable)
```

## Author
**Ghana Gokul**  
- [LinkedIn](https://linkedin.com/in/ghanagokul/)  
- [GitHub](https://github.com/ghanagokul)  

If you have any questions or suggestions, feel free to reach out or open an issue in this repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
