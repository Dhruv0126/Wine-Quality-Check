# Wine Quality Checking with Neural Network

This project demonstrates how to use a simple neural network to classify wine quality based on various chemical properties using Keras and TensorFlow in Python.

## Dataset
The dataset (`wines.csv`) contains 178 samples of wines with the following features:
- alcohol
- malic_acid
- ash
- alcalinity_of_ash
- magnesium
- total_phenols
- flavanoids
- nonflavanoid_phenols
- proanthocyanins
- color_intensity
- hue
- OD280/OD315
- proline
- class (target: 1, 2, or 3)

Some values may be missing (NaN) and should be handled appropriately for production use.

## Model Structure
The neural network is built using Keras Sequential API with the following layers:
- Input layer: 13 features
- Dense layer: 4 units, ReLU activation
- Dense layer: 5 units, ReLU activation
- Output layer: 3 units, Softmax activation (for 3 wine classes)

## Training
The model is trained using categorical cross-entropy loss for 20 epochs on the entire dataset.

## Evaluation & Prediction
After training, the model's predictions are compared to the true classes. The notebook prints:
- Accuracy
- Confusion matrix
- Classification report
- First 10 predictions vs. true labels

## Usage
1. Place `wines.csv` in the same directory as the notebook.
2. Open and run `wine_quality_check(NN).ipynb` in Jupyter Notebook or JupyterLab.
3. The notebook will train the model and display evaluation metrics.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- keras
- tensorflow
- seaborn (optional, for visualization)

Install requirements with:
```bash
pip install pandas numpy scikit-learn keras tensorflow seaborn
```

## Notes
- The model is trained and evaluated on the same data (no train/test split). For real applications, use a train/test split or cross-validation.
- The neural network weights are initialized to zeros for demonstration; in practice, use default initializers for better performance.

---
Feel free to modify and extend this project for your own experiments! 