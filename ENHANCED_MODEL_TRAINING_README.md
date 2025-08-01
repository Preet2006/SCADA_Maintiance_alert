# Enhanced Model Training Pipeline

## Overview

This enhanced pipeline compares multiple machine learning models for failure prediction, going beyond just CatBoost to include a comprehensive evaluation of various algorithms.

## 🚀 What's New

### Models Included:
1. **CatBoost** - Your original choice with optimized parameters
2. **XGBoost** - Gradient boosting with advanced regularization
3. **LightGBM** - Light gradient boosting machine
4. **Random Forest** - Ensemble of decision trees
5. **Logistic Regression** - Linear model baseline
6. **SVM** - Support Vector Machine with RBF kernel

### Enhanced Features:
- **Cross-validation** for robust performance estimation
- **Ensemble methods** (voting classifier)
- **Comprehensive metrics** (Accuracy, F1-Macro, CV scores)
- **Visual comparisons** (model comparison plots)
- **Automatic best model selection**
- **Detailed evaluation reports**

## 📊 Model Comparison Metrics

The pipeline evaluates each model using:
- **Accuracy**: Overall prediction accuracy
- **F1-Macro**: Balanced F1 score across all classes
- **Cross-validation**: 5-fold stratified CV for robust estimation
- **Confusion Matrix**: Detailed class-wise performance

## 🎯 Ensemble Methods

### Voting Classifier
Combines predictions from the top 4 tree-based models:
- Random Forest
- XGBoost  
- LightGBM
- CatBoost

Uses soft voting (probability averaging) for better performance.

## 📁 Output Files

After running the pipeline, you'll get:

### Models:
- `future_model.joblib` - Best performing model
- `model_comparison_results.joblib` - All trained models

### Predictions:
- `predictions_comparison.csv` - Predictions from best model and ensemble

### Visualizations:
- `confusion_matrix.png` - Confusion matrix for best model
- `model_comparison.png` - Bar charts comparing all models
- `feature_importance.png` - Feature importance (for tree-based models)

## 🚀 How to Run

### Option 1: Direct Execution
```bash
python future_model_training.py
```

### Option 2: Using the Runner Script
```bash
python run_enhanced_training.py
```

## 📈 Expected Results

The pipeline will output a comprehensive comparison table showing:

```
Model                Accuracy  F1-Macro  CV-F1-Mean  CV-F1-Std
CatBoost             0.XXX     0.XXX     0.XXX       0.XXX
XGBoost              0.XXX     0.XXX     0.XXX       0.XXX
LightGBM             0.XXX     0.XXX     0.XXX       0.XXX
Random Forest        0.XXX     0.XXX     0.XXX       0.XXX
Logistic Regression  0.XXX     0.XXX     0.XXX       0.XXX
SVM                  0.XXX     0.XXX     0.XXX       0.XXX
```

## 🔧 Model Configurations

### CatBoost (Original)
- Iterations: 1000
- Learning rate: 0.05
- Depth: 6
- Class weights: [1, 2, 3]
- Early stopping: 50 rounds

### XGBoost
- Estimators: 200
- Max depth: 6
- Learning rate: 0.1
- Subsampling: 0.8

### LightGBM
- Estimators: 200
- Max depth: 6
- Learning rate: 0.1
- Subsampling: 0.8

### Random Forest
- Estimators: 200
- Max depth: 10
- Class weight: balanced

### Logistic Regression
- Max iterations: 1000
- Class weight: balanced
- Multi-class: multinomial

### SVM
- Kernel: RBF
- C: 1.0
- Gamma: scale
- Class weight: balanced

## 🎯 Key Benefits

1. **Comprehensive Comparison**: See how CatBoost performs against other state-of-the-art models
2. **Robust Evaluation**: Cross-validation ensures reliable performance estimates
3. **Ensemble Power**: Voting classifier often outperforms individual models
4. **Automatic Selection**: Best model is automatically selected and saved
5. **Detailed Analysis**: Rich visualizations and metrics for decision making

## 📊 Interpretation Guide

### F1-Macro Score
- **0.9+**: Excellent performance
- **0.8-0.9**: Very good performance  
- **0.7-0.8**: Good performance
- **<0.7**: Needs improvement

### Cross-Validation
- Lower standard deviation = More stable model
- Higher mean = Better average performance

### Model Selection Criteria
The pipeline automatically selects the best model based on **F1-Macro score**, which is optimal for imbalanced classification problems.

## 🔄 Next Steps

After running the enhanced training:

1. **Analyze Results**: Review the comparison table and visualizations
2. **Select Model**: Use the best performing model for production
3. **Fine-tune**: If needed, hyperparameter tune the best model
4. **Deploy**: Use the saved model in your API

## 🛠️ Troubleshooting

### Common Issues:

1. **Memory Error**: Reduce model complexity or use smaller datasets
2. **Import Errors**: Ensure all packages are installed:
   ```bash
   pip install xgboost lightgbm scikit-learn catboost imbalanced-learn
   ```
3. **Data Issues**: Check that your CSV file exists and has the expected format

### Performance Tips:

1. **For Large Datasets**: Consider using a subset for initial comparison
2. **For Speed**: Reduce CV folds or model complexity
3. **For Memory**: Use smaller ensemble sizes

## 📝 Notes

- The pipeline uses **stratified cross-validation** to maintain class balance
- **SMOTE** is applied to handle class imbalance
- **Time-based splitting** preserves temporal order
- All models are trained on the same balanced dataset for fair comparison

---

**Happy Model Training! 🚀** 