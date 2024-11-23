<div align="center">
  <img src="Credit-risk-analysis-github-presentation-graphic.jpg" alt="Credit Risk Analysis" style="border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

# Credit Risk Classification Analysis

## Project Overview

This project implements a machine learning solution for credit risk assessment using supervised learning techniques. The model is designed to predict loan default risk using historical lending data from a peer-to-peer lending services company. Using logistic regression with L2 regularization, the model achieves 99% accuracy in classifying loans as either healthy or high-risk.

Key business recommendations based on the analysis:

1. The model demonstrates excellent reliability for loan approval automation, with high precision for both healthy (100%) and high-risk loans (84%)
2. Special attention should be paid to applications flagged as high-risk (84% precision), while healthy loan predictions are extremely reliable (100% precision)
3. The model successfully captures 99% of actual high-risk loans, making it an extremely effective tool for risk mitigation
4. Implementation could significantly reduce manual review workload while maintaining strict risk controls, given the 99% overall accuracy
5. Regular model retraining is recommended to maintain the high recall (99%) and precision metrics as lending patterns evolve

## Technical Details

### Model Architecture
- **Algorithm**: Logistic Regression with L2 regularization
- **Preprocessing**: StandardScaler for feature normalization
- **Train-Test Split**: 75-25 ratio with stratification
- **Random State**: 1 (for reproducibility)

### Feature Set
- Loan Size
- Interest Rate
- Borrower Income
- Debt-to-Income Ratio
- Number of Accounts
- Derogatory Marks
- Total Debt

### Target Variable
- `0`: Healthy Loan (Majority Class)
- `1`: High-Risk Loan (Minority Class)

## Model Performance Metrics

### Classification Metrics
| Metric    | Healthy Loans (0) | High-Risk Loans (1) |
|-----------|------------------|-------------------|
| Precision | 1.00            | 0.84             |
| Recall    | 0.99            | 0.94             |
| F1-Score  | 1.00            | 0.91             |

### Confusion Matrix Analysis

### Key Performance Indicators
- **Overall Accuracy**: 99%
- **Type I Error Rate**: 0.57% (False Positives)
- **Type II Error Rate**: 5.97% (False Negatives)
- **ROC-AUC Score**: 0.99

## Test Size Recommendations

1. **Optimal Test Split: 20-30%**
   - Current model uses 75/25 split, which aligns with best practices
   - Rationale: Balances training data volume with validation robustness
   - For our dataset (~77,536 records), provides sufficient test samples (~19,384)

2. **Cross-Validation Considerations**
   - Recommend 5-fold cross-validation for model stability
   - K-fold CV particularly important given class imbalance
   - Helps validate model performance across different data subsets

3. **Stratification Requirements**
   - Maintain class distribution in train/test splits
   - Critical due to imbalanced target variable (healthy vs. high-risk loans)
   - Ensures representative sampling of both loan categories

4. **Data Volume Guidelines**
   - Minimum test size: 3,000 samples for reliable metrics
   - Maximum test size: 30% to maintain adequate training data
   - Current split provides statistical significance for both classes

## Statistical Analysis

### Model Strengths
1. **High Precision for Healthy Loans (1.00)**
   - Minimizes false positives
   - Ensures reliable approval decisions

2. **Strong Recall for High-Risk Loans (0.94)**
   - Effectively captures potential defaults
   - Reduces financial exposure

3. **Balanced Performance**
   - Handles class imbalance effectively
   - Maintains reliability across both classes

### Areas for Optimization
1. **High-Risk Precision (0.84)**
   - Room for improvement in reducing false high-risk classifications
   - Consider ensemble methods or feature engineering

## Implementation Stack

### Core Technologies
- **Python** 3.10.14
- **Pandas** for data manipulation
- **Scikit-learn** for model implementation
- **NumPy** for numerical operations
- **Pathlib** for file handling

### Development Environment
- Jupyter Notebook
- Anaconda Environment

## Data Architecture

### Source Data
- Format: CSV
- Location: `Resources/lending_data.csv`
- Records: ~77,536
- Features: 7 numerical variables
- Target: Binary classification

### Data Processing Pipeline
1. Data loading and validation
2. Feature standardization
3. Train-test splitting
4. Model fitting and prediction
5. Performance evaluation

## Business Impact

### Risk Management Benefits
- 99% accuracy in loan classification
- Reduced manual review requirements
- Standardized risk assessment process

### Financial Implications
- Minimized default risk exposure
- Optimized loan approval efficiency
- Enhanced portfolio quality management

## Recommendations

Based on the comprehensive analysis, this model is recommended for production implementation with the following considerations:

1. **Primary Strengths**
   - Exceptional accuracy in healthy loan identification
   - Strong risk detection capabilities
   - Robust performance metrics

2. **Implementation Strategy**
   - Deploy as primary screening tool
   - Maintain human oversight for high-risk predictions
   - Regular model retraining with new data

3. **Monitoring Requirements**
   - Track prediction drift
   - Monitor class distribution changes
   - Validate performance metrics monthly

## Potential Future Enhancements

1. **Model Improvements**
   - Feature engineering for high-risk precision
   - Ensemble modeling evaluation
   - Deep learning comparison study

2. **System Integration**
   - API development for real-time scoring
   - Automated retraining pipeline
   - Performance monitoring dashboard

## Disclaimer

This analysis is part of a data analytics project and should be used as one component of a comprehensive credit risk assessment strategy. Additional factors and expert review should be incorporated into final lending decisions.

---
*Repository maintained by [Sergei Sergeev]*
*Email: sergei.sergeev.n@gmail.com*
*Last Updated: [2024-11-23]*