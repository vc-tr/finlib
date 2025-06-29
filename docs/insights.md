# Research Insights: LSTM Forecasting in Finance

## Research Tasks

1. **Read two seminal papers on LSTM forecasting in finance**
2. **For each paper, extract:**
   - Model architecture (layers, hidden sizes, bidirectional/attention choices)
   - Training tricks (schedulers, clipping, early-stop settings)
   - Report any pitfalls (data leakage, overfitting warnings)

---

## Paper Analysis

### Paper 1: Improving Stock Price Prediction using Linear Regression and Long Short-Term Memory Model (LR-LSTM)

**Source:** [ACM Digital Library](https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3659211.3659343)

#### Data Preprocessing
- **Data Collection:**
  - Three-year dataset: 777 records
  - Five-year dataset: 1,136 records
  - Datasets overlap (last three years of five-year dataset matches three-year dataset)
  - Key attributes: Date, Opening Price, Highest Price, Lowest Price, Closing Price, Adjusted Closing Price, Traded Volume

- **Preprocessing Steps:**
  - Noise removal
  - Data normalization using Min-Max Scaling to range [0, 1]:
    
    $$x_{scaled} = \frac{x_i - \min(x)}{\max(x) - \min(x)}$$
    
  - Train/test split: 70/30

#### Model Architecture
- **Layers:** Not mention, will check out reference [3]
- **Hidden sizes:** N/A
- **Bidirectional/Attention choices:** Univariate Long Short-Term Memory (uniLSTM)

#### Training Configuration
- **Schedulers:** *[To be filled]*
- **Clipping:** *[To be filled]*
- **Early stopping:** *[To be filled]*

#### Pitfalls & Warnings
- **Data leakage:** *[To be investigated]*
- **Overfitting:** *[To be investigated]*

---

### Paper 2: 

**Source:** [ACM Paper](https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3599957.3606240)

#### Data Preprocessing
- **Data Collection:**
  - OHCLV, IBM, 785 observations $\approx$ 3 years and 2 months

- **Preprocessing Steps:**
  - Normalization
  - Dickey Fuller Test
  - 9 delays

#### Model Architecture
- **Layers:** 3 LSTM layers + 1 dropout layer + 1 dense layer
- **Hidden sizes:** (0,20,512)(0,20,64)(0,8)
- **Bidirectional/Attention choices:** N/A
- **no.Parameters:** 1,210,921

#### Training Configuration
- **Schedulers:** N/A
- **Clipping:** N/A
- **Early stopping:** N/A
- **Accuracy Metrics:** Mean Square Error, Root Mean Square Error, Mean Absolute Error

#### Pitfalls & Warnings
- **Data leakage:** *[To be investigated]*
- **Overfitting:** *[To be investigated]*
---
## Additional Resources

### Papers for Further Analysis
- [A hybrid stock trading system using genetic network programming and mean conditional value at risk](https://www.academia.edu/34764533/A_hybrid_stock_trading_system_using_genetic_network_programming_and_mean_conditional_value_at_risk#loswp-work-container)
- [ACM Paper 2](https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3694860.3694870)
- [ACM Paper 3](https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3700058.3700075)






