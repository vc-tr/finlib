# Research Insights: LSTM Forecasting in Finance

## Research Tasks

1. **Read two seminal papers on LSTM forecasting in finance**
   - Examples: "Deep Learning for Stock Prediction," "Attention-augmented LSTMs for Time Series"
   
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
- **Layers:** *[To be filled]*
- **Hidden sizes:** *[To be filled]*
- **Bidirectional/Attention choices:** *[To be filled]*

#### Training Configuration
- **Schedulers:** *[To be filled]*
- **Clipping:** *[To be filled]*
- **Early stopping:** *[To be filled]*

#### Pitfalls & Warnings
- **Data leakage:** *[To be investigated]*
- **Overfitting:** *[To be investigated]*

---

## Additional Resources

### Papers for Further Analysis
- [A hybrid stock trading system using genetic network programming and mean conditional value at risk](https://www.academia.edu/34764533/A_hybrid_stock_trading_system_using_genetic_network_programming_and_mean_conditional_value_at_risk#loswp-work-container)
- [ACM Paper 1](https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3694860.3694870)
- [ACM Paper 2](https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3700058.3700075)

### Notes
- Complete architecture details for Paper 1 pending full paper review
- Second paper analysis slot available for additional research





