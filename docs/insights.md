Tasks:
	1.	Read two seminal papers on LSTM forecasting in finance (e.g. “Deep Learning for Stock Prediction,” “Attention-augmented LSTMs for Time Series”).
	2.	For each, extract:
	•	Model architecture (layers, hidden sizes, bidirectional/attention choices)
	•	Training tricks (schedulers, clipping, early-stop settings)
	•	Report any pitfalls (data leakage, overfitting warnings).
---
Improving Stock Price Prediction using Linear Regression and Long Short-Term Memory Model (LR-LSTM) 
 - https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3659211.3659343:

•	Model architecture:
	- Preprocess:
		+ Data collecting: a three-year dataset with 777 records and a five-year dataset with 1136 records. 
		For comparability, the three-year dataset overlapped with the last three years of the five-year dataset.
		included key stock attributes: Date, Opening Price, Highest Price, Lowest Price, Closing Price, Adjusted Closing Price, and Traded Volume.
		+ Noise Removal
		+ Data Normalization: scale the input data to a standard range [0, 1] using Min-Max Scaling Technique
			$x_scaled = \frac{x_i - min(x)}{max(x)-min(x)}$
		+ 70/30 split
	- Layers:
	- Hidden sizes:
	- Bidirectional/Attention choices:

•	Training tricks (schedulers, clipping, early-stop settings)
•	Report any pitfalls (data leakage, overfitting warnings).


---
Extra sources:
- https://www.academia.edu/34764533/A_hybrid_stock_trading_system_using_genetic_network_programming_and_mean_conditional_value_at_risk#loswp-work-container
- https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3694860.3694870
- https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3700058.3700075





