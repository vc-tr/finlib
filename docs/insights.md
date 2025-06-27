Tasks:

	1.	Read two seminal papers on LSTM forecasting in finance (e.g. “Deep Learning for Stock Prediction,” “Attention-augmented LSTMs for Time Series”).

	MA-LSTM: A stock prediction model based on multi-level attention-based LSTM network
	https://dl.acm.org/doi/10.1145/3700058.3700061

	2.	For each, extract:
	•	Model architecture (layers, hidden sizes, bidirectional/attention choices)

		What is LSTM neural network:
			- Long short-term memory
			- Handling sequential data and time series prediction
			- Have many layers
			- Input gate, forget gate, output gate

			- LSTM is a Recurrent neural network that avoid gradient to go too higher too low
			- 2 path:Long-term memory and short-term memory
			- Use sigmoid (e^x / 1+ e^x) (0 <y< 1): Percent to remember and tanh activation functions (e^x - e^-x / e^x + e^-x) (-1< y< 1): Potential memory
			- 3 stage


	•	Training tricks (schedulers, clipping, early-stop settings)
	•	Report any pitfalls (data leakage, overfitting warnings).
