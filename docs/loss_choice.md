Tasks:
	1.	Compare MSE vs MAE vs Huber (Smooth L1) on a tiny synthetic time series:
	•	Derive each loss’s gradient sensitivity to outliers.
	•	Run 1-epoch training on a 100-point sequence, logging train/val loss for each loss function.
	2.	Write up your recommendation in docs/loss_choice.md, citing your small experiment.
	3.	(Optional) Add a command-line flag to train.py (--loss {mse,mae,huber}), wire up nn.SmoothL1Loss() for “huber,” and write a smoke-test in tests/test_train.py that runs with all three loss options without error.