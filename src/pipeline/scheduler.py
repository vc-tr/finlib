"""
Tasks:
	1.	Design a block cross-validation splitter that prevents lookahead leakage (e.g., rolling-window with no overlap).
	2.	Implement it in src/pipeline/scheduler.py as a function that, given N samples and k folds, returns train/val indices.
	3.	In tests/test_dataset.py, generate a synthetic sequence and assert that no test index ever precedes any train index in time.
	4.	Update dataset.py to accept a splitter argument and integrate your function.
 """