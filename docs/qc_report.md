
Tasks:
	1.	Use the YahooDataFetcher to pull 5 days of SPY 1-min bars.
	2.	Analyze the raw timestamps:
	•	Identify missing intervals, duplicated rows, DST gaps.
	•	Quantify how many minutes are missing per day.
	3.	Propose concrete backfill or drop rules (e.g., forward-fill OHLC but zero out volume).
	4.	Write up your findings and recommendations in docs/qc_report.md, referencing the code in src/pipeline/pipeline.py.