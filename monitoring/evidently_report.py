import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv("prediction_logs.csv")

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=df.iloc[:len(df)//2],
    current_data=df.iloc[len(df)//2:]
)

report.save_html("evidently_report.html")

print("Evidently report generated")