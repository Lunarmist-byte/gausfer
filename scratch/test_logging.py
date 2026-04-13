import os
from datetime import datetime
import traceback

log_file = "error_report.log"
try:
    with open(log_file, "a") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TEST LOG ENTRY\n")
        f.write("Error: Test error message\n")
        f.write("Traceback: (simulated)\n")
        f.write("-" * 40 + "\n")
    print(f"Successfully wrote to {log_file}")
    with open(log_file, "r") as f:
        print("Last line:", f.readlines()[-1].strip())
except Exception as e:
    print(f"Failed to write log: {e}")
