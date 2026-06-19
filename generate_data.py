import pandas as pd
import numpy as np
import os
import random

np.random.seed(42)
random.seed(42)

NUM_STUDENTS = 980
NUM_CERTIFIED = 789
NUM_NOT_CERTIFIED = NUM_STUDENTS - NUM_CERTIFIED  # 191

# Build the exact outcome list (no randomness in the COUNT, only in WHO gets it)
outcomes = ["Certified"] * NUM_CERTIFIED + ["Not Certified"] * NUM_NOT_CERTIFIED
random.shuffle(outcomes)

records = []
for i in range(1, NUM_STUDENTS + 1):
    sid   = f"STU{str(i).zfill(4)}"
    name  = f"Student_{i}"
    email = f"student{i}@gmail.com"

    # joining time between 13:00 and 13:20
    join_h, join_m = 13, random.randint(0, 20)
    joining_time   = f"{join_h:02d}:{join_m:02d}"

    # Exact split: 789 Certified / 191 Not Certified (assigned above, shuffled)
    target_status = outcomes[i - 1]
    if target_status == "Certified":
        duration = random.randint(96, 120)
    else:
        # 20 min (16.7%) up to 94 min (78.3%) -> no one below 20 min,
        # and the highest Not Certified students sit just under 80%, around 78%
        duration = random.randint(20, 94)

    leave_total = join_h * 60 + join_m + duration
    leave_h     = leave_total // 60
    leave_m     = leave_total % 60
    leaving_time = f"{leave_h:02d}:{leave_m:02d}"

    att_pct = round((duration / 120) * 100, 6)  # out of 120 min max session
    status  = "Certified" if att_pct >= 80 else "Not Certified"

    records.append({
        "student_id":    sid,
        "name":          name,
        "email":         email,
        "joining_time":  joining_time,
        "leaving_time":  leaving_time,
        "total_duration": duration,
        "Total_Attendance": duration,
        "Attendance_Percentage": att_pct,
        "Certification_Status": status
    })

df = pd.DataFrame(records)
os.makedirs("data", exist_ok=True)
df.to_csv("data/session_1.csv", index=False)

print("Dataset Created Successfully")
print(df[["student_id","name","email","joining_time","leaving_time","total_duration"]].head())
print("\nMissing Values:")
print(df[["student_id","name","email"]].isnull().sum())
print()
cert = df["Certification_Status"].value_counts()
print("Certification Count")
print(f"Certification_Status")
print(f"Not Certified    {cert.get('Not Certified', 0)}")
print(f"Certified        {cert.get('Certified', 0)}")
print(f"Name: count, dtype: int64")
