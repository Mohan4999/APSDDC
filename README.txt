================================================================================
  STUDENT ATTENDANCE MANAGEMENT & CERTIFICATION SYSTEM
  Internship Milestone Project
================================================================================

PROJECT OVERVIEW
----------------
End-to-end pipeline for student attendance analysis, certification
determination, and machine learning classification.

FOLDER STRUCTURE
----------------
  Attendence_ML_Project/
  +-- data/
  |   +-- session_1.csv              Raw attendance dataset (980 students)
  +-- graphs/
  |   +-- bar_graph.png              Certified vs Not Certified bar chart
  |   +-- scatter_plot.png           Attendance Analysis scatter plot
  +-- outputs/
  |   +-- Final_Attendance_Report.csv  Full report with certification status
  +-- attendance.py                  Main pipeline script
  +-- generate_data.py               Dataset generator
  +-- README.txt                     This file

HOW TO RUN
----------
  Step 1: python generate_data.py     # Generate dataset (run once)
  Step 2: python attendance.py        # Run full pipeline

DATASET COLUMNS
---------------
  student_id, name, email, joining_time, leaving_time,
  total_duration, Total_Attendance, Attendance_Percentage, Certification_Status

CONSTRAINTS IMPLEMENTED
-----------------------
  1.  Attendance >= 80% required for certification
  2.  No duplicate student records
  3.  No missing/null values
  4.  Valid attendance values only (duration >= 0)
  5.  Dataset contains 980 unique students
  6.  ML: Random Forest classification (Certified / Not Certified)
  7.  Graphs generated only after preprocessing
  8.  Final report exported as CSV
  9.  Student ID is unique
  10. System processes all records efficiently (< 5 seconds)

FORMULA
-------
  Attendance Percentage = (total_duration / 120) x 100
  Certification         = Certified if Attendance % >= 80 else Not Certified

================================================================================