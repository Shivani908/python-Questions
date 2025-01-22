# Importing required libraries 
import pandas as pd 
import numpy as np
import json 
import matplotlib.pyplot as plt 
import seaborn as sns

# 1. Understanding Structured Data
print("1. Understanding Structured Data")
structured_data = pd.DataFrame({ 'Employee ID': [101, 102, 103], 'Name': ['Alice', 'Bob', 'Charlie'], 'Department': ['HR', 'IT', 'Finance'], 'Salary': [60000, 75000, 55000] })
print(structured_data)