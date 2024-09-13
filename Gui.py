import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load data from CSV file
file_path = 'M.csv'
data = pd.read_csv(file_path)

# Ensure 'education' is numeric (0 and 1)
education_mapping_model = {' Graduate': 1, ' Non-Graduate': 0}

# Apply the mapping and handle missing values
data[' education'] = data[' education'].map(education_mapping_model).fillna(0).astype(int)

# Select specific columns for features and label
features = [' education', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' bank_asset_value']
X = data[features]
y = data[' loan_status']

# Initialize k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Function to make predictions
def predict():
    try:
        education = int(education_var.get())
        income_annum = float(income_annum_var.get())
        loan_amount = float(loan_amount_var.get())
        loan_term = int(loan_term_var.get())
        cibil_score = int(cibil_score_var.get())
        bank_asset_value = float(bank_asset_value_var.get())

        # Adjust DataFrame to use the same feature names as during model training
        input_data = pd.DataFrame([[education, income_annum, loan_amount, loan_term, cibil_score, bank_asset_value]],
                                  columns=[' education', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' bank_asset_value'])

        prediction = knn.predict(input_data)
        result_label.config(text=f"Loan Status: {prediction[0]}")
    except ValueError:
        result_label.config(text="Please enter valid inputs.")

# Function to clear inputs
def clear_inputs():
    education_var.set("")
    income_annum_var.set("")
    loan_amount_var.set("")
    loan_term_var.set("")
    cibil_score_var.set("")
    bank_asset_value_var.set("")
    result_label.config(text="Loan Status: ")

# Function to map education numeric values back to labels
def map_education(value):
    return ' Graduate' if value == 1 else ' Non Graduate'

# Initialize main window
root = tk.Tk()
root.title("KNN-Based Model for Predicting Loan Approval")

# Center the window on the screen
window_width = 1280
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width // 2) - (window_width // 2)
y_position = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Set background color
root.configure(bg='#2b2b2b')  # Background for main window

# Define input variables
education_var = tk.StringVar()
income_annum_var = tk.StringVar()
loan_amount_var = tk.StringVar()
loan_term_var = tk.StringVar()
cibil_score_var = tk.StringVar()
bank_asset_value_var = tk.StringVar()

# Create input frame
frame_inputs = tk.Frame(root, bg='#333333')
frame_inputs.pack(side='left', fill='y', padx=20, pady=20)

# Title Label
title_label = tk.Label(frame_inputs, text="KNN-Based Model for Predicting Loan Approval", font=("Helvetica", 16, 'bold'), bg='#B84C4C', fg='#ffffff')
title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

# Input Label and Entry Widgets
input_labels = ["Education (0: Non-Graduate, 1: Graduate):", "Income (Annually):", "Loan Amount:", "Loan Term:", "CIBIL Score:", "Bank Asset Value:"]
input_vars = [education_var, income_annum_var, loan_amount_var, loan_term_var, cibil_score_var, bank_asset_value_var]

for i, (label_text, var) in enumerate(zip(input_labels, input_vars)):
    label = tk.Label(frame_inputs, text=label_text, font=("Helvetica", 12), bg='#333333', fg='#ffffff')
    label.grid(row=i+1, column=0, padx=20, pady=10, sticky='w')

    entry = tk.Entry(frame_inputs, textvariable=var, font=("Helvetica", 12), bg='#E0E0E0', relief='solid', bd=1, width=12)
    entry.grid(row=i+1, column=1, padx=20, pady=10)

# Predict and Clear Buttons
button_frame = tk.Frame(frame_inputs, bg='#333333')
button_frame.grid(row=7, column=0, columnspan=2, pady=20)

predict_button = tk.Button(button_frame, text="Predict", command=predict, font=("Helvetica", 12, 'bold'), bg='#CC4C4C', fg='#ffffff', relief='solid', bd=1, width=10)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_inputs, font=("Helvetica", 12, 'bold'), bg='#CC4C4C', fg='#ffffff', relief='solid', bd=1, width=10)
clear_button.grid(row=0, column=1, padx=10)

# Result Label
result_label = tk.Label(frame_inputs, text="Loan Status:", font=("Helvetica", 12, 'bold'), bg='#333333', fg='#ffffff')
result_label.grid(row=8, column=0, columnspan=2, padx=20, pady=20)

# Data Display Frame (Treeview)
frame_data = tk.Frame(root, bg='#B84C4C')
frame_data.pack(side='right', fill='both', expand=True, padx=20, pady=20)

data_label = tk.Label(frame_data, text="Training Data", font=("Helvetica", 16, 'bold'), bg='#CC4C4C', fg='#ffffff')
data_label.pack(pady=10)

tree = ttk.Treeview(frame_data, columns=("Education", "Income", "Loan Amount", "Loan Term", "CIBIL Score", "Bank Asset Value", "Loan Status"), show="headings", height=15)
tree.pack(fill='both', expand=True)

# Define column headings
tree.heading("Education", text="Education")
tree.heading("Income", text="Income")
tree.heading("Loan Amount", text="Loan Amount")
tree.heading("Loan Term", text="Loan Term")
tree.heading("CIBIL Score", text="CIBIL Score")
tree.heading("Bank Asset Value", text="Bank Asset Value")
tree.heading("Loan Status", text="Loan Status")

# Set column widths
tree.column("Education", width=100)
tree.column("Income", width=70)
tree.column("Loan Amount", width=90)
tree.column("Loan Term", width=70)
tree.column("CIBIL Score", width=80)
tree.column("Bank Asset Value", width=130)
tree.column("Loan Status", width=90)

# Insert training data into the table with mapped education labels
for i in range(len(X)):
    tree.insert("", "end", values=(map_education(X.iloc[i][' education']), X.iloc[i][' income_annum'], X.iloc[i][' loan_amount'], X.iloc[i][' loan_term'], X.iloc[i][' cibil_score'], X.iloc[i][' bank_asset_value'], y.iloc[i]))

# Configure grid row and column weights
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=2)
root.grid_columnconfigure(1, weight=1)

# Apply some styling
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=10)

# Set font size and colors for Treeview
style.configure('Treeview', background='#FFFFFF', foreground='black', rowheight=25, font=('Helvetica', 10))
style.configure('Treeview.Heading', font=('Helvetica', 10, 'bold'), background='#CC4C4C', foreground='black')

# Run the application
root.mainloop()
