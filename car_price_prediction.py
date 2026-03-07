import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
# ------------------ Load Dataset ------------------
df = pd.read_csv("car_price_prediction.csv")

df.columns = df.columns.str.replace(" ", "_").str.replace(".", "").str.replace("-", "_")

print("Columns after cleaning:", df.columns.tolist())

# Handle Levy
df['Levy'] = df['Levy'].replace('-', 0).astype(float)

# Drop rows with missing values
df = df.dropna()

categories = sorted(df['Category'].unique())

print("First 5 rows of dataset:")
print(df.head())

# ------------------ Data Preprocessing ------------------

# Rename columns to match expected
df = df.rename(columns={
    'Prod_year': 'Year',
    'Price': 'Selling_Price',
    'Mileage': 'Kms_Driven',
    'Fuel_type': 'Fuel_Type',
    'Gear_box_type': 'Transmission',
    'Leather_interior': 'Leather_Interior',
    'Drive_wheels': 'Drive_Wheels'
})

# Clean Kms_Driven
df['Kms_Driven'] = df['Kms_Driven'].str.replace(' km', '').astype(int)

# Clean Engine_volume
df['Engine_volume'] = df['Engine_volume'].str.replace(' Turbo', '').astype(float)

# Convert categorical data to numbers
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()
le_category = LabelEncoder()
le_leather = LabelEncoder()
le_drive = LabelEncoder()
le_wheel = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Transmission'] = le_transmission.fit_transform(df['Transmission'])
df['Category'] = le_category.fit_transform(df['Category'])
df['Leather_Interior'] = le_leather.fit_transform(df['Leather_Interior'])
df['Drive_Wheels'] = le_drive.fit_transform(df['Drive_Wheels'])
df['Wheel'] = le_wheel.fit_transform(df['Wheel'])

# ------------------ Feature Selection ------------------

X = df[['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Engine_volume', 'Cylinders', 'Airbags', 'Category', 'Leather_Interior', 'Drive_Wheels', 'Wheel', 'Levy']]
y = np.log(df['Selling_Price'] + 1)

# ------------------ Train Test Split ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Model Training ------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------ Prediction ------------------

y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log) - 1

# Get original y_test
y_test_original = df.loc[y_test.index, 'Selling_Price']

# ------------------ Model Evaluation ------------------

print("\nModel Performance:")

print("Mean Absolute Error:", mean_absolute_error(y_test_original, y_pred))
print("R2 Score:", r2_score(y_test_original, y_pred))
print("Model accuracy:", round(r2_score(y_test_original, y_pred) * 100, 2), "%")

# ------------------ GUI for Prediction ------------------

def predict_price():
    try:
        year = int(year_entry.get())
        kms = int(kms_entry.get())
        fuel = fuel_var.get()
        transmission = transmission_var.get()
        engine_volume = float(engine_entry.get())
        cylinders = float(cylinders_entry.get())
        airbags = int(airbags_entry.get())
        category = category_var.get()
        leather = leather_var.get()
        drive = drive_var.get()
        wheel = wheel_var.get()
        levy = float(levy_entry.get())

        if not fuel or not transmission or not category or not leather or not drive or not wheel:
            tk.messagebox.showerror("Error", "Please fill all fields")
            return

        fuel_encoded = le_fuel.transform([fuel])[0]
        transmission_encoded = le_transmission.transform([transmission])[0]
        category_encoded = le_category.transform([category])[0]
        leather_encoded = le_leather.transform([leather])[0]
        drive_encoded = le_drive.transform([drive])[0]
        wheel_encoded = le_wheel.transform([wheel])[0]

        new_car = np.array([[year, kms, fuel_encoded, transmission_encoded, engine_volume, cylinders, airbags, category_encoded, leather_encoded, drive_encoded, wheel_encoded, levy]])

        predicted_price_log = model.predict(new_car)
        predicted_price = np.exp(predicted_price_log) - 1

        result_label.config(text=f"Predicted Car Price: {round(predicted_price[0], 2)}")
    except Exception as e:
        tk.messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("450x650")

# Load background image
bg_image = ImageTk.PhotoImage(Image.open("assets/car_price_prediction_background.png"))

# Create canvas
canvas = tk.Canvas(root, width=450, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_image, anchor="nw")

# Title
title_label = tk.Label(root, text="🚗CAR PRICE PREDICTOR📊", fg="white", bg="#000000", font=("Arial", 16, "underline","bold"))
canvas.create_window(225, 20, window=title_label)

# Year
year_label = tk.Label(root, text="Enter Car Year:")
canvas.create_window(225, 50, window=year_label)
year_entry = tk.Entry(root)
canvas.create_window(225, 72, window=year_entry)

# Kms
kms_label = tk.Label(root, text="Enter Kms Driven:")
canvas.create_window(225, 98, window=kms_label)
kms_entry = tk.Entry(root)
canvas.create_window(225, 120, window=kms_entry)

# Fuel
fuel_label = tk.Label(root, text="Fuel Type:")
canvas.create_window(225, 146, window=fuel_label)
fuel_var = tk.StringVar()
fuel_menu = ttk.Combobox(root, textvariable=fuel_var, values=["Hybrid", "Petrol", "Diesel", "CNG"])
canvas.create_window(225, 168, window=fuel_menu)

# Transmission
transmission_label = tk.Label(root, text="Transmission:")
canvas.create_window(225, 194, window=transmission_label)
transmission_var = tk.StringVar()
transmission_menu = ttk.Combobox(root, textvariable=transmission_var, values=["Automatic", "Tiptronic", "Variator", "Manual"])
canvas.create_window(225, 216, window=transmission_menu)

# Engine Volume
engine_label = tk.Label(root, text="Enter Engine Volume:")
canvas.create_window(225, 242, window=engine_label)
engine_entry = tk.Entry(root)
canvas.create_window(225, 264, window=engine_entry)

# Cylinders
cylinders_label = tk.Label(root, text="Enter Cylinders:")
canvas.create_window(225, 290, window=cylinders_label)
cylinders_entry = tk.Entry(root)
canvas.create_window(225, 312, window=cylinders_entry)

# Airbags
airbags_label = tk.Label(root, text="Enter Airbags:")
canvas.create_window(225, 338, window=airbags_label)
airbags_entry = tk.Entry(root)
canvas.create_window(225, 360, window=airbags_entry)

# Category
category_label = tk.Label(root, text="Category:")
canvas.create_window(225, 386, window=category_label)
category_var = tk.StringVar()
category_menu = ttk.Combobox(root, textvariable=category_var, values=categories)
canvas.create_window(225, 408, window=category_menu)

# Leather
leather_label = tk.Label(root, text="Leather Interior:")
canvas.create_window(225, 434, window=leather_label)
leather_var = tk.StringVar()
leather_menu = ttk.Combobox(root, textvariable=leather_var, values=["Yes", "No"])
canvas.create_window(225, 456, window=leather_menu)

# Drive
drive_label = tk.Label(root, text="Drive Wheels:")
canvas.create_window(225, 482, window=drive_label)
drive_var = tk.StringVar()
drive_menu = ttk.Combobox(root, textvariable=drive_var, values=["4x4", "Front", "Rear"])
canvas.create_window(225, 504, window=drive_menu)

# Wheel
wheel_label = tk.Label(root, text="Wheel:")
canvas.create_window(225, 530, window=wheel_label)
wheel_var = tk.StringVar()
wheel_menu = ttk.Combobox(root, textvariable=wheel_var, values=["Left wheel", "Right-hand drive"])
canvas.create_window(225, 552, window=wheel_menu)

# Levy
levy_label = tk.Label(root, text="Enter Levy:")
canvas.create_window(225, 578, window=levy_label)
levy_entry = tk.Entry(root)
canvas.create_window(225, 600, window=levy_entry)

# Button
predict_button = tk.Button(root, text="Predict Price", command=predict_price, bg="#140979", fg="white", font=("Arial", 12, "bold"))
canvas.create_window(225, 626, window=predict_button)

# Result
result_label = tk.Label(root, text="", fg="#ff0073", font=("Arial", 14, "bold"))
canvas.create_window(225, 660, window=result_label)

root.mainloop()