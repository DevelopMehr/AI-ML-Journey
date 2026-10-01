import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# our student data 
# feature : cgpa , pocket money
# target : scholarship (yes , no)

x = np.array([
    [3.8, 15000],
    [2.9, 8000],
    [3.5, 22000],
    [3.1, 12000],
    [3.9, 18000],
    [2.7, 6000],
])

y = ([1, 0, 1, 0, 1, 0])
print("Original Data:")
print(x)

# step 1 split first 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.33, random_state=42)

print("\n x_train:")
print(x_train)
print("\n x_test:")
print(x_test)

# step 2 fit scalar on training data only

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\n x_train after scaling")
print(x_train_scaled.round(2))

print("\n what scaler learned from training data:")
print("Mean per feature:",scaler.mean_.round(2))
print("Std per feature:",scaler.scale_.round(2))

print("\n Train mean after scaling(should be ~0):")
print(x_train_scaled.mean(axis=0).round(2))

print("\n Train std after scaling(should be ~1):")
print(x_train_scaled.std(axis=0).round(2))

# step 3 new student comes in 
# must scale with same scaler before predicting

new_student = np.array([[3.6, 17000]])
new_student_scaled = scaler.transform(new_student)
print("\n New student original", new_student)
print("New student scaled ", new_student_scaled.round(2))