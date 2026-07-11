# AML-Experiment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 4])

n = len(x)

mean_x = np.mean(x)
mean_y = np.mean(y)


numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x)**2)

if denominator == 0:
    b1 = 0
else:
    b1 = numerator / denominator

b0 = mean_y - b1 * mean_x

print(f"Regression line coefficients:")
print(f"  Slope (b1): {b1:.2f}")
print(f"  Intercept (b0): {b0:.2f}")

y_pred = b0 + b1 * x


mae = np.mean(np.abs(y - y_pred))


mse = np.mean((y - y_pred)**2)


rmse = np.sqrt(mse)


ss_total = np.sum((y - mean_y)**2)
ss_residual = np.sum((y - y_pred)**2)

if ss_total == 0:
    r_squared = 1.0 if ss_residual == 0 else 0.0
else:
    r_squared = 1 - (ss_residual / ss_total)

print("\nEvaluation Metrics:")
print(f"  MAE: {mae:.2f}")
print(f"  MSE: {mse:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R-squared: {r_squared:.2f}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, s=100, label='Original Data')
sns.lineplot(x=x, y=y_pred, color='red', label=f'Regression Line (y = {b0:.2f} + {b1:.2f}x)')
plt.title('Linear Regression with Least Squares Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
