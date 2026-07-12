import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np

points = []


# ADD POINT

def add_point():
    try:
        x = float(x_entry.get())
        y = float(y_entry.get())

        points.append((x, y))

        point_text.set(
            "   ".join(
                f"({x}, {y})"
                for x, y in points
            )
        )

        x_entry.delete(0, tk.END)
        y_entry.delete(0, tk.END)

    except ValueError:
        messagebox.showerror(
            "Error",
            "Enter valid X and Y values"
        )


# LINEAR REGRESSION

def regression():
    if len(points) < 2:
        messagebox.showerror(
            "Error",
            "Add at least 2 points"
        )
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    n = len(x)

    m = (
        n * np.sum(x * y)
        - np.sum(x) * np.sum(y)
    ) / (
        n * np.sum(x ** 2)
        - np.sum(x) ** 2
    )

    c = (
        np.sum(y) - m * np.sum(x)
    ) / n

    return x, y, m, c


# COMPUTE RESULT

def compute():
    result = regression()

    if result is None:
        return

    x, y, m, c = result

    y_pred = m * x + c

    mse = np.mean((y - y_pred) ** 2)

    rmse = np.sqrt(mse)

    mae = np.mean(abs(y - y_pred))

    r2 = 1 - (
        np.sum((y - y_pred) ** 2)
        / np.sum((y - np.mean(y)) ** 2)
    )

    result_text.set(
        f"M (Slope): {m:.4f}\n"
        f"C (Intercept): {c:.4f}\n"
        f"Equation: Y = {m:.4f}x + {c:.4f}\n\n"
        f"Model Evaluation\n\n"
        f"MSE: {mse:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"R² Score: {r2:.4f}"
    )


# VISUALIZE GRAPH

def visualize():
    result = regression()

    if result is None:
        return

    x, y, m, c = result

    y_pred = m * x + c

    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs(y - y_pred))

    r2 = 1 - (
        np.sum((y - y_pred) ** 2)
        / np.sum((y - np.mean(y)) ** 2)
    )

    # REGRESSION GRAPH

    order = np.argsort(x)

    plt.figure()

    plt.scatter(
        x,
        y,
        label="Actual Data"
    )

    plt.plot(
        x[order],
        y_pred[order],
        label="Regression Line"
    )

    plt.title("Linear Regression Graph")

    plt.xlabel("X Value")

    plt.ylabel("Y Value")

    plt.legend()

    plt.grid(True)

    plt.show()


    # MODEL METRICS GRAPH

    metrics = [
        "MSE",
        "RMSE",
        "MAE",
        "R²"
    ]

    values = [
        mse,
        rmse,
        mae,
        r2
    ]

    plt.figure()

    plt.bar(
        metrics,
        values
    )

    plt.title(
        "Model Evaluation Graph"
    )

    plt.xlabel(
        "Evaluation Metric"
    )

    plt.ylabel(
        "Metric Value"
    )

    plt.grid(True)

    plt.show()


# RESET

def reset():
    points.clear()

    x_entry.delete(0, tk.END)

    y_entry.delete(0, tk.END)

    point_text.set(
        "No points added yet."
    )

    result_text.set("")


# GUI WINDOW

root = tk.Tk()

root.title(
    "Linear Regression Calculator"
)

root.geometry("650x650")


title = tk.Label(
    root,
    text="Linear Regression Calculator",
    font=("Arial", 20, "bold")
)

title.pack(pady=20)


# INPUT FRAME

input_frame = tk.Frame(root)

input_frame.pack(pady=10)


tk.Label(
    input_frame,
    text="X Value"
).grid(row=0, column=0)


x_entry = tk.Entry(
    input_frame
)

x_entry.grid(
    row=1,
    column=0,
    padx=10
)


tk.Label(
    input_frame,
    text="Y Value"
).grid(row=0, column=1)


y_entry = tk.Entry(
    input_frame
)

y_entry.grid(
    row=1,
    column=1,
    padx=10
)


add_button = tk.Button(
    input_frame,
    text="Add",
    command=add_point
)

add_button.grid(
    row=1,
    column=2,
    padx=10
)


# POINT LIST

point_text = tk.StringVar()

point_text.set(
    "No points added yet."
)


point_label = tk.Label(
    root,
    textvariable=point_text
)

point_label.pack(pady=15)


# BUTTONS

button_frame = tk.Frame(root)

button_frame.pack(pady=10)


tk.Button(
    button_frame,
    text="Compute Regression",
    command=compute
).grid(
    row=0,
    column=0,
    padx=5
)


tk.Button(
    button_frame,
    text="Visualize",
    command=visualize
).grid(
    row=0,
    column=1,
    padx=5
)


tk.Button(
    button_frame,
    text="Reset",
    command=reset
).grid(
    row=0,
    column=2,
    padx=5
)


# RESULT

result_text = tk.StringVar()


result_label = tk.Label(
    root,
    textvariable=result_text,
    font=("Arial", 12),
    justify="left"
)

result_label.pack(
    pady=25
)


root.mainloop()