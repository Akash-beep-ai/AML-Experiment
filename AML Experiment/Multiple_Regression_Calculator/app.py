from flask import Flask,render_template,request
import numpy as np,pandas as pd,io,base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/calculate",methods=["POST"])
def calculate():
    y=np.array([float(v) for v in request.form["y"].split(",")])
    x1=np.array([float(v) for v in request.form["x1"].split(",")])
    x2=np.array([float(v) for v in request.form["x2"].split(",")])
    x3=np.array([float(v) for v in request.form["x3"].split(",")])
    X=pd.DataFrame({"Study Hours":x1,"Attendance":x2,"Assignment":x3})
    model=LinearRegression().fit(X,y)
    pred=model.predict(X)
    mae=mean_absolute_error(y,pred)
    mse=mean_squared_error(y,pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y,pred)
    plt.figure()
    plt.scatter(y,pred)
    plt.plot([min(y),max(y)],[min(y),max(y)],'r--')
    buf=io.BytesIO()
    plt.savefig(buf,format="png",bbox_inches="tight")
    plt.close()
    graph=base64.b64encode(buf.getvalue()).decode()
    # Performance Metrics Graph
    # Performance Metrics Graph
    plt.figure(figsize=(7,5))

    metrics = ["MSE", "MAE", "RMSE", "R²"]
    values = [mse, mae, rmse, r2]
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

    bars = plt.bar(metrics, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.title("Multiple Linear Regression Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", bbox_inches="tight")
    plt.close()

    graph2 = base64.b64encode(buf2.getvalue()).decode()

    equation = (
        f"Y = {model.intercept_:.4f} + "
        f"({model.coef_[0]:.4f})×Study Hours + "
        f"({model.coef_[1]:.4f})×Attendance + "
        f"({model.coef_[2]:.4f})×Assignment"
    )

    return render_template(
        "result.html",
        equation=equation,
        mae=f"{mae:.4f}",
        mse=f"{mse:.4f}",
        rmse=f"{rmse:.4f}",
        r2=f"{r2:.4f}",
        graph=graph,
        graph2=graph2
    )

if __name__ == "__main__":
    app.run(debug=True)