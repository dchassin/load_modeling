import marimo

__generated_with = "0.1.35"
app = marimo.App(width="full", layout_file="layouts/fit_model.grid.json")


@app.cell
def __(mo):
    mo.md("""
    # Predicting load using SCADA and AMI load data
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    This Marimo notebook illustrates how to fit a z-domain transfer function to load data based on temperature and use the transfer function to predict load.
    """)
    return


@app.cell
def __(mo):
    mo.md("First, load your data. The data must include both the input $x(t)$ and output $y(t)$ you wish to use for your transfer function, as well as the time-index, i.e.,")

    return


@app.cell
def __(np, pd):
    pd.DataFrame({"t":pd.date_range("2020-01-01 00:00:00","2020-01-01 03:00:00",freq="1h"),"x":np.random.normal(0,1,4),"y":np.random.normal(0,1,4)})
    return


@app.cell
def __(mo):
    mo.md("""Now let's load a sample data set:""")
    return


@app.cell
def __(pd):
    data = pd.read_csv("data.csv", index_col=[0], parse_dates=[0]).resample("1h").asfreq("1h").fillna(method="ffill")
    columns = [data.index.name]
    columns.extend(data.columns)
    data[0:5]
    return columns, data


@app.cell
def __(columns, mo):
    time_field = mo.ui.dropdown(columns, value = columns[0], label = "First, you must choose the time index:")
    time_field
    return time_field,


@app.cell
def __(columns, mo):
    output_field = mo.ui.dropdown(columns, value = columns[1], label = "Then, you must choose the output values:")
    output_field
    return output_field,


@app.cell
def __(columns, mo):
    input_field = mo.ui.dropdown(columns, value = columns[2], label = "Finally, you must choose the input values:")
    input_field
    return input_field,


@app.cell
def __(data, input_field, output_field, plot_data):
    fig1 = plot_data(data,[output_field.value,input_field.value])
    # mo.mpl.interactive(fig1)
    fig1
    return fig1,


@app.cell
def __(mo):
    mo.md("""
    Once you have identified the data columns you wish to model, you must select the model order.  For most load models, you should choose a model order that covers a little more than 24 hours.  The exact order is the one that minimizes the errors.  Try adjusting the order to find the best model that fits the data.
    """)
    return


@app.cell
def __(get_order_value, mo, set_order_value):
    order = mo.ui.slider(1,52,value=get_order_value(),debounce=True,on_change=set_order_value)
    order
    return order,


@app.cell
def __(mo):
    get_order_value, set_order_value = mo.state(1)
    mo.md("""Or you can search for the fit that minimizes the mean error.""")
    return get_order_value, set_order_value


@app.cell
def __(mo, set_order_value):
    def auto_fit_order():
        # TODO: search for the best value
        return 26
    auto_fit = mo.ui.button(
        on_click = lambda x: set_order_value(auto_fit_order()),
        label = "Find best order"
    )
    auto_fit
    return auto_fit, auto_fit_order


@app.cell
def __(data, fit_tf, input_field, mo, order, output_field, polystr):
    tf = fit_tf(input_field.value,output_field.value,data,order.value)
    mo.md(f"""
    The discrete-time transfer function is 

    $\\frac{{P(z)}}{{T(x)}} = \\frac{{{polystr(tf['num'],'z')}}}{{{polystr(tf['den'],'z')}}}$
    """)
    return tf,


@app.cell
def __(mo):
    mo.md("""
    Now you can evaluate the performance of the model on a test data set.
    """)
    return


@app.cell
def __(
    input_field,
    output_field,
    pd,
    plot_data,
    predict_tf,
    tf,
    time_field,
):
    test = pd.read_csv("test.csv", index_col=["t"], parse_dates=[0])
    test = test.join(predict_tf(tf,test,"F").set_index(time_field.value),on=time_field.value)
    fig2 = plot_data(test,[output_field.value,"F",input_field.value]) 
    #mo.mpl.interactive(fig2) 
    fig2
    return fig2, test


@app.cell
def __(mo, order):
    mo.md(f"""
    The ${order.value}$-order model errors are as follows:
    """)
    return


@app.cell
def __(pd, test):
    err = (test["P"]-test["F"])
    MAE = err.abs().mean()
    MAPE = (err.abs()/test["P"]*100).mean()
    ME = err.mean()
    SD = err.std()
    pd.DataFrame(
        data = {"Error" : [
            f"{MAE.round(2):.2f} MW", 
            f"{MAPE.round(2):.1f}%", 
            f"{ME.round(2):.2f} MW", 
            f"{SD.round(2):.2f} MW",
            ]},
        index = ["MAE","MAPE","ME","SD"],
        )
    return MAE, MAPE, ME, SD, err


@app.cell
def __(ME, SD, err, norm, np, plt):
    nbins = 20
    bin_width = (err.max()-err.min())/nbins
    count,bins = np.histogram(err.tolist(),bins=np.arange(ME-4*SD,ME+4*SD,bin_width))
    plt.plot(bins[:-1],count/sum(count)/bin_width,"+")
    plt.plot(bins,norm(ME,SD).pdf(bins))
    plt.xlabel("Error (MW)")
    plt.ylabel("Probability")
    plt.grid('on')
    plt.ylim([0,1])
    plt.gca()
    return bin_width, bins, count, nbins


@app.cell
def __(np):
    def fit_tf(input,output,data,order):
        """Compute the best fit z-domain transfer function for the data.

        input - Input name (str)
        output - Output name (str)
        data - Data (pandas.DataFrame)
        order - Model order (int)
        """
        X = np.matrix(data[input]).transpose()
        Y = np.matrix(data[output]).transpose()
        K = order
        L = len(Y)
        M = np.hstack([np.hstack([Y[n+1:L-K+n] for n in range(K)]),
                       np.hstack([X[n+1:L-K+n] for n in range(K+1)])])
        Mt = M.transpose()
        x = np.linalg.solve(Mt*M,Mt*Y[K+1:])
        a = x.transpose().round(4).tolist()[0][0:K]
        a.insert(0,1.0)
        b = x.transpose().round(4).tolist()[0][K:]
        return {"input":input,"output":output,"den":a,"num":b,"time":data.index.name,"order":K}

    return fit_tf,


@app.cell
def __(np, pd):
    def predict_tf(model,data,name=None):
        X = np.matrix(data[model["input"]]).transpose()
        Y = np.matrix(data[model["output"]]).transpose()
        L = len(Y)
        K = model["order"]
        M = np.hstack([np.hstack([Y[n+1:L-K+n] for n in range(K)]),
                       np.hstack([X[n+1:L-K+n] for n in range(K+1)])])
        x = model["den"][1:]
        x.extend(model["num"])
        return pd.DataFrame(data={name if name else model["output"]:(M@x).tolist()[0],data.index.name:data.index[K+1:].tolist()})
    return predict_tf,


@app.cell
def __():
    def polystr(p,v='x'):
        """Format a polynomial

        p - Polynomial (list)
        v - Variable (str)
        """
        return f'{p[0]:g}' + ''.join([f'{z:+g}{v}^{{{-n-1:d}}}' for n,z in enumerate(p[1:])])
    return polystr,


@app.cell
def __(plt):
    def plot_data(data,axes,ylabel=None):
        """Plot data

        data - Data (pandas.DataFrame)
        axes - Axes to plot (list of str)
        ylabel - Y-axis label (str, default is None)
        """
        plt.figure(figsize=(10,5))
        for axis in axes:
            plt.plot(data[axis],label=axis)
            plt.xlabel(data.index.name)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid('on')
            plt.gcf().autofmt_xdate()
        return plt.gcf()
    return plot_data,


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import datetime as dt
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    return dt, mo, norm, np, pd, plt


if __name__ == "__main__":
    app.run()
