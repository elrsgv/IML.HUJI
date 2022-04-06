from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
#pio.renderers.default = 'png'
#pio.renderers.default = 'browser'


def test_univariate_gaussian():

    print("# Question 1 - Draw samples and print fitted model")
    mu, stddiv = 10, 1
    XsQ1 = np.random.normal(mu,stddiv,1000) # expected, std-div, size(can be tuple)
    estimator = UnivariateGaussian()
    estimator.fit(XsQ1)
    print("Estimated mean and variance of 1000 samples are:")
    print(f"({estimator.mu_}, {estimator.var_})")
    print()

    print("# Question 2 - Empirically showing sample mean is consistent")
    ms = np.arange(10,1010,10)
    abs_dist = []
    for m in ms:
        estimator.fit(XsQ1[0:m])
        abs_dist.append( abs(estimator.mu_ - mu) )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ms, y=abs_dist, name=r'$|\mu-\widehat\mu|$'))
    fig.update_layout(
        title=r"Question 2 - Empirically showing sample mean is consistent",
        xaxis_title = r"$m\text{ - number of samples}$",
        yaxis_title = r"$|\mu-\widehat\mu|$",
        width=1400, height = 400)
    fig.update_xaxes(range=[0, 1010])
    fig.update_yaxes(range=[0, (round(max(abs_dist)*10+0.5)/10)])
    #fig.show() # renderer="browser"
    fig.write_image(f"ex1/Q2.png", scale=1, width=1400, height=400)
    #fig.write_html('tmp2.html', auto_open=True) # the Latex is not supported well.
    print()

    print("# Question 3 - Plotting Empirical PDF of fitted model")
    pdf_samples = estimator.pdf(XsQ1)

    line_x   = np.linspace(round(min(XsQ1)-.5), round(max(XsQ1)+.5), 1000 )
    pdf_ideal = 1/(np.sqrt(2*np.pi)*stddiv)*np.exp(-(line_x-mu)**2/(2*stddiv**2)) # Bonus - not required
    pdf_empirical = 1/(np.sqrt(2*np.pi)*np.sqrt(estimator.var_))*np.exp(-(line_x-estimator.mu_)**2/(2*estimator.var_))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=line_x, y=pdf_ideal, mode='lines', marker=dict(color="lightgray"),
                             name=r'$\text{PDF}(\mathcal{N}\left(\mu,\sigma\right))$'))
    fig.add_trace(go.Scatter(x=XsQ1, y=pdf_samples, mode='markers', marker=dict(color="blue"),
                             name=r'$\text{PDF}(\{x_1,...,x_1000\})$'))
    fig.add_trace(go.Scatter(x=line_x, y=pdf_empirical, mode='lines', marker=dict(color="blue"),
                             name=r'$\text{PDF}(\mathcal{N}\left(\widehat\mu,\widehat\sigma\right))$'))
    fig.update_layout(
        title=r"Question 3 - Plotting Empirical PDF of fitted model",
        xaxis_title = r"$\text{sample space}$",
        yaxis_title = r"$\text{probability density}$")
    fig.write_image(f"ex1/Q3.png", scale=1, width=1400, height=400)
    #fig.write_html('tmp3.html', auto_open=True)  # the Latex is not supported well.
    print()

    # Bonus that was not in the questions:
    # ------------------------------------
    #a = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #              -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    #l = UnivariateGaussian.log_likelihood(1, 1, a)

# -----------------------------------------------------------------------------------------
def test_multivariate_gaussian():

    print("# Question 4 - Draw samples and print fitted model")
    mu = np.array([0.0, 0.0, 4.0, 0.0])
    sigma = np.array([ [1.0, 0.2, 0.0, 0.5], [0.2, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.0, 1.0]])
    #print("mu is ") ; print(mu) ; print ("Sigma is ") ; print(sigma)
    XsQ1 = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian()
    estimator.fit(XsQ1)
    print("Estimated mu vector is ")
    print(estimator.mu_)
    print("Estimated Covariance (Sigma) matrix is ")
    print(estimator.cov_)
    print()

    print("# Question 5 - Likelihood evaluation")
    f1 = np.linspace(-10, 10, 200)
    f3 = f1
    heatmap = np.zeros((200,200))
    for f1i in range(np.size(f1)):
        if (f1i % 10) == 0: print("..." + str(f1i), end="") # Track run-time progress
        for f3i in range(np.size(f3)):
            mu = [ f1[f1i] , 0 , f3[f3i] , 0 ]
            loglike = MultivariateGaussian.log_likelihood(mu, sigma, XsQ1)
            heatmap[f3i,f1i] = loglike
    print()
    fig = go.Figure(data=go.Heatmap(z=heatmap,x=f1,y=f3))
    fig.update_layout(
        title=r"$\text{Question 5 - Likelihood evaluation} \mu=[f_1,0,f_3,0]$",
        xaxis_title=r"$f_1$",
        yaxis_title=r"$f_3$")
    fig.write_image(f"ex1/Q5.png", scale=1, width=800, height=600)
    print()

    print("# Question 6 - Maximum likelihood ")
    # get the argmax in this 2D array
    index_tuple = np.unravel_index(np.argmax(heatmap), np.shape(heatmap))
    # get the f1 and f3 values of the arg-max point we found
    f1_hat = np.around(f1[index_tuple[1]], decimals=3)
    f3_hat = np.around(f1[index_tuple[0]], decimals=3)
    print(f"(f1,f3) = ({f1_hat}, {f3_hat})")
    print()

    # Bonus that was not in the questions:
    #estimator.pdf(XsQ1)

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
