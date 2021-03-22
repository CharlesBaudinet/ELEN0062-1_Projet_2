from matplotlib import pyplot as plt
from main import *


def graph(X, Y, x_label, y_label, title, path, bool_lambda):
    plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(X, Y)
    if bool_lambda is False:
        plt.savefig(path, dpi=200)
    plt.show()

def questionC(save=False):
    """
    Plot the expected error, the bayes model and the function without noise.
    :param save: To save or not the graph obtained
    """
    n_y = 100
    random_state = 5
    drawer = check_random_state(random_state)
    n_points = 1000
    y_noise_i = []

    X = np.sort(drawer.uniform(0, 2, n_points))
    y_without_noise = fct(X, noise=False)
    for point in range(n_points):
        y_noise_i.append(fct(X[point], noise=False) + drawer.normal(0, math.sqrt(0.1), n_y))

    y_noise = np.mean(y_noise_i, 1)
    plt.figure()
    plt.plot(X, y_noise, label="Bayes model")
    plt.plot(X, y_without_noise, color='r', label="Theoritical curve")
    plt.title("Residual Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    if save == True:
        plt.savefig("value of bayes", dpi=200)

    inter = []
    for k in range(len(y_without_noise)):
        inter.append(y_noise_i[k] - y_without_noise[k])
    res_err = np.mean(np.square(inter), 1)

    plt.figure()
    plt.xlabel("x")
    plt.plot(X, res_err, label="Empirical Residual error")
    plt.plot(X, res_err * 0 + 0.1, color='r', label="Theoritical Residual error")
    plt.legend()
    plt.show()
    if save == True:
        plt.savefig("Residual error", dpi=200)

def plot_OLS(X_pred, bias_model_squared_ols, variance_ols, expected_error):
    """
    Plot all results obtained for the OLS: Bias squared, Variance and expected error
    :param X_pred: X used to compute the bias, variance and expected error
    :param bias_model_squared_ols
    :param variance_ols
    :param expected_error
    """
    plt.figure()
    #############################################################################
    #############################################################################
    for m in range(0, int(len(variance_ols) / 2)):
        plt.plot(X_pred, bias_model_squared_ols[m], label="m =" + str(m))

    plt.legend()
    plt.title("Evolution of the squared bias")
    plt.xlabel("x")
    plt.ylabel("Squared bias")
    #plt.savefig("Evolution of the squared bias 1", dpi=200)
    plt.show()

    for m in range(int(len(variance_ols) / 2), len(variance_ols)):
        plt.plot(X_pred, bias_model_squared_ols[m], label="m =" + str(m))

    plt.legend()
    plt.title("Evolution of the squared bias")
    plt.xlabel("x")
    plt.ylabel("Squared bias")
    #plt.savefig("Evolution of the squared bias 2", dpi=200)
    plt.show()

    #############################################################################
    #############################################################################
    for m in range(0, int(len(variance_ols) / 2)):
        plt.plot(X_pred, variance_ols[m], label="m =" + str(m))

    plt.legend()
    plt.title("Evolution of the Variance")
    plt.xlabel("x")
    plt.ylabel("Variance")
    #plt.savefig("Evolution of the Variance 1", dpi=200)
    plt.show()


    for m in range(int(len(variance_ols) / 2), len(variance_ols)):
        plt.plot(X_pred, variance_ols[m], label="m =" + str(m))

    plt.legend()
    plt.title("Evolution of the Variance")
    plt.xlabel("x")
    plt.ylabel("Variance")
    #plt.savefig("Evolution of the Variance 2", dpi=200)
    plt.show()

    #############################################################################
    #############################################################################
    for m in range(0, int(len(variance_ols) / 2)):
        plt.plot(X_pred, expected_error[m], label="m =" + str(m))

    plt.legend()
    plt.title("Evolution of the Expected error")
    plt.xlabel("x")
    plt.ylabel("Expected error")
    #plt.savefig("Evolution of the expected error 1", dpi=200)
    plt.show()


    for m in range(int(len(variance_ols) / 2), len(variance_ols)):
        plt.plot(X_pred, expected_error[m], label="m =" + str(m))

    plt.legend()
    plt.title("Evolution of the Expected error")
    plt.xlabel("x")
    plt.ylabel("Expected error")
    #plt.savefig("Evolution of the expected error 2", dpi=200)
    plt.show()

def plot_mean_value(bias_model_squared, variance, expected_error):
    '''
    Plot the evolution of measures w.t.r to the model complexity
    :param bias_model_squared: Bias for each complexity m
    :param variance: Variance for each complexity m
    :param expected_error: Expected error for each complexity m
    :param plot_graph: Boolean variable to indicate plotting or not
    :return: If plot_graph is false, we return the mean value for each variable.
    '''

    mean_bias_squared = np.mean(bias_model_squared, 1)
    mean_variance = np.mean(variance, 1)
    mean_expected_error = np.mean(expected_error, 1)
    m = [0, 1, 2, 3, 4, 5]
    to_plot = [mean_bias_squared, mean_variance, mean_expected_error]
    name = ["mean bias squared", "mean variance", "mean expected error"]
    plt.figure
    for nb_to_plot in range(len(to_plot)):
        plt.plot(m, to_plot[nb_to_plot], label=name[nb_to_plot])
    plt.legend()
    plt.xlabel("m")
    plt.title("Evolution of measures w.t.r to the model complexity")
    #plt.savefig("mean_evolution ", dpi=200)
    plt.show()

def graph_bias_explanation(bias_model_squared, sec_term_var, X_pred):
    """
    :param bias_model_squared
    :param sec_term_var
    :param X_pred
    """
    plt.plot(X_pred, sec_term_var, label="Empirical mean over LS")
    plt.plot(X_pred, fct(noise=False, X=X_pred), label="Real value of the function")
    plt.plot(X_pred, bias_model_squared, label="Bias squared")
    plt.legend()
    plt.show()

def plot_proof_equivalence(X_real_ridge, bias_model_squared_ridge, bias_model_squared_ols, variance_ridge, variance_ols):
    """
    Plot to illustrate that the ridge bias for lambda = 0 is the same than the OLS
    :param X_real_ridge
    :param bias_model_squared_ridge
    :param bias_model_squared_ols
    :param variance_ridge
    :param variance_ols
    """
    bias_model_squared_ridge[0]
    bias_model_squared_ols[0]

    plt.plot(X_real_ridge, bias_model_squared_ridge[0], color='blue', label="Ridge with lambda = 0")
    plt.plot(X_real_ridge, bias_model_squared_ols[5], '--', color='red', label="OLS m = 5")
    plt.legend()
    plt.title("Bias squared similitude between OLS and Ridge for Lamda = 0")
    plt.show()

    plt.plot(X_real_ridge, variance_ridge[0], color='blue', label="Ridge with lambda = 0")
    plt.plot(X_real_ridge, variance_ols[5], '--', color='red', label="OLS m = 5")
    plt.legend()
    plt.title("Variance similitude between OLS and Ridge for Lamda = 0")
    plt.show()


def plot_ridge(bias_model_squared_ridge, variance_ridge, expected_error_ridge, n_lambda, X_real_ridge):
    """
    Plot all results obtained for the Ridge: Bias squared, Variance and expected error
    :param bias_model_squared_ridge:
    :param variance_ridge:
    :param expected_error_ridge:
    :param n_lambda:
    :param X_real_ridge: X used to compute the bias, variance and expected error
    """
    possible_lambda = np.linspace(0, 2, n_lambda)

    ### For bias
    for lam in range(n_lambda):
        plt.plot(X_real_ridge, bias_model_squared_ridge[lam],
                 label="Lambda = " + str(round(possible_lambda[lam], 3)))
    plt.legend()
    plt.title("Ridge Bias w.r.t x")
    #plt.savefig("Ridge Bias", dpi=200)
    plt.show()

    ### For Variance
    for lam in range(n_lambda):
        plt.plot(X_real_ridge, variance_ridge[lam], label="Lambda = " + str(round(possible_lambda[lam], 3)))
    plt.legend()
    plt.title("Ridge Variance w.r.t x")
    #plt.savefig("Ridge Variance", dpi=200)
    plt.show()

    ### For Expected error
    for lam in range(n_lambda):
        plt.plot(X_real_ridge, expected_error_ridge[lam], label="Lambda = " + str(round(possible_lambda[lam], 3)))
    plt.legend()
    plt.title("Ridge Expected error w.r.t x")
    #plt.savefig("Ridge Expected error", dpi=200)
    plt.show()

def graph_bias_explaination(bias_model_squared , sec_term_var, X_pred):
    """
    Plot the mean of the prediction, our bias squared and the real function. (Used to illustrate the bias obtained)
    :param bias_model_squared
    :param sec_term_var
    :param X_pred
    """
    plt.plot(X_pred.reshape(-1,1),sec_term_var, label = "Empirical mean over LS")
    plt.plot(X_pred, fct(noise = False, X = X_pred), label = "Real value of the function")
    plt.plot(X_pred,bias_model_squared, label = "Bias squared")
    plt.legend()
    #plt.savefig("Bias explanation", dpi=200)
    plt.show()

def compute_mean_ridge(bias_model_squared_ridge, variance_ridge, expected_error_ridge, n_lambda):
    """
    Plot the evolution of the Ridge measures w.t.r to Lambda
    :param bias_model_squared_ridge
    :param variance_ridge
    :param expected_error_ridge
    :param n_lambda
    """
    mean_lambda_bias = np.mean(bias_model_squared_ridge, 1)
    mean_lambda_variance = np.mean(variance_ridge, 1)
    mean_lambda_expected_error = np.mean(expected_error_ridge, 1)

    min_exp = np.min(mean_lambda_expected_error)
    index = np.where(mean_lambda_expected_error == min_exp)[0]

    possible_lambda = np.linspace(0, 2, n_lambda)

    plt.plot(possible_lambda, mean_lambda_bias, label="Bias")
    plt.plot(possible_lambda, mean_lambda_variance, label="Variance")
    plt.plot(possible_lambda, mean_lambda_expected_error, label="Expected error")

    add_label = "Best Lambda = " + str(np.round(possible_lambda[index][0], 3))
    plt.plot([possible_lambda[index]], [min_exp], marker='o', markersize=3, color="red", label=add_label)
    plt.legend()
    plt.xlabel("Lambda")
    plt.title("Evolution of measures w.t.r to Lambda")
    #plt.savefig("Ridge mean", dpi=200)
    plt.show()
