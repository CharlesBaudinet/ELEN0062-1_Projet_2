from plot_file import *
from random import *
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from scipy.integrate import nquad
from math import pi as pi
from math import exp
from math import sqrt


def integral_computation():
    rho_plus = 0.75
    rho_moins = -rho_plus

    def proba_y_moins_1(x1, x2):
        """
        :param x1
        :param x2
        :return: Return the normal probability associated to x1 and x2 for rho < 0
        """
        first_term = 1 / (2 * pi * sqrt((1 - rho_moins ** 2)))
        a = 1 / (1 - rho_moins ** 2)
        second_term_int = a * (x1 ** 2 + x2 ** 2 - 2 * rho_moins * x1 * x2)
        second_term = exp((-1 / 2) * second_term_int)
        return first_term * second_term

    def proba_y_plus_1(x1, x2):
        """
        :param x1
        :param x2
        :return: Return the normal probability associated to x1 and x2 for rho > 0
        """

        first_term = 1 / (2 * pi * sqrt((1 - rho_plus ** 2)))
        a = 1 / (1 - rho_plus ** 2)
        second_term_int = a * (x1 ** 2 + x2 ** 2 - 2 * rho_plus * x1 * x2)
        second_term = exp((-1 / 2) * second_term_int)
        return first_term * second_term

    ## Error on P(x with y == 1) first
    first_term_plus1_1 = nquad(proba_y_plus_1, [[0, np.inf], [-np.inf, 0]])

    ## Error on P(x with y == 1) second
    first_term_plus1_2 = nquad(proba_y_plus_1, [[-np.inf, 0], [0, np.inf]])

    ## Error on P(x with y == -1) first
    first_term_moins1_1 = nquad(proba_y_moins_1, [[-np.inf, 0], [-np.inf, 0]])

    ## Error on P(x with y == -1) second
    first_term_moins1_2 = nquad(proba_y_moins_1, [[0, np.inf], [0, np.inf]])

    percentage = 1 / 2 * (
                first_term_plus1_1[0] + first_term_plus1_2[0] + first_term_moins1_1[0] + first_term_moins1_2[0])

    print("Integral computation: " + str(round(percentage,5)))

def empirical_computation():
    """
    Compute the empirical error by generating 10**6 samples and apply our Bayes condition on it.
    """
    rho_plus = 0.75
    mean = [0, 0]
    cov_plus = [[1, rho_plus], [rho_plus, 1]]

    x_positive = np.random.multivariate_normal(mean, cov_plus, size=10 ** 6)

    cpt_y_true = 0
    cpt_y_false = 0
    for x in x_positive:
        if x[0] * x[1] > 0:
            cpt_y_true += 1
        else:
            cpt_y_false += 1
    perc_emp = cpt_y_false / (cpt_y_true + cpt_y_false)
    print("Empirical percentage: " + str(round(perc_emp,5)))

def generate_data(interv_x=[0, 2], n_points=100, n_ls=10, µ_epsilon=0, sig_epsilon_square=0.1, random_state=5):
    """
    :param interv_x: Interval where x is defined
    :param n_points: Number of points we will generate for each learning sample
    :param n_ls: Number of learning samples
    :param µ_epsilon: mean of the error
    :param sig_epsilon_square: Variance of the error
    :param random_state: Seed
    :return: X_ls: Contains all samples x. y_ls contains all y associated to x_ls
    """
    drawer = check_random_state(random_state)
    X_ls = []
    y_ls = []
    for n in range(n_ls):
        X = drawer.uniform(interv_x[0], interv_x[1], n_points)
        X = np.sort(X)
        y_ls.append(fct(X))

        new_x = []
        for x in X:
            new_x.append([x**0, x, x ** 2, x ** 3, x ** 4, x ** 5])
        X_ls.append(new_x)
    return X_ls, y_ls


def fct(X, noise=True, random_state=5):
    """
    :param X
    :param noise: If we want to add some noise or not
    :param random_state: seed
    :return: the value of the function (with noise or not) associated to my X
    """
    if noise == False:
        return -pow(X, 3) + 3 * pow(X, 2) - 2 * pow(X, 1) + 1
    else:
        µ_epsilon = 0
        sig_epsilon = math.sqrt(0.1)

        drawer = check_random_state(random_state)
        noise_e = drawer.normal(0, sig_epsilon, X.shape)

        return -pow(X, 3) + 3 * pow(X, 2) - 2 * pow(X, 1) + 1 + noise_e


def generate_set_prediction(interv_x=[0, 2], n_points=100, random_state=5):
    '''

    :param interv_x: Interval where x is defined
    :param n_points: Number of points we will generate
    :param random_state: seed
    :return:
    '''
    drawer = check_random_state(random_state)
    X = drawer.uniform(interv_x[0], interv_x[1], n_points)
    X = np.sort(X)
    return X, fct(X, noise=True)

def main_function_OLS(plot_pred_m = False):
    '''
    Computation of the Bias, Variance and Expected error.
    :return: Bias, Variance, Expected error
    '''
    nb_m = 6  # Number of m
    interv_x = [0, 2]  # Interval where x is defined
    n_points = 30  # Number points we have in each learning sample
    noise_µ = 0  # Mean of noise
    noise_std = 0.1  # Standard deviation of noise
    n_ls = 100  # Number of learning samples
    random_state = 13  # Seed
    n_points_real = 500 # Number of points used to plot the results

    # Generate n_ls learning samples of n_points each. We then associate the y value corresponding for each x
    X_samples, y_samples = generate_data(interv_x=interv_x, n_points=n_points, n_ls=n_ls, random_state=random_state)

    # Generate x values that will be used to compute the measures and to plot the results
    X_real, y_real = generate_set_prediction(interv_x=interv_x, n_points=n_points_real, random_state=random_state)

    # Trick used to use non linear variables in some kind of linear formulation.
    X_real_new = [X_real**0, X_real, X_real ** 2, X_real ** 3, X_real ** 4, X_real ** 5]
    X_real_df = pd.DataFrame(data=X_real_new)
    X_real_df = X_real_df.transpose()

    bias_squared_m = []
    variance_m = []
    expected_error_m = []

    sec_term_var = []

    ## For each complexity m
    for m in range(nb_m):
        prediction = []

        ## For each Learning sample X_samples[n]
        if plot_pred_m:
            plt.figure()

        for n in range(n_ls):
            X_ls = X_samples[n]
            X_ls = pd.DataFrame(data=X_ls)
            X_ls = X_ls.transpose()
            X_ls = X_ls[0:m + 1]
            X_ls = X_ls.transpose()
            y = pd.DataFrame(data=y_samples[n])
            inter = X_real_df.transpose()
            inter = inter[0:m + 1]
            X_real_m = inter.transpose()
            reg = LinearRegression().fit(X_ls, y)
            prediction.append(reg.predict(X_real_m).reshape(-1, 1)) # Preditions on my X_real value

            if plot_pred_m:
                y_plot = reg.predict(X_real_m).reshape(-1, 1)
                plt.plot(X_real_new[1], y_plot)

        if plot_pred_m:
            plt.title("Functions predicted for m = " + str(m))
            plt.xlabel('x')
            plt.ylabel('prediction y')
            plt.show()



        ##### Compuation of the Bias:
        hb = fct(X_real, noise=False).reshape(-1, 1)
        sec_i = np.mean(prediction, 0)
        bias_squared_m.append(np.square(hb - sec_i))

        ##### Computation of the Variance:
        sec_var = np.mean(prediction, 0)
        inter_sq = np.square(prediction - sec_var)
        var = np.mean(inter_sq, 0)
        variance_m.append(var)

        sec_term_var.append(sec_var)
        ##### Expected error
        expected_error_m.append(variance_m[-1] + bias_squared_m[-1])

    return bias_squared_m, variance_m, expected_error_m, X_real, nb_m, sec_term_var


def main_function_Ridge():
    interv_x = [0, 2]  # Interval where x is defined
    n_points = 30  # Number points we have in each learning sample
    n_points_real = 500
    noise_µ = 0  # Mean of noise
    noise_std = 0.1  # Standard deviation of noise
    n_ls = 100  # Number of points used to plot the results
    n_lambda = 4 # Number of different lambda
    random_state = 13 # Seed used
    # Generate n_ls learning samples of n_points each. We then associate the y value corresponding for each x
    X_samples, y_samples = generate_data(interv_x=interv_x, n_points=n_points, n_ls=n_ls, random_state=random_state)

    # Generate x values that will be used to compute the measures and to plot the results
    X_real, y_real = generate_set_prediction(interv_x=interv_x, n_points=n_points_real, random_state=random_state)
    X_real_new = [X_real ** 0, X_real, X_real ** 2, X_real ** 3, X_real ** 4, X_real ** 5]
    X_real_df = pd.DataFrame(data=X_real_new)
    X_real_df = X_real_df.transpose()

    bias_model_squared = []
    variance = []
    expected_error = []
    possible_lambda = np.linspace(0, 2, n_lambda)
    sec_var = []

    for lamb in possible_lambda:
        prediction = []
        for n in range(n_ls):
            X_ls = X_samples[n]
            X_ls = pd.DataFrame(data=X_ls)
            y = pd.DataFrame(data=y_samples[n])
            inter = X_real_df.transpose()
            X_real_m = inter.transpose()
            reg = Ridge(alpha=lamb).fit(X_ls, y)
            prediction.append(reg.predict(X_real_m))

        ##### Compuation of the Bias:
        hb = fct(X_real, noise=False).reshape(-1, 1)
        sec_i = np.mean(prediction, 0)
        bias_model_squared.append(np.square(hb - sec_i))

        ##### Computation of the Variance:
        sec = np.mean(prediction, 0)
        inter_sq = np.square(prediction - sec)
        var = np.mean(inter_sq, 0)
        variance.append(var)

        sec_var.append(sec)

        ##### Expected error
        expected_error.append(variance[-1] + bias_model_squared[-1])

    return bias_model_squared, variance, expected_error, X_real, n_lambda, [sec_var]



if __name__ == '__main__':

    #### Question 1.1 b ######
    compute_percentage = True

    #### Question 2.c ######
    plot_question_3 = False

    #### Question 2.d ######
    plot_results_ols = False

    #### Question 2.e ######
    plot_mean_value_bool = False

    #### Question 2.f ######
    plot_results_ridge = False
    plot_mean_ridge = False

    ### Additional plots
    plot_pred_m = False  # Plot the predictions for each m (used to illustrate the variance between predictions)
    plot_bias_explanation = False  # Plot the mean of the prediction, our bias squared and the real function. (Used to
                                    # illustrate the bias obtained
    plot_equivalence = False  # Plot to illustrate that the ridge bias for lambda = 0 is the same than the OLS


    #### Analytical Parts ############
    if compute_percentage:
        integral_computation()
        empirical_computation()


    if plot_question_3:
        questionC(False)

    ###########  OLS part ############
    bias_model_squared_ols, variance_ols, expected_error_ols, X_pred, nb_m, sec_var_OLS = main_function_OLS(plot_pred_m)

    if plot_results_ols:
        plot_OLS(X_pred, bias_model_squared_ols, variance_ols, expected_error_ols)

    if plot_bias_explanation:
        m = 2
        graph_bias_explanation(bias_model_squared_ols[m], sec_var_OLS[m], X_pred)

    if plot_mean_value_bool:
        plot_mean_value(bias_model_squared_ols, variance_ols, expected_error_ols)

    ###########  Ridge part ############

    bias_model_squared_ridge, variance_ridge, expected_error_ridge, X_real_ridge, n_lambda, for_visualization_ridge = main_function_Ridge()

    if plot_results_ridge:
        plot_ridge(bias_model_squared_ridge, variance_ridge, expected_error_ridge, n_lambda, X_real_ridge)

    if plot_equivalence:
        plot_proof_equivalence(X_real_ridge, bias_model_squared_ridge, bias_model_squared_ols, variance_ridge, variance_ols)

    if plot_mean_ridge:
        compute_mean_ridge(bias_model_squared_ridge, variance_ridge, expected_error_ridge, n_lambda)

