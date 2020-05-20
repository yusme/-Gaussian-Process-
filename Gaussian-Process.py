import numpy as np
import matplotlib.pylab as plt

#Gaussian Process Models:
# the kernel function encode Prior Knowledge  concerning the correlation between
# the component of a difference point
# mesure of proximity between the member X, also K define the function Space within with the search of the solution
# take place.
def exponential_cov(x, y, params):
    return params[0] * np.exp(-0.5 * params[1] * np.subtract.outer(x, y) ** 2)


# this depend of the kernel that we want to choose to describe the type of covariance expected in the dataset
# def exponential_cov(x,y, params):
# num = np.subtract(x, y)
# return params[0] * np.exp(-(0.5 * params[1])*num**2)


# sampling from a Gaussian Process
# sampling from gaussian prior process p(x|y)
def conditional(x_new, x, y, params):
    A = exponential_cov(x_new, x_new, params)
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)

    mean = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))

    return mean, sigma


# we Update our Belief about X by Calculate the Posterior
def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]  # kernel ist the function exponential_cov
    Sinv = np.linalg.inv(sigma)
    y_predict = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)

    return y_predict, sigma_new


def plotGaussian():
    theta = [1, 8]
    sigma_0 = exponential_cov(0, 0, theta)

    xpts = np.arange(-3, 3, step=0.01)
    # plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0, alpha=0.3)


    # new Data: task predict the y_new and new Sigma
    x = [1.0]
    y = [np.random.normal(scale=sigma_0)]

    # calculate the Kernel for X_new
    sigma_1 = exponential_cov(x, x, theta)
    x_pred = np.linspace(-3, 3, 1000)

    # with the New_sigma, I can predict the y_new and new_sigmas  for  all x_data from -3 to 3_pred
    predictions = [predict(i, x, exponential_cov, theta, sigma_1, y) for i in x_pred]
    y_pred, sigmas = np.transpose(predictions)

    print "sigmas", sigma_1

    # plt.errorbar(x_pred, y_pred, yerr = sigmas, capsize=0,  alpha=0.3)


    #  new Data - two point
    x_new = -0.7
    m, s = conditional([x_new], x, y, theta)
    y_new = np.random.normal(m, s)

    x.append(x_new)
    y.append(y_new)

    # Predict my new_sigma for X_new
    sigma_2 = exponential_cov(x, x, theta)
    print "sigma_2", sigma_2

    print "x,y", x, y

    predictions = [predict(i, x, exponential_cov, theta, sigma_2, y) for i in x_pred]
    y_pred, sigmas = np.transpose(predictions)

    #  new Data - many point
    x_many = [-2.1, -1.5, 0.3, 1.8, 2.5]
    m_many, s_many = conditional(x_many, x, y, theta)
    y_many = np.random.multivariate_normal(m_many, s_many)

    print "x_many", len(x_many), x_many
    print "y_many", len(y_many), y_many

    x += x_many
    y += y_many.tolist()

    # calculate the kernel for the new Data
    # Predict my new_sigma for X_new
    kernel_many = exponential_cov(x, x, theta)

    # predict the new Sigmas
    predictions = [predict(i, x, exponential_cov, theta, kernel_many, y) for i in x_pred]
    y_pred, sigmas = np.transpose(predictions)

    plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0, alpha=0.3)
    v = [-3.0, 3.0, 3.0, -3.0]
    plt.axis(v)
    plt.show()


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def main():
    print 'doing main'
    #plot2()
    plotGaussian()
    print 'end plot'



if __name__ == '__main__':
    main()
