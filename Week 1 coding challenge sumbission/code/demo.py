from numpy import *

import pandas as pd

import matplotlib.pyplot as plt


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(m, b, data):
    totalError = 0.0
    numInstances = data.shape[0]
    for i in range(numInstances):
        adr = data[i, 0]  # Row 'i' column 'ADR'
        rating = data[i, 1]  # Row 'i' column 'Rating'

        # The real rating
        currentTarget = rating

        # Predicted rating with our current fitting line
        # y = mx + b
        currentOutput = m * adr + b

        # Compute squared error
        currentSquaredError = (currentTarget - currentOutput) ** 2

        # Add it to the total error
        totalError += currentSquaredError

    sse = totalError / numInstances

    return sse


def gradient_descent_step(m, b, data, learningRate):
    N = data.shape[0]
    m_grad = 0
    b_grad = 0

    for i in range(N):
        # Get current pair (x,y)
        x = data[i, 0]
        y = data[i, 1]

        # Partial derivative respect 'm'
        dm = -((2 / N) * x * (y - (m * x + b)))

        # Partial derivative respect 'b'
        db = - ((2 / N) * (y - (m * x + b)))

        # Update gradient
        m_grad = m_grad + dm
        b_grad = b_grad + db

    # Set the new 'better' updated 'm' and 'b'
    m_updated = m - learningRate * m_grad
    b_updated = b - learningRate * b_grad
    '''
    Important note: The value '0.0001' that multiplies the 'm_grad' and 'b_grad' is the 'learning rate', but it's a concept
    out of the scope of this challenge. For now, just leave that there and think about it like a 'smoother' of the learn, 
    to prevent overshooting, that is, an extremly fast and uncontrolled learning.
    '''

    return m_updated, b_updated


def predict(inter, slope):
    freedom = int(input("Enter % Of Freedom(0-100): "))
    if (100 > freedom < 0):
        print("Invalid Input")
    else:
        happiness = (slope * (freedom / 100) + inter) * 1000
        print("The predicted happiness is: {0}".format(happiness))
        predict(inter, slope)


def run():
    csv_data = pd.read_csv("2017.csv")
    csv_data.head()

    data = csv_data.as_matrix()

    m = 0
    b = 0
    learningRate = 0.0001
    error = compute_error_for_line_given_points(m, b, data)

    print('For the fitting line: y = %sx + %s\nSSE: %.2f' % (m, b, error))

    for i in range(1000):
        m, b = gradient_descent_step(m, b, data, learningRate)
        error = compute_error_for_line_given_points(m, b, data)
        print('At step %d - Line: y = %.6fx + %.6f - Error: %.6f' % (i + 1, m, b, error))

    print('\nBest  line: y = %.6fx + %.6f - Error: %.6f' % (m, b, error))

    plt.scatter(x=data[:, 0], y=data[:, 1])
    plt.xlabel("Freedom")
    plt.ylabel("Happiness Score")
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Freedom vs Happiness Score')
    ax.scatter(x=data[:, 0], y=data[:, 1], label='Data')
    plt.plot(data[:, 0], m * data[:, 0] + b, color='red', label='Our Fitting Line')
    ax.set_xlabel('Freedom')
    ax.set_ylabel('Happiness Score')
    ax.legend(loc='best')

    plt.show()

    predict(b, m)


if __name__ == '__main__':
    run()
