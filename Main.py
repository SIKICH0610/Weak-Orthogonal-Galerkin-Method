# coding by zhaolong deng
import numpy as np
import sympy as sym
import sympy as sp
from scipy.stats import qmc
import itertools
from scipy.special import eval_hermitenorm
from itertools import product
from scipy.sparse.linalg import cg
from itertools import combinations_with_replacement

def covariance_matrix(random_variable_vec, correlation_matrix):
    '''Generating the covariance matrix, return a np.matrix'''
    # random_varaible_vec(np.array/vector): a 1-N vector that contains several lists which contain the standard division of the random variable
    # correlation_matrix(np.matrix) : a N-N matrix that arrange the correlation of given number of random variables, following the graded lexicographic order
    assert len(random_variable_vec) == correlation_matrix.shape[0]
    assert np.array_equal(correlation_matrix, correlation_matrix.T)
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            correlation_matrix[i, j] = correlation_matrix[i, j] * random_variable_vec[i] * random_variable_vec[j]
        # The 1 here since that we are using normal distribution, which is the \sigma
    return correlation_matrix


def qmc_integration(func, variables, lower_limit, upper_limit, num_samples):
    '''Computing the expectation of a given function, which is a bit faster than monte_carlo_integration,
    return a float value'''
    # func(sym.function): any function that is bounded and can be interation over L^2
    # num_samples(int): size of trails, produce enough precession digits
    dim = len(variables)
    if len(lower_limit) != dim or len(upper_limit) != dim:
        raise ValueError("Length of lower and upper limits must match the number of variables.")

    # Converting sympy function to a numpy-aware function
    func_np = sym.lambdify(variables, func, "numpy")
    sampler = qmc.Sobol(d=dim, scramble=True)

    # Generating samples and scaling them to the specified range
    samples = sampler.random(num_samples)
    samples_scaled = np.array(lower_limit) + samples * (np.array(upper_limit) - np.array(lower_limit))

    # Evaluate the function at the sample points and calculate the integral estimate
    integral_estimate = np.mean(func_np(*np.transpose(samples_scaled)))

    return integral_estimate

def monte_carlo_integration(func, variables, lower_limit, upper_limit, num_samples):
    '''Computing the expectation of a given function, return a float value'''
    # func(sym.function): any function that is bounded and can be interation over L^2
    # num_samples(int): size of trails, produce enough precession digits
    dim = len(variables)
    if len(lower_limit) != dim or len(upper_limit) != dim:
        raise ValueError("Length of lower and upper limits must match the number of variables.")

    # Converting sympy function to a numpy-aware function
    func_np = sym.lambdify(variables, func, "numpy")

    # Create a random number generator
    rng = np.random.default_rng(12345)

    # Generating multi-dimensional samples and scaling them to the specified range
    samples = rng.uniform(size=(num_samples, dim))
    samples_scaled = lower_limit + samples * (np.array(upper_limit) - np.array(lower_limit))

    # Evaluate the function at the sample points and calculate the integral estimate
    integral_estimate = np.mean(func_np(*np.transpose(samples_scaled)))

    return integral_estimate

def Hermite_generating(multivariable_density, set_of_variables, index_j):
    '''Computing the multivariable Hermite polynomials with a giving multivariable density'''
    # multivariable_density(sym.function): a multivariable density function in a sym form
    # set_of_variables(array/list): a list contains the variable, which must be matched with the variable that using in the density function
    # index_j(array): using for computing the partial derivative to reach the correct hermite polynomials
    assert(len(set_of_variables) == len(index_j))
    diff_part = multivariable_density
    multivariable_density = (-1) ** sum(index_j)/multivariable_density
    for i in range(len(set_of_variables) - 1):
        for j in range(index_j[i]):
            if index_j[i] != 0:
                diff_part = sym.diff(diff_part, set_of_variables[i])
    multivariable_density = multivariable_density * diff_part
    return multivariable_density

def characteristic_function(cov_matrix, x):
    '''Compute the Gaussian characteristic function using a covariance matrix and a vector x'''
    x = sp.Matrix(x)
    cov_matrix_sp = sp.Matrix(cov_matrix)

    inv_cov = cov_matrix_sp.inv()
    result = -0.5 * x.T * inv_cov * x
    result = sp.exp(result[0, 0])
    return result

def symmetric_group(n):
    """Generate the symmetric group S_n."""
    # n(int): the degree of symmetric_group number
    return list(itertools.permutations(range(1, n + 1)))

def multilist_index(theta):
    """Return a s(\theta) as the multi list_index"""
    # theta(np.array/list) the degree of multi-hermite polynomial
    result = []
    for idx, num in enumerate(theta, 1):
        result.extend([idx] * num)
    return result


def normalization_factor(r_theta, c_theta, cov_matrix):
    '''Computing for the double product, should produce same result as double product over j_k'''
    # r_theta, c_theta(np.array) degree of polynomial
    # cov_matrix(np.matrix) covariance matrix generated by function.
    if (sum(r_theta)!= sum(c_theta)):
        return 0
    else:
        inv_cov = np.linalg.inv(cov_matrix)
        S_n = symmetric_group(sum(r_theta))
        s_alpha = multilist_index(r_theta)
        size = len(s_alpha)
        s_beta = multilist_index(c_theta)
        result = 0
        for set in S_n:
            product = 1
            for i in range(size):
                product *= inv_cov[(s_alpha[i] - 1),  (s_beta[set[i] - 1] - 1)]
            result += product
        return result


def multivariate_hermite_polynomial_from_covariance(cov_matrix, set_of_variables, index_j):
    '''Generating the Multivariate-hermite polynomials through the covariance matrix, by using the generating function'''
    # cov_matrix(sym.matrix): a covariance matrix that generated by function covariance_matrix
    # set_of_variables(array/list): a list contains the variable, which must be matched with the variable that using in symbol
    # index_j(array): using for computing the partial derivative to reach the correct hermite polynomials
    dim = len(set_of_variables)
    x = sp.Matrix(set_of_variables)
    N = sum(index_j)

    char_function = characteristic_function(cov_matrix, x)
    derivative = char_function

    for i in range(dim):
        for _ in range(index_j[i]):
            derivative = sp.diff(derivative, x[i])

    hermite_poly = sp.simplify((-1) ** N * derivative/(char_function * np.sqrt(normalization_factor(index_j, index_j, cov_matrix))))

    return hermite_poly


def reset_multiindex(num_of_random, expansion_degree):
    multiindex = [[0] * (num_of_random + 1) for _ in range(expansion_degree)]
    multiindex = [[expansion_degree] + index[1:] for index in multiindex]
    return multiindex



def single_index(array):
    '''Loop over all the valid index, and arrange them in a proper arrangement'''
    # array(np.array): the array is a tuple that contains several rvs with a form that only the first entry contains the number,
    # like: (3,0,0,0)
    combinations = []
    number_of_rv = len(array)
    expansion_degree = sum(array)
    for combo in combinations_with_replacement(range(number_of_rv), expansion_degree):
        counts = [0] * number_of_rv
        for slot in combo:
            counts[slot] += 1
        combinations.append(tuple(counts))
    return combinations



def lexicographic_order(num_of_random, expansion_degree):
    '''Generating the Lexicographic order within a matrix, the matrix will contain objects like [(0,0,0),(0,0,0)]'''
    # num_of_random(int): how many random variables are in the problem, using to determine the size of each index
    # expansion_degree(int): how many numbers are allowed in a single index
    K = int(np.math.factorial(num_of_random + expansion_degree - 1)/(np.math.factorial(expansion_degree) * np.math.factorial(num_of_random - 1)))
    result_matrix1 = np.empty((K, K), dtype=object)
    result_matrix2 = np.empty((K, K), dtype=object)
    multiindex = [[0] * (num_of_random) for _ in range(2)]
    multiindex = [[expansion_degree] + index[1:] for index in multiindex]
    index = single_index(multiindex[len(multiindex) - 1])
    for i in range(K):
        result_matrix1[i,:] = index
        result_matrix2[:,i] = index
    result_matrix = np.empty((K, K), dtype=object)
    for row in range(K):
        for column in range(K):
            if column >= row:
                result_matrix[row, column] = [result_matrix1[row, column]] + [result_matrix2[row, column]]
            else:
                result_matrix[row, column] = list([tuple([0] * num_of_random), tuple([0] * num_of_random)])
    return result_matrix

def double_product_rule(r_theta, c_theta, cov_matrix):
    '''This function helps to compute the double product between two hermite polynomials, return an float number as result'''
    # r_theta(np.array) :the row_theta is an array that contains how much number in corresponding row
    # c_theta(np.array) :the column_theta is an array that contains how much number in corresponding column
    # cov_matrix(np.matrix): a covariance matrix that match with variables
    numerator = normalization_factor(r_theta, c_theta, cov_matrix)
    # print(numerator)
    dominator = np.sqrt(normalization_factor(r_theta,r_theta,cov_matrix)) * np.sqrt(normalization_factor(c_theta,c_theta,cov_matrix))
    # print(dominator)
    # print("----------------")
    return numerator/dominator

# def double_product_rule(r_theta, c_theta, cov_matrix):
#     '''This function helps to compute the double product between two hermite polynomials, return an float number as result'''
#     # r_theta(np.array) :the row_theta is an array that contains how much number in corresponding row
#     # c_theta(np.array) :the column_theta is an array that contains how much number in corresponding column
#     # cov_matrix(np.matrix): a covariance matrix that match with variables
#     assert len(r_theta) == len(c_theta)
#     size = len(r_theta)
#     containing_lst = []
#     result_list = []
#     double_product_result = 0
#     for row in range(size):
#         array_list = [r_theta[row]] + [0] * (size - 1)
#         expansion_degree = single_index(array_list)
#         containing_lst.append(expansion_degree)
#
#     combinations = product(*containing_lst)
#     list_of_matrices = [np.array(combination) for combination in combinations]
#
#     for matrix in list_of_matrices:
#         matrix_column_sum = matrix.sum(axis=0)
#         if np.array_equal(matrix_column_sum, c_theta):
#             result_list.append(matrix)
#
#     for matrix in result_list:
#         num = np.sum(matrix * cov_matrix)
#         double_product_result += num
#
#     return 2 * double_product_result
#
#
# def double_product_over_jk(r_theta, c_theta, cov_matrix):
#     '''Computing for th term \mathbb{\{\phi_j * \phi_k\}, if |r_theta| != |c_theta| return zero, otherwise, return a
#     float number'''
#     # r_theta(np.array) :the row_theta is an array that contains how much number in corresponding row
#     # c_theta(np.array) :the column_theta is an array that contains how much number in corresponding column
#     # cov_matrix(np.matrix): a covariance matrix that match with variables
#     if sum(r_theta) != sum(c_theta):
#         return 0
#     numerator_part = double_product_rule(r_theta, c_theta, cov_matrix)
#     dominator_part = np.sqrt(double_product_rule(r_theta, r_theta, cov_matrix)**2) +\
#                         np.sqrt(double_product_rule(c_theta, c_theta, cov_matrix) ** 2)
#
#     return numerator_part / dominator_part




x, y, z = sym.symbols('x y z')
f = sp.exp(-x**2 -y**2 - z**2)
nb = 1/(2*np.pi) * sp.exp(-0.5 * ((x-1)**2 + (y-1)**2))

print(qmc_integration(nb,[x,y], [-3,-3], [3, 3], 800000))
# print(monte_carlo_integration(nb,[x,y], [-100,-100], [100, 100], 800000))

correlation_matrix = np.matrix([[1,0.2,0.4],[0.2,1,0.8],[0.4,0.8,1]])
cov_matrix1 = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
cov_matrix2 = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
cov_matrix = covariance_matrix([1,1,1], cov_matrix2)
x1, x2, x3 ,x4= sp.symbols('x1 x2 x3 x4')
f = x1**2 + x2**2 + x3**2
j = (0, 2, 0)
k = (0, 1, 1, 1)


# print(normalization_factor((1,0),(1,0), cov_matrix2))

def make_symmetric(matrix):
    """Convert an upper triangular matrix to a symmetric matrix."""
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[j][i] = matrix[i][j]
    return matrix

def computing_coefficient(random_variable_vec, standard_dev, correlation_matrix, function_y):
    # random_variable_vec: getting the number of the input and using for generating the covariance matrix
    # correlation_matrix: using for generating the covariance matrix
    # function_y: a square-integrate function respect to the system
    # expansion_degree: the maximum of expansion degree
    # standard_dev(list): a list that contains all the standard deviation of the random variables.
    num_of_random = len(random_variable_vec)
    expansion_degree = num_of_random - 1
    cov_matrix = covariance_matrix(standard_dev, correlation_matrix) # OK
    Matrix_A_index = lexicographic_order(num_of_random, expansion_degree)
    K = int(np.math.factorial(num_of_random + expansion_degree - 1)/(np.math.factorial(expansion_degree) * np.math.factorial(num_of_random - 1)))

    Matrix_A = np.zeros((K, K))
    vector_b = np.zeros(K)
    index_array = np.array([expansion_degree] + [0] * expansion_degree)
    index_array = single_index(index_array)

    Matrix_A_index = make_symmetric(Matrix_A_index)
    negative_estimating_region = [-1] * num_of_random
    print(Matrix_A_index)
    positive_estimating_region = [1] * num_of_random
    print(K)
    for i in range(K):
        hermite_poly = multivariate_hermite_polynomial_from_covariance(cov_matrix, random_variable_vec, index_array[i])
        # function = sp.simplify(function_y * hermite_poly)
        vector_b[i] = qmc_integration(function_y, random_variable_vec, negative_estimating_region, positive_estimating_region, 8000) * \
                      qmc_integration(hermite_poly, random_variable_vec, negative_estimating_region, positive_estimating_region, 8000)

        for j in range(K):
            Matrix_A[i][j] = double_product_rule(Matrix_A_index[i][j][0], Matrix_A_index[i][j][1], cov_matrix)

    result_vec = np.linalg.solve(Matrix_A, vector_b)

    return result_vec

print(computing_coefficient([x1,x2,x3],[1,1,1], cov_matrix, f))






