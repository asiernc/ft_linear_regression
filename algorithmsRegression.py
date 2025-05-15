from utilsData import get_estimated_price, get_thetas

def r_square(prices, mileages, std_dev, mean):
    theta0, theta1 = get_thetas()
    theta1_original = theta1 / std_dev
    theta0_original = theta0 - (theta1_original * mean)
    
    mean_real_prices = sum(prices) / len(prices)
    predictions = [get_estimated_price(m, theta0_original, theta1_original) for m in mileages]
    
    sst = sum((y - mean_real_prices) ** 2 for y in prices)
    ssr = sum((prices[i] - predictions[i]) ** 2 for i in range(len(prices)))
    
    r_square = 1 - (ssr / sst)
    
    return r_square

def mse(price, mileage, mean, std_dev):
	theta0, theta1 = get_thetas()
	theta1_original = theta1 / std_dev
	theta0_original = theta0 - (theta1_original * mean)

	n = len(price)
	if n == 0:
		print("Error. No data avalaible")
		return float('nan')

	predictions = [theta0_original + theta1_original * m for m in mileage]
	sum_square_errors = sum((price[i] - predictions[i]) ** 2 for i in range(len(price)))

	return sum_square_errors / n

def train(mileage, price, learning_rate=0.01, iterations=1500):
	theta0 = 0
	theta1 = 0
	m = len(mileage)

	if m == 0:
		print("Error: No data available for training.")
		return float('nan'), float('nan')

	for _ in range(iterations):
		sum_errors0 = 0.0
		sum_errors1 = 0.0
		# range in derivada calcular errores individuales y media para obtener predicciones
		for i in range(m):
			prediction = get_estimated_price(mileage[i], theta0, theta1)
			error = prediction - price[i]
			sum_errors0 += error
			sum_errors1 += error * mileage[i]
		
		# gradient descent
		tmp_theta0 = learning_rate * ( sum_errors0 / m )
		tmp_theta1 = learning_rate * ( sum_errors1 / m )

		theta0 -= tmp_theta0
		theta1 -= tmp_theta1

	return theta0, theta1