from utilsData import get_estimated_price

def train(mileage, price, mean, std_dev, learning_rate=0.01, iterations=1500):
	theta0 = 0
	theta1 = 0
	m = len(mileage)

	if m == 0:
		print("Error: No data available for training.")
		return float('nan'), float('nan')

	# mileage = [m / 1000 for m in mileage]
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

		# theta1_original = theta1 / std_dev
		# theta0_original = theta0 - (theta1_original * mean)


	return theta0, theta1