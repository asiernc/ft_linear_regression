import json
import sys
import csv
import numpy as np

def load_data(filename="data.csv"):
	mileages = []
	prices = []
	try:
		with open(filename, "r") as f:
			for line in f:
				try:
					mileage, price = line.strip().split(',')
					mileages.append(float(mileage))
					prices.append(float(price))
				except ValueError:
					print(f'Skipping invalid line: {line}')
	except (FileNotFoundError,FileExistsError):
		print(f'File with filename {filename} does not exists.')
		sys.exit(-1)
	if len(mileages) == 0 or len(prices) == 0:
		print("Error: No valid data found in the file.")
		sys.exit(-1)
	return np.array(mileages), np.array(prices)

def standarize_data(mileages):
	mean = np.mean(mileages)
	std_dev = np.std(mileages)
	standardized_mileages = (mileages - mean) / std_dev
	print(f"Media: {mean}")
	print(f"Desviación estándar: {std_dev}")
	print(f"Mileages estandarizados:\n{standardized_mileages}")
	return standardized_mileages, mean, std_dev

def destandarize_data(standaritzed_mileages, mean, std_dev):
	original = standaritzed_mileages * std_dev + mean
	return original


def get_estimated_price(mileage, theta0, theta1, mean, std_dev):
	#return theta0 + (theta1 * (mileage / 1000))
	mileage_original = mileage * std_dev + mean
	return theta0 + (theta1 * (mileage_original / 1000))

def train(mileage, price, mean, learning_rate=0.01, iterations=1500):
	theta0 = 0
	theta1 = 0
	m = mean

	if m == 0:
		print("Error: No data available for training.")
		return float('nan'), float('nan')

	mileage = [m / 1000 for m in mileage]
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


def get_thetas(filename="thetas.json"):
	try:
		with open(filename, "r") as f:
			data = json.load(f)
	except (FileNotFoundError, FileExistsError):
		print(f'File with filename {filename} does not exists.')
		sys.exit(-1)
	return data["theta0"], data["theta1"]

def save_thetas(theta0, theta1, filename="thetas.json"):
	with open(filename, "w") as file:
		json.dump({"theta0": theta0, "theta1": theta1}, file)

def get_mileage():
	while True:
		try:
			mileage = float(input("Introduce the km of the car: "))
			if mileage < 0:
				print("\nError: Mileage cannot be negative. Please enter a positive value.")
			else:
				return mileage
		except ValueError:
			print("Error: Please enter a valid numeric value.")

# estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)

def main():
	theta0, theta1 = get_thetas()
	mileages, prices = load_data()
	standardized_mileages, mean, std_dev = standarize_data(mileages)
	while True:
		try:
			option = int(input('Write 1 for train the model\nWrite 2 for get an estimated price\nEnter your choice:'))
			if option == 1:
				theta0, theta1 = train(standardized_mileages, prices, mean, learning_rate=0.01, iterations=1500)
				print({"theta0": theta0, "theta1": theta1})
				if not (theta0 == float('nan') or theta1 == float('nan')):
					save_thetas(theta0, theta1)
					print("Model trained and thetas saved.")
				else:
					print("Error: Training failed due to invalid data.")
			elif option == 2:
				mileage = get_mileage()
				standardized_mileages = (mileage - mean) / std_dev
				price = get_estimated_price(standardized_mileages, theta0, theta1)
				print(f'Estimated price: {price}')
			else:
				print("Invalid option. Please enter 1 or 2")
		except ValueError:
			print("Error: Please enter a valid numeric option.")

if __name__ == "__main__":
	main()