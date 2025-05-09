import json
import sys
import csv
import numpy as np
from utilsData import load_data, save_thetas, get_thetas, get_estimated_price
from algorithmsRegression import train, mse
import matplotlib.pyplot as plt

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
def draw(mileages, prices, theta0, theta1, mean, std_dev):
	theta1_original = theta1 / std_dev
	theta0_original = theta0 - (theta1_original * mean)
	min_mileage = min(mileages)
	max_mileage = max(mileages)
	regression_mileage = np.linspace(min_mileage, max_mileage, 100)
	regression_prices = [theta0_original + theta1_original * pt for pt in regression_mileage]

	plt.scatter(mileages, prices, color='blue', label='Real data')
	plt.plot(regression_mileage, regression_prices, color='red', label='Regression')
	plt.ylim(0, max(prices) * 1.1)
	plt.xlabel('Mileages')
	plt.ylabel('Price')
	plt.legend()
	plt.savefig("plot.png")

def main():
	theta0, theta1 = get_thetas()
	mileages, prices = load_data()
	standardized_mileages, mean, std_dev = standarize_data(mileages)
	while True:
		try:
			option = int(input('Write 1 for train the model\nWrite 2 for get an estimated price\nEnter your choice:'))
			if option == 1:
				theta0, theta1 = train(standardized_mileages, prices, mean, std_dev, learning_rate=0.01, iterations=1500)
				print({"theta0": theta0, "theta1": theta1})
				if not (theta0 == float('nan') or theta1 == float('nan')):
					save_thetas(theta0, theta1)
					print(f'Mean Square Error: {mse(prices, standardized_mileages)}')
					print("Model trained and thetas saved.")
				else:
					print("Error: Training failed due to invalid data.")
			elif option == 2:
				mileage = get_mileage()
				mileage_std = (mileage - mean) / std_dev 
				price = get_estimated_price(mileage_std, theta0, theta1)
				print(f'Estimated price: {price}')
			else:
				print("Invalid option. Please enter 1 or 2")
			theta0_e, theta1_e = get_thetas()
			draw(mileages, prices, theta0_e, theta1_e, mean, std_dev)
		except ValueError:
			print("Error: Please enter a valid numeric option.")

if __name__ == "__main__":
	main()