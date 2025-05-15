import json
import sys
import csv
from utilsData import load_data, save_thetas, get_thetas, get_estimated_price
from algorithmsRegression import train, mse, r_square
import matplotlib.pyplot as plt
from colorama import Fore, Style

def standarize_data(mileages):
	
	mean = sum(mileages) / len(mileages)
	variance = sum((m - mean) ** 2 for m in mileages) / len(mileages)
	std_dev = variance ** 0.5
	standardized_mileages = (mileages - mean) / std_dev
	
	if std_dev == 0:
		print("Error: Standard deviation is zero. Cannot standarize data.")	
	
	standardized_mileages = [(x - mean) / std_dev for x in mileages]
	
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

def draw(mileages, prices, theta0, theta1, mean, std_dev):

	theta1_original = theta1 / std_dev
	theta0_original = theta0 - (theta1_original * mean)
	
	predictions = [theta0_original + theta1_original * m for m in mileages]

	current_mse = mse(prices, mileages, mean, std_dev)

	plt.figure(figsize=(10,6))

	plt.scatter(mileages, prices, color='blue', label='Real Data')
	plt.plot(mileages, predictions, color='red', label=f'Regresión (MSE: {current_mse:.2f})')

	for x, y_real, y_pred in zip(mileages, prices, predictions):
		plt.plot([x, x], [y_real, y_pred], color='gray', linestyle=':', alpha=0.5)

	plt.xlabel('Mileages (m)')
	plt.ylabel('Price ($)')
	plt.title('ft_linear_regression: Price vs Mileage')
	plt.legend()
	plt.grid(alpha=0.2)
	plt.savefig("linear_regression.png")
	plt.close()

def train_model(standardized_mileages, mileages, prices, mean, std_dev):
	theta0, theta1 = train(standardized_mileages, prices, learning_rate=0.01, iterations=1500)
	if not (theta0 == float('nan') or theta1 == float('nan')):
		save_thetas(theta0, theta1)
		print("Model trained and thetas saved.")
		draw(mileages, prices, theta0, theta1, mean, std_dev)
	else:
		print("Error: Training failed due to invalid data.")

def estimate_price(theta0, theta1, mean, std_dev):
	mileage = get_mileage()
	theta1_original = theta1 / std_dev
	theta0_original = theta0 - (theta1_original * mean)
	price = get_estimated_price(mileage, theta0_original, theta1_original)
	print(f'Estimated price: {price:.2f}$')
    
def menu():
	print(Fore.GREEN + "Write 1 to train the model" + Style.RESET_ALL)
	print(Fore.GREEN + "Write 2 to get an estimated price" + Style.RESET_ALL)
	print(Fore.GREEN + "Write 3 to see the Coefficient of determination (R²)" + Style.RESET_ALL)
	print(Fore.RED + "Write 4 to exit" + Style.RESET_ALL)
	while True:
		try:
			option = int(input(Fore.BLUE + "Enter your choice: " + Style.RESET_ALL))
			if option in [1, 2, 3, 4]:
				return option
			else:
				print(Fore.RED + "Invalid option. Please enter 1, 2, 3, 4 or 5." + Style.RESET_ALL)
		except ValueError:
			print(Fore.RED + "Error: Please enter a valid numeric option." + Style.RESET_ALL)

def main():
	while True:
		option = menu()
		theta0, theta1 = get_thetas()
		mileages, prices = load_data()
		standardized_mileages, mean, std_dev = standarize_data(mileages)
		if option == 1:
			train_model(standardized_mileages, mileages, prices, mean, std_dev)
		elif option == 2:
			estimate_price(theta0, theta1, mean, std_dev)
		elif option == 3:
			print(f'Coefficient of determination R²: {r_square(prices, mileages, std_dev, mean):.2f}')
		elif option == 4:
			break

if __name__ == "__main__":
	main()