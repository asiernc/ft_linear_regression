import numpy as np
import sys
import json

get_estimated_price = lambda mileage, theta0, theta1: theta0 + theta1 * mileage

def get_thetas(filename="thetas.json"):
	try:
		with open(filename, "r") as f:
			data = json.load(f)
	except (FileNotFoundError, FileExistsError):
		print(f'File with filename {filename} does not exists.')
		sys.exit(-1)
	return data["theta0"], data["theta1"]

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
					continue
	except (FileNotFoundError,FileExistsError):
		print(f'File with filename {filename} does not exists.')
		sys.exit(-1)
	if len(mileages) == 0 or len(prices) == 0:
		print("Error: No valid data found in the file.")
		sys.exit(-1)
	return np.array(mileages), np.array(prices)

def save_thetas(theta0, theta1, filename="thetas.json"):
	with open(filename, "w") as file:
		json.dump({"theta0": theta0, "theta1": theta1}, file)