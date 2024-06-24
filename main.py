import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
import dask.dataframe as dd
class con_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# function to load data from CSV files
def load_data():
    trips_by_distance = pd.read_csv('data/Trips_by_Distance.csv')
    trips_full_data = pd.read_csv('data/Trips_Full_Data.csv')

    print(f"{con_colors.OKCYAN} Loading data from CSV files..." + con_colors.ENDC)

    return trips_by_distance, trips_full_data

# function to analyze population data
def analyze_population_data(df):
    # Summarize people staying at home and not staying at home
    total_staying_home = df['Population Staying at Home'].sum()
    total_not_staying_home = df['Population Not Staying at Home'].sum()
    print(f"Total population staying at home: {con_colors.OKGREEN} {total_staying_home}" + con_colors.ENDC)
    print(f"Total population not staying at home: {con_colors.OKGREEN} {total_not_staying_home}" + con_colors.ENDC)

# function to identify significant trip dates
def identify_significant_trip_dates(df):
    # Identify dates with significant number of specific trips
    dates_10_25 = df[df['Trips 10-25 Miles'] > 10000000]['Date']
    dates_50_100 = df[df['Trips 50-100 Miles'] > 10000000]['Date']
    print("Dates with more than 10 million people making 10-25 mile trips:",con_colors.OKGREEN, dates_10_25.unique(), con_colors.ENDC)
    print("Dates with more than 10 million people making 50-100 mile trips:",con_colors.OKGREEN, dates_50_100.unique(), con_colors.ENDC)

# function to visualize trip data
def visualize_trip_data(df):
    # Visualize trips data
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Trips 10-25 Miles'], label='Trips 10-25 Miles')
    plt.plot(df['Date'], df['Trips 50-100 Miles'], label='Trips 50-100 Miles', color='red')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Number of Trips')
    plt.title('Trip Distribution Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_chunk(chunk):
    return chunk['Number of Trips'].sum()

def parallel_processing(df, num_processors):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:
        results = list(executor.map(process_chunk, np.array_split(df, num_processors)))
    print(f"Total trips processed with {num_processors} processors: {sum(results)}")

# main function to run the program
def main():
    trips_by_distance, trips_full_data = load_data()
    analyze_population_data(trips_by_distance)
    identify_significant_trip_dates(trips_full_data)
    visualize_trip_data(trips_full_data)

    df = pd.DataFrame(trips_by_distance)
    parallel_processing(df, 10)
    parallel_processing(df, 20)

if __name__ == "__main__":
    main()