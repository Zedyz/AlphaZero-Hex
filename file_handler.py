import csv

def write_to_file(data1, data2, data3, filename=None):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data1])

