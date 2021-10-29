import csv

# Function to read landmark files, which contain the landmark
#   associated to the pixel coordinates in each mask image
#--------------------------------------------------------------------------------------------------------------------
def readCSV(file):
    landmarks = {}
    ids = []
    coordinates = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count != 0):
                landmarks[line_count] = {'id': int(row[0]),
                                         'x' : int(row[1]),
                                         'y' : int(row[2])}
                ids.append(int(row[0]))
                coordinates.append([int(row[1]),int(row[2])])

            line_count += 1
    
    return landmarks,ids,coordinates

#--------------------------------------------------------------------------------------------------------------------
