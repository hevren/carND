import csv

with open('/home/he/prj/CarND-Traffic-Sign-Classifier-Project/signnames.csv', 'r') as csvfile:
    signreader = csv.reader(csvfile, delimiter=',')
    signnames = list(signreader)

print('-----------------------------------')
print(signnames[30][1])
