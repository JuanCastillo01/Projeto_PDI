import shutil
import csv


with open("Annotations/RoCoLe-classes_csv.csv") as csv_File:
    image = csv.reader(csv_File, delimiter =',')
    line_count = 0
    for row in image:
        if line_count !=0:
            #   Quando saudavel tera Heal no inicio do nome
            if row[1] == "healthy":
                shutil.copy(f"Photos/{row[0]}", f"Base_Images/Heal_{row[0]}")

            #   Quando Doente tera Sick no inicio do nome e 1,2,3 ou e baseado no nivel/tipo
            else:
                shutil.copy(f"Photos/{row[0]}", f"Base_Images/Sick_{row[2][-1]}_{row[0]}")
                print(row[2])
        line_count = line_count + 1