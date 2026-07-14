import csv

array_to_write = []

num_rows_original = 0

with open("full-matches.csv",'r' ,newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        array_to_write.append(row[0])
        num_rows_original += 1

# now check for dupes in candidate links
with open("candidate_videos_to_add.csv",'r' ,newline='',encoding='utf-8') as second_f:
    reader_2 = csv.reader(second_f)
    for row in reader_2:
        if row[0] not in array_to_write:
            array_to_write.append(row[0])


extension_part_of_array = array_to_write[num_rows_original:len(array_to_write)]

# now write the output csv's

with open("full_matches_extended.csv","w",newline='',encoding='utf-8') as out_f:
    writer = csv.writer(out_f)
    for link in array_to_write:
        writer.writerow([link])


with open("new_matches.csv",'w',newline='',encoding='utf-8') as second_out_f:
    writer_2 = csv.writer(second_out_f)
    for link in extension_part_of_array:
        writer_2.writerow([link])
