import datetime
aviation_data = []
with open('AviationData.txt') as f:
    aviation_data = f.read().splitlines()

aviation_list = []
for line in aviation_data:
    temp = line.split("|")
    aviation_list.append([x.strip() for x in temp])

lax_code = []

for row in aviation_list:
    if "LAX94LA336" in " ".join(row):
        lax_code.append(row)

# print("Found using linear search")
# print(lax_code)
# this search is very slow as we are going r

activation_dict_list = []

column_names = aviation_data[0].split("|")
column_names = [x.strip() for x in column_names]
# print("Column Names")
# print(column_names)

for line in aviation_data[1:]:
    column_data = line.split("|")
    temp_dict = {}
    for i in range(len(column_names)):
        temp_dict[column_names[i].strip()] = column_data[i].strip()

    activation_dict_list.append(temp_dict)

lax_dict = []


for row in activation_dict_list:
    if "LAX94LA336" in row.values():
        lax_dict.append(row)
# print(activation_dict_list[0])
# print("Found using dict search")
# print(lax_dict)
# I found it harder to search in dict

state_accidents = {}
# Search using dict
# for row in activation_dict_list:
#     state = row["Location"][-2:].upper()

# search using list
location_index = column_names.index("Location")
for row in aviation_list:
    state = row[location_index][-2:].upper()
    # print("Location = {} and state = {}".format(row["Location"], state))
    if state in state_accidents:
        state_accidents[state] += 1
    else:
        state_accidents[state] = 1

state_accidents_sorted = sorted(
    state_accidents.items(), key=lambda x: x[1], reverse=True)

# print(state_accidents)
# print(state_accidents_sorted)

print("State with most accidents = '{}' \nTotal number of accidents = {}".format(
    state_accidents_sorted[0][0], state_accidents_sorted[0][1]))

# count how many fatalities and serious injuries occurred during each month.


event_date_index = column_names.index("Event Date")
total_fatal_column = column_names.index("Total Fatal Injuries")
total_serios_injury_column = column_names.index(
    "Total Serious Injuries")

event_date_dict = {}
for row in aviation_list:
    try:
        event_date = datetime.datetime.strptime(
            row[event_date_index], "%m/%d/%Y")
    except ValueError:
        event_date = None
    if event_date is not None:
        months = {}
        if event_date.year in event_date_dict:
            months = event_date_dict[event_date.year]
        fatal = 0
        if row[total_fatal_column] != "":
            fatal = int(row[total_fatal_column])
        injuries = 0
        if row[total_serios_injury_column] != "":
            injuries = int(row[total_serios_injury_column])
        long_month = event_date.strftime("%B")
        if long_month in months:
            months[long_month] += fatal + injuries
        else:
            months[long_month] = fatal + injuries
        event_date_dict[event_date.year] = months


def print_data(event_date_dict):
    event_data_sorted = sorted(event_date_dict)
    print(event_data_sorted)
    months_map = {"December": 12, "November": 11, "October": 10, "September": 9, "August": 8, "July": 7,
                  "June": 6, "May": 5, "April": 4, "March": 3, "February": 2, "January": 1}

    for year in event_data_sorted:
        print("Total Fatalities and serios injuries in the year {}".format(year))
        months_sorted = sorted(
            event_date_dict[year], key=lambda x: months_map[x])
        for month in months_sorted:
            print("{} = {}".format(month, event_date_dict[year][month]))
        print()


print_data(event_date_dict)
