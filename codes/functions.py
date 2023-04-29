
# Data Cleaning Function
def cleaning_func(left,right):
    
    #---CLEAN LEFT DATASET----
    #lower case
    left[["name", "address", "city"]] = left[["name", "address", "city"]].astype(str).applymap(str.lower)
    #fill NA and change to integer in postal code
    left["postal_code"] = left["postal_code"].fillna(0).astype(int)
    #drop column "categories"
    left.drop(columns=["categories"], inplace=True)
    #replace all non-alphanumeric and non-space characters with an empty string
    left["name"] = left["name"].str.replace(r'[^\w\s]','', regex=True)
    left["address"] = left["address"].str.replace(r'[^\w\s]','', regex=True)
    #strip leading spaces for name and then sort
    left["name"] = left["name"].str.strip()
    left = left.sort_values(by="name", ascending=True)
    #check for duplicates
    duplicate_left = left[left.duplicated(['name', 'address'])]
    duplicate_left
    #drop duplicates
    left_no_duplicates = left.drop_duplicates(subset=['name', 'address'], keep='first')

    #---CLEAN RIGHT DATASET----
    #lower case and drop trailing and leading spaces
    right[["name", "address", "city"]] = right[["name", "address", "city"]].astype(str).applymap(str.lower)
    #fill NA and change to integer in zip code
    right["zip_code"] = right["zip_code"].str[:5].fillna(0).astype(int)
    #drop column "size"
    right.drop(columns=["size"], inplace=True)
    #replace all llc and inc and non-alphanumeric and non-space characters with an empty string
    right["name"] = right["name"].str.replace("inc", "").str.replace("llc", "").str.replace(r'[^\w\s]','', regex=True)
    right["address"] = right["address"].str.replace(r'[^\w\s]','', regex=True)
    #strip leading spaces for name and then sort
    right["name"] = right["name"].str.strip()
    right = right.sort_values(by="name", ascending=True)
    #check for duplicates
    duplicate_right = right[right.duplicated(['name', 'address'])]
    duplicate_right
    #drop duplicates
    right_no_duplicates = right.drop_duplicates(subset=['name', 'address'], keep='first')

     #---SORT----
    left_copy = left_no_duplicates.copy()
    left_copy["combine"] = left_copy["name"] + " " + left_copy["address"]
    sorted_left = left_copy.sort_values(by=["state", "combine"], ascending=[True, True])

    right_copy = right_no_duplicates.copy()
    right_copy["combine"] = right_copy["name"] + " " + right_copy["address"]
    right_copy.head()
    sorted_right = right_copy.sort_values(by=["state", "combine"], ascending=[True, True])
    
    return sorted_left, sorted_right

# Proportion Function
def prop_alg(sorted_left,sorted_right):
    import pandas as pd
    
    # Define a function to calculate the proportion of matching words between two strings
    def calculate_word_match(string1, string2):
        # Split each string into words
        words1 = set(string1.split())
        words2 = set(string2.split())
        # Calculate the number of matching words using set intersection
        matches = words1 & words2
        # Calculate the proportion of matching words
        proportion = len(matches) / len(set(words1).union(words2))
        return proportion
    max_length_diff = 4
    matched_rows_dict = {}
    matched_rows = []
    #iterate through left dataset
    for left_index, left_row in sorted_left.iterrows():
        left_city, left_state, left_zip, left_combine = left_row['city'], left_row['state'], left_row['postal_code'], left_row['combine']
        # filter right dataset by state, city and zip based on current row
        right_filtered = sorted_right[(sorted_right['state'] == left_state) & (sorted_right['city'] == left_city) & (sorted_right['zip_code'] == left_zip)]
        max_proportion = 0
        matched_row = None
        #iterate through the rows of the filtered set
        for right_index, right_row in right_filtered.iterrows():
            right_combine = right_row['combine']
            #proceed to calculate proportion if length difference is <= max_length diff 
            length_diff = abs(len(left_combine) - len(right_combine))
            if length_diff <= max_length_diff:
                proportion = calculate_word_match(left_combine, right_combine)
                #check for highest proportion
                if proportion > max_proportion:
                    max_proportion = proportion
                    matched_row = {'left_entity_id': left_row['entity_id'], 'right_business_id': right_row['business_id'], 'confidence': proportion}
                # stop iterating if proportion is 1
                elif proportion == 1:
                    break
        #eliminate duplicate matched rows in the final output.
        if matched_row is not None:
            key = tuple(matched_row.values())
            if key not in matched_rows_dict:
                matched_rows_dict[key] = matched_row
                matched_rows.append(matched_row)
    # Create a new DataFrame from the matched rows list
    match = pd.DataFrame(matched_rows)
    # Filter rows with confidence greater than 0.8
    match = match[match['confidence'] > 0.8] 
    match.sort_values(by='confidence', ascending=False, inplace=True)

    return match

# Dynamic Function
def dynamic_alg(sorted_left,sorted_right):
    import pandas as pd
    # Define a function to calculate the proportion of matching words between two strings
    def calculate_word_match(string1, string2):
        import pandas as pd
        # Split each string into words
        words1 = string1.split()
        words2 = string2.split()
        len1 = len(words1)
        len2 = len(words2)
        # Create a matrix to store the results of subproblems
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Fill in the matrix
        for i in range(len1):
            for j in range(len2):
                if words1[i] == words2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        # Calculate the proportion of matching words using the matrix
        matches = dp[-1][-1]
        proportion = matches / (len1 + len2 - matches)
        return proportion
    max_length_diff = 4
    matched_rows_dict = {}
    matched_rows = []
    #iterate through left dataset
    for left_index, left_row in sorted_left.iterrows():
        left_city, left_state, left_zip, left_combine = left_row['city'], left_row['state'], left_row['postal_code'], left_row['combine']
        # filter right dataset by state, city and zip based on current row
        right_filtered = sorted_right[(sorted_right['state'] == left_state) & (sorted_right['city'] == left_city) & (sorted_right['zip_code'] == left_zip)]
        max_proportion = 0
        matched_row = None
        #iterate through the rows of the filtered set
        for right_index, right_row in right_filtered.iterrows():
            right_combine = right_row['combine']
            #proceed to calculate proportion if length difference is <= max_length diff 
            length_diff = abs(len(left_combine) - len(right_combine))
            if length_diff <= max_length_diff:
                proportion = calculate_word_match(left_combine, right_combine)
                #check for highest proportion
                if proportion > max_proportion:
                    max_proportion = proportion
                    matched_row = {'left_entity_id': left_row['entity_id'], 'right_business_id': right_row['business_id'], 'confidence': proportion}
                # stop iterating if proportion is 1
                elif proportion == 1:
                    break
        #eliminate duplicate matched rows in the final output.
        if matched_row is not None:
            key = tuple(matched_row.values())
            if key not in matched_rows_dict:
                matched_rows_dict[key] = matched_row
                matched_rows.append(matched_row)
    # Create a new DataFrame from the matched rows list
    match = pd.DataFrame(matched_rows)
    # Filter rows with confidence greater than 0.8
    match = match[match['confidence'] > 0.8] 
    match.sort_values(by='confidence', ascending=False, inplace=True)

    return match


# Fuzzywuzzy Function
def fuzzy_alg(sorted_left,sorted_right):
    import pandas as pd
    from fuzzywuzzy import fuzz
       
    # Use fuzzy wuzzy to calculate similarity of two strings
    def calculate_string_similarity(string1, string2):
        similarity = fuzz.token_set_ratio(string1, string2) / 100.0
        return similarity
    max_length_diff = 4
    matched_rows_dict = {}
    matched_rows = []
    #iterate through left dataset
    for left_index, left_row in sorted_left.iterrows():
        left_city, left_state, left_zip, left_combine = left_row['city'], left_row['state'], left_row['postal_code'], left_row['combine']
        # filter right dataset by state, city and zip based on current row
        right_filtered = sorted_right[(sorted_right['state'] == left_state) & (sorted_right['city'] == left_city) & (sorted_right['zip_code'] == left_zip)]
        max_similarity = 0
        matched_row = None
        #iterate through the rows of the filtered set
        for right_index, right_row in right_filtered.iterrows():
            right_combine = right_row['combine']
            #proceed to calculate proportion if length difference is <= max_length diff 
            length_diff = abs(len(left_combine) - len(right_combine))
            if length_diff <= max_length_diff:
                similarity = calculate_string_similarity(left_combine, right_combine)
                #check for highest proportion
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_row = {'left_entity_id': left_row['entity_id'], 'right_business_id': right_row['business_id'], 'confidence': similarity}
                # stop iterating if similarity is max
                elif similarity == 1:
                    break
        #eliminate duplicate matched rows in the final output.
        if matched_row is not None:
            key = tuple(matched_row.values())
            if key not in matched_rows_dict:
                matched_rows_dict[key] = matched_row
                matched_rows.append(matched_row)

    # Create a new DataFrame from the matched rows list
    match = pd.DataFrame(matched_rows)
    # Filter rows with confidence greater than 0.8
    match = match[match['confidence'] > 0.8] 
    match.sort_values(by='confidence', ascending=False, inplace=True)

    return match.to_csv('match_records.csv', index=False)


# Data Visualization Function
def viz_func(left_no_duplicates,right_no_duplicates):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Plot a bar chart of the state counts
    state_counts_left = left_no_duplicates['state'].value_counts()
    state_counts_right = right_no_duplicates['state'].value_counts()
    state_counts = pd.DataFrame({
        'Left': state_counts_left,
        'Right': state_counts_right}).fillna(0)
    ax = state_counts.plot(kind='bar', figsize=(10, 5))
    ax.set_xlabel('State')
    ax.set_ylabel('Count')
    ax.legend()
    plt.show()

    # group the data by company name and count the number of addresses in the left dataframe
    left_counts = left_no_duplicates.groupby('name').count()['address']
    # count the number of companies with multiple addresses in the left dataframe, sorted by count of addresses in descending order
    print(left_counts.value_counts().sort_index(ascending=False))

    # group the data by company name and count the number of addresses in the left dataframe
    left_counts = left_no_duplicates.groupby('name').count()['address']
    # filter companies with more than 80 addresses in left dataframe
    left_counts = left_counts[left_counts > 80]
    # sort the left_counts by the count of addresses in descending order
    left_counts = left_counts.sort_values(ascending=False)
    # plot the count of addresses for each company in the left dataframe, sorted by the count of addresses in descending order
    ax = left_counts.plot(kind='bar', figsize=(10, 5))
    ax.set_title('Number of addresses for each company in the left dataframe')
    ax.set_xlabel('Company name')
    ax.set_ylabel('Count')
    plt.show()

    # group the data by company name and count the number of addresses in the right dataframe
    right_counts = right_no_duplicates.groupby('name').count()['address']
    # print the frequency of the counts of addresses for each company in the right dataframe
    print(right_counts.value_counts().sort_index(ascending=False))
    
    # group the data by company name and count the number of addresses
    right_counts = right_no_duplicates.groupby('name').count()['address']
    # filter companies with at least 3 addresses in right dataframe
    right_counts = right_counts[right_counts >= 3 ]
    # plot the count of addresses for each company in the right dataframe
    ax = right_counts.plot(kind='bar', figsize=(10, 5))
    ax.set_title('Number of addresses for each company in the right dataframe')
    ax.set_xlabel('Company name')
    ax.set_ylabel('Count')
    plt.show()
    