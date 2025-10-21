import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import numpy as np
from logic import convert_data 

def analyze(phones):
    print("-" * 40)
    highest_phone_price = phones.loc[phones["price"].idxmax()]
    print("Highest phone price is:\n", highest_phone_price)

    print("-" * 40)
    lowest_phone_price = phones.loc[phones["price"].idxmin()]
    print("Lowest phone price is:\n", lowest_phone_price)

    print("-" * 40)
    average_price_and_rating = phones.groupby('brand')[['price', 'rating']].mean()
    print("Average price and rating in each brand: ", average_price_and_rating)

    print("-" * 40)
    budjet = 300
    flagship = 700
    
    budjet_phones = phones[phones["price"] < budjet]
    print("Budjet phones:\n", budjet_phones)

    print("-" * 40)
    average_battery_budjet = budjet_phones['battery_capacity'].mean()
    print('Average battery capacity for budjet phones:', average_battery_budjet)

    print("-" * 40)
    mid_range_phones = phones[(phones["price"] > budjet) & (phones["price"] < flagship)]
    print("Mid range phones:\n", mid_range_phones)

    print("-" * 40)
    average_battery_mid_range = mid_range_phones['battery_capacity'].mean()
    print('Average battery capacity for mid-range phones: ', average_battery_mid_range)

    print("-" * 40)
    flagship_phones = phones[phones["price"] > flagship]
    print("Flagship phones:\n", flagship_phones)

    print("-" * 40)
    average_battery_flagship = flagship_phones['battery_capacity'].mean()
    print('Average battery capacity for flagship phones: ', average_battery_flagship)

    print("-" * 40)
    lowest_phone_ratings = phones.loc[phones["rating"].idxmin()]
    print("Lowest phone ratings is:\n", lowest_phone_ratings)

    print("-" * 40)
    highest_phone_ratings = phones.loc[phones["rating"].idxmax()]
    print("Highest phone ratings is:\n", highest_phone_ratings)
    
    print("-" * 40)
    biggest_battery = phones.loc[phones["battery_capacity"].idxmax()]
    print("Biggest battery capacity:\n", biggest_battery)

    plt.plot(phones['price'], phones['rating'])
    plt.scatter(phones['price'], phones['rating'])
    plt.show()

    plt.plot(phones['price'], phones['battery_capacity'])
    plt.scatter(phones['price'], phones['battery_capacity'])
    plt.show()

    plt.plot(phones['price'], phones['storage'])
    plt.scatter(phones['price'], phones['storage'])
    plt.show()

def normallize(phones):
    convert_data(phones)
    scaler = skl.preprocessing.MinMaxScaler(feature_range=(0, 1))
    normalize_numeric = scaler.fit_transform(phones[['price', 'storage', 'RAM', 'battery_capacity', 'weight', 'rating']])
    normalize_numeric = pd.DataFrame(
        normalize_numeric,
        columns=['price', 'storage', 'RAM', 'battery_capacity', 'weight', 'rating']
    )
    normalize_numeric['price'] = normalize_numeric['price'] * 2
    
    encoder = skl.preprocessing.OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(phones[['brand']])
    encoded_phones = pd.DataFrame(
        encoded, 
        columns=encoder.get_feature_names_out(["brand"])
    )
    encoded_phones = 0.5 * encoded_phones

    vocabulary = ["battery", "gaming", "waterproof", "camera"]
    vectorizer = skl.feature_extraction.text.TfidfVectorizer(vocabulary = vocabulary)
    desctiption = vectorizer.fit_transform(phones['description'])
    desctiption = pd.DataFrame(
        desctiption.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    weight = {"battery": 0.3, "gaming": 0.6, "waterproof": 0.45, "camera": 0.2}
    for word, w in weight.items():
        if word in desctiption.columns:
            desctiption[word] *= w

    merged = pd.concat([normalize_numeric * 0.6, encoded_phones * 0.2, desctiption * 0.2], axis=1)
    print(merged)
    return merged

def similarite(phones):
    normilize_phones = normallize(phones)
    choose = input("Enter a phonen model: ")
    chosen_row = phones[phones["model"] == choose]
    if chosen_row.empty:
        print("Model not found")
        return

    phone_id = chosen_row.index[0]
    similarities = skl.metrics.pairwise.cosine_similarity(normilize_phones.iloc[[phone_id]], normilize_phones).flatten()
    
    df = phones.copy()
    df['Similarity'] = similarities

    df = df[df['model'] != choose].sort_values(by=('Similarity'), ascending = False)
    df.to_csv('recommendation_result.csv', index = False, encoding = 'utf-8')
    print("Results successfully saved")
    
    df = pd.DataFrame({
        "Brand": phones['brand'],
        "Phone": phones['model'],
        "Similarity": similarities
    }).sort_values(by="Similarity", ascending=False)
    
    print('\n Recommended phones:\n')
    print(df.head(5))

    return df

def recommend_for_user(rating, item_item_matrix):
    
    rating = rating.groupby(['user_id', 'phone_model'], as_index=False)['rating'].mean()

    print(rating)
    rating.to_csv('sorted_user_rating.csv', index=False, encoding='utf-8')

    print("-" * 40)

    matrix = rating.pivot(
        values='rating',
        columns='phone_model', 
        index='user_id'
    )
    print(matrix)

    
    user_id = int(input('Enter user ID: '))
    find_rating(user_id, rating ,item_item_matrix)

'''def find_rating(user_id, rating_df, item_item_matrix):
    user_ratings = rating_df[rating_df['user_id'] == user_id]
    user_rated_phones = user_ratings['phone_model']

    item_item_matrix_no_dup = item_item_matrix.drop_duplicates()
    
    filtered_users_phones = np.intersect1d(user_rated_phones, item_item_matrix_no_dup.columns)
    filtered_sim = item_item_matrix_no_dup[filtered_users_phones]

    predictions = {}
    for phone in filtered_sim.index:
        if phone in filtered_users_phones:
            continue
        weights = filtered_sim.loc[phone, user_rated_phones].values
        scores = user_ratings.set_index('phone_model').loc[user_rated_phones, 'rating'].values
        predicted = np.dot(weights, scores) / np.sum(weights)
        predictions[phone] = predicted

    prediction_csv = pd.Series(predictions).sort_values(ascending=False)
    prediction_csv.to_csv('predictions.csv', index=True, encoding='utf-8')'''

def find_rating(user_id, rating_df, item_item_matrix):
    user_ratings = rating_df[rating_df['user_id'] == user_id]
    user_rated_phones = user_ratings['phone_model']

    item_item_matrix_no_dup = item_item_matrix.loc[~item_item_matrix.index.duplicated(keep='first')]
    item_item_matrix_no_dup = item_item_matrix_no_dup.loc[:, ~item_item_matrix_no_dup.columns.duplicated()]

    filtered_users_phones = np.intersect1d(user_rated_phones, item_item_matrix_no_dup.columns)
    filtered_sim = item_item_matrix_no_dup[filtered_users_phones]

    predictions = {}
    for phone in filtered_sim.index:
        if phone in filtered_users_phones:
            continue
        common_phones = [p for p in user_rated_phones if p in filtered_sim.columns]
        weights = filtered_sim.loc[phone, common_phones].values
        scores = user_ratings.set_index('phone_model').loc[common_phones, 'rating'].values

        if np.sum(weights) == 0:
            continue

        pair_wei_sco = list(zip(weights, scores))
        sorted_pair = sorted(pair_wei_sco, key=lambda pair: pair[0], reverse=True)
        
        top_k_pairs = sorted_pair[:10]
        top_weight = [pair[0] for pair in top_k_pairs]
        top_scores = [pair[1] for pair in top_k_pairs]

        predicted = np.dot(top_weight, top_scores) / np.sum(weights)
        predictions[phone] = predicted

    prediction_csv = pd.Series(predictions).sort_values(ascending=False)
    prediction_csv.to_csv('predictions.csv', index=True, encoding='utf-8')
    return prediction_csv




def item_item_matrix(phones):

    normallized_phones = normallize(phones)
    
    similarities = skl.metrics.pairwise.cosine_similarity(normallized_phones)

    df = pd.DataFrame(
        similarities,
        index=phones['model'],
        columns=phones['model']
    )

    df.to_csv('reccomend_for_user.csv', index=True, encoding='utf-8')

    return df