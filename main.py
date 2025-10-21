import json, os, csv, copy, sqlite3
from phone import Phone
import pandas as pd
from logic import list_phones, add_phone, update_phone, delete_phone, convert_data
from ki import analyze, similarite, recommend_for_user, item_item_matrix

CSVFILE = "phones_info_100.csv"

def main():
    running = True
    df = pd.read_csv(CSVFILE)
    df.dropna(inplace=True)
    dr = pd.read_csv("user_phone_ratings.csv")
    print(df.head())

    phones = [
        Phone(
            id=row["id"],
            brand=row["brand"],
            model=row["model"],
            price=row["price"],
            storage=row["storage"],
            RAM=row["RAM"],
            battery_capacity=row["battery_capacity"],
            weight=row["weight"],
            rating=row["rating"],
            description = row['description']
        )
        for i, row in df.iterrows()
    ]

    def save_to_csv(phones, filename=CSVFILE):
        df=pd.DataFrame([p.set_task() for p in phones])
        df.to_csv(filename,index=False, encoding="utf-8")
        print(f"CSV updated: {filename}")

    def sort_by_price(phones):
        phones[:] = sorted(phones, key=lambda x: x.price)
        save_to_csv(phones)

    while running:
        print("\n===== Menu =====\n")
        print("1. Show 5 phones")
        print("2. Add phone")
        print("3. Update phone")
        print("4. Delete phone")
        print("5. Convert data")
        print('6. Analyze')
        print('7. Normallize')
        print('8. Recomment for user')
        print("0. Exit")

        choice = int(input("Select option: "))

        match choice: 
            case 1:
                sort_by_price(phones)
                list_phones(phones, limit = 5)
            case 2:
                add_phone(phones)
            case 3:
                update_phone(phones)
            case 4:
                delete_phone(phones)
            case 5:
                convert_data(df)
            case 6:
                convert_data(df)
                analyze(df)
            case 7:
                similarite(df)
            case 8:
                convert_data(df)
                item_matrix = item_item_matrix(df)
                recommend_for_user(dr, item_matrix)
            case 9:
                item_item_matrix(df)
            case 0:
                save_to_csv(phones)
                running = False
            case _:
                print("Error!!!")

if __name__ == "__main__":
    main()