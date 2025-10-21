import json, os, csv, copy, sqlite3
from phone import Phone
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

CSVFILE = "phones_info_100.csv"

def save_to_csv(phones, filename=CSVFILE):
        df=pd.DataFrame([p.set_task() for p in phones])
        df.to_csv(filename,index=False, encoding="utf-8")
        print(f"CSV updated: {filename}")

def list_phones(phones, limit=None):
    if not phones:
        print("There are no phones")
        input("Press Enter to continue...")
        return
    
    count = 0
    for phone in phones:
        print(f"\n{phone.id}, {phone.brand}, {phone.model}, {phone.price}$, {phone.rating}/5")
        print(f"Storage: {phone.storage}GB, RAM: {phone.RAM}GB, Battery: {phone.battery_capacity}mAh, Weight: {phone.weight}g")
        print(f"{phone.description}")
        print("-" * 40)
        count += 1
        if count >= limit:
            input("Press Enter to continue...")
            break

def convert_data(df):
    df["price"] = df["price"].astype(str).str.replace('USD', "", regex=False)
    df['storage'] = df['storage'].astype(str).str.replace('GB', "", regex=False)
    df['storage'] = df['storage'].astype(str).str.replace('TB', "042", regex=False)
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '', regex=False)
    df['weight'] = df['weight'].astype(str).str.replace('g', '', regex=False)
    df['battery_capacity'] = df['battery_capacity'].astype(str).str.replace('mAh', '', regex=False)

    numeric_cols = ['price', 'storage', 'RAM', 'battery_capacity', 'weight', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    df.dropna(subset=numeric_cols, inplace = True)

def add_phone(phones):
    new_id = max([phone.id for phone in phones], default=0) + 1
    brand = input("Phone brend: ")
    model = input("Phone model: ")
    price = int(input("Phone price: "))
    rating = float(input("Phone rating from 1 to 5: "))
    if rating <= 0 or rating > 5:
        input("Rating input is incorrect")
        return
    storage = int(input("Phone storage: "))
    RAM = int(input("Phone RAM: "))
    battery_capacity = int(input("Phone battery capacity: "))
    weight = int(input("Phone weight: "))
    description = input("Phone description: ")
    
    
    new_phone = Phone(
        id=new_id,
        brand= brand,
        model=model,
        price=price,
        storage=storage,
        RAM=RAM,
        battery_capacity=battery_capacity,
        weight=weight,
        rating=rating,
        description=description
    )

    phones.append(new_phone)
    save_to_csv(phones)
    print(f"Phone {brand} {model} added successfully!")

def update_phone(phones):
    id = int(input("Enter phone id which you want to update: "))
    for idx, phone in enumerate(phones):
        if phone.id == id:
            id = id
            brand = input("Phone brend: ")
            model = input("Phone model: ")
            price = int(input("Phone price: "))
            rating = float(input("Phone rating from 1 to 5: "))
            if rating <= 0 or rating > 5:
                input("Rating input is incorrect")
                return
            storage = int(input("Phone storage: "))
            RAM = int(input("Phone RAM: "))
            battery_capacity = int(input("Phone battery capacity: "))
            weight = int(input("Phone weight: "))
            description = input("Phone description: ")

            phones[idx] = Phone(
                id=id,
                brand= brand,
                model=model,
                price=price,
                storage=storage,
                RAM=RAM,
                battery_capacity=battery_capacity,
                weight=weight,
                rating=rating,
                description=description
            )

            save_to_csv(phones)
            print(f"Phone with ID {id} was successfully updated") 
            return
    print(f"Phone with Id {id} not found")

def delete_phone(phones):
    id = int(input("Enter phone id which you want to delete: "))
    for i, phone in enumerate(phones):
        if phone.id == id:
            phones.pop(i)
            print("Phone with id {id} was successfully deleted")
            return