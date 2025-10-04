import json
import numpy as np
import pickle
__locations=None
__data_columns=None
__model=None


def get_estimated_price(location, sqft, bath, bhk):
    location_key = "location_" + location.lower()
    try:
        location_index = __data_columns.index(location_key)
    except:
        location_index = -1
    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if location_index >=0:
        x[location_index]=1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations


def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations

    with open(r"C:\Users\VC ROOM\Desktop\Data Science\price detection project\server\artifacts\columns.json", "r") as f:
        __data_columns=json.load(f)['data_columns']
        __locations=[col.replace("location_", "") for col in __data_columns[3:]]

    global __model
    with open(r"C:\Users\VC ROOM\Desktop\Data Science\price detection project\server\artifacts\bangalore_price_model.pickle", "rb") as f:
        __model=pickle.load(f)
    print("Loading saved artifacts...done")





if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
