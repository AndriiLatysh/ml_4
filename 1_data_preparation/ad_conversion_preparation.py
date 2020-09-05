import pandas as pd
import numpy as np


def set_printing_option():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    # pd.set_option("display.max_columns", None)


def convert_text_to_binary(text):
    if text == "y":
        return 1
    elif text == "n":
        return 0
    else:
        return None


def convert_rating_to_grade(rating):
    rating_mapping = {"excellent": 5, "good": 4, "ok": 3, "bad": 2, np.NaN: 1}
    if rating in rating_mapping:
        return rating_mapping[rating]
    else:
        return None


def convert_age_to_range(age):
    if 6 <= age < 18:
        return "6-18"
    elif 18 <= age < 35:
        return "18-35"
    elif 35 <= age < 99:
        return "35-99"
    else:
        return "unknown"


set_printing_option()

conversion_df = pd.read_csv("data/ad_conversion.csv")

conversion_df["last name"].fillna("", inplace=True)

conversion_df.at[8, "gender"] = "F"
conversion_df.at[14, "gender"] = "M"
conversion_df.at[29, "gender"] = "F"
conversion_df.at[37, "gender"] = "M"

for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] > 100:
        conversion_df.at[z, "seen count"] = 0

conversion_df.insert(conversion_df.columns.get_loc("gender"), "full name", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "full name"] = (
            conversion_df.at[z, "first name"] + " " + conversion_df.at[z, "last name"]).strip()

conversion_df.drop(columns=["first name", "last name"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "birthday", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "birthday"] = pd.Timestamp(day=conversion_df.at[z, "day of birth"],
                                                   month=conversion_df.at[z, "month of birth"],
                                                   year=conversion_df.at[z, "year of birth"])

conversion_df.drop(columns=["day of birth", "month of birth", "year of birth"], inplace=True)

for z in range(len(conversion_df)):
    conversion_df.at[z, "followed ad"] = convert_text_to_binary(conversion_df.at[z, "followed ad"])
    conversion_df.at[z, "made purchase"] = convert_text_to_binary(conversion_df.at[z, "made purchase"])
    conversion_df.at[z, "user rating"] = convert_rating_to_grade(conversion_df.at[z, "user rating"])

conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "age", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "age"] = (pd.Timestamp("01-01-2019") - conversion_df.at[z, "birthday"]).days // 365

conversion_df.drop(columns=["birthday"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "age bucket", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "age bucket"] = convert_age_to_range(conversion_df.at[z, "age"])

conversion_df.drop(columns=["age"], inplace=True)

conversion_df = conversion_df.astype({"followed ad": "int64", "made purchase": "int64", "user rating": "int64"})

conversion_df.insert(conversion_df.columns.get_loc("user rating"), "ad effectiveness", None)
for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] == 0:
        conversion_df.at[z, "ad effectiveness"] = -1
    else:
        conversion_df.at[z, "ad effectiveness"] = (conversion_df.at[z, "followed ad"] +
                                                   conversion_df.at[z, "made purchase"]) / \
                                                  (2 * conversion_df.at[z, "seen count"])

conversion_df = conversion_df.astype({"ad effectiveness": "int64"})

print(conversion_df.dtypes)

colors_grouped = (conversion_df[["color scheme", "followed ad", "made purchase"]]).groupby("color scheme").mean()
print(colors_grouped)

print(conversion_df)

conversion_df.to_csv("data/prepared_ad_conversion.csv")
