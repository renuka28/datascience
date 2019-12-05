import numpy as np
from scipy.stats import chisquare
import pandas as pd
import csv
import re
import os


def normalize_text(text):
    text = text.lower()
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    return text


def normalize_values(text):
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    try:
        text = int(text)
    except Exception:
        text = 0
    return text


def count_matches(row):
    split_answer = row["clean_answer"].split(" ")
    split_question = row["clean_question"].split(" ")
    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0
    match_count = 0
    for item in split_answer:
        if item in split_question:
            match_count += 1
    return match_count / len(split_answer)


# set to store all the terms/words used in questions
terms_used = []
terms_used_unique = []


def find_mean_question_overlap(jeopardy):
    question_overlap = []
    jeopardy = jeopardy.sort_values("Air Date")

    for i, row in jeopardy.iterrows():
        split_question = row["clean_question"].split(" ")
        split_question = [q for q in split_question if len(q) > 5]
        match_count = 0
        for word in split_question:
            if word in terms_used:
                match_count += 1
        for word in split_question:
            terms_used.append(word)
        if len(split_question) > 0:
            match_count /= len(split_question)
        question_overlap.append(match_count)
    jeopardy["question_overlap"] = question_overlap
    return jeopardy["question_overlap"].mean()


def determine_value(row):
    value = 0
    if row["clean_value"] > 800:
        value = 1
    return value


def count_usage(jeopardy, term):
    low_count = 0
    high_count = 0
    for i, row in jeopardy.iterrows():
        if term in row["clean_question"].split(" "):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count


def find_observed(jeopardy, comparison_terms):
    observed = []
    for term in comparison_terms:
        observed.append(count_usage(jeopardy, term))
    return observed


def calculate_chi_squared(jeopardy, observed):
    high_value_count = jeopardy[jeopardy["high_value"] == 1].shape[0]
    low_value_count = jeopardy[jeopardy["high_value"] == 0].shape[0]

    chi_squared = []
    for obs in observed:
        total = sum(obs)
        total_prop = total / jeopardy.shape[0]
        high_value_exp = total_prop * high_value_count
        low_value_exp = total_prop * low_value_count

        obs = np.array([obs[0], obs[1]])
        exp = np.array([high_value_exp, low_value_exp])
        chi_squared.append(chisquare(obs, exp))

    return chi_squared


def find_and_print_chi_squared(jeopardy, terms):
    print("calculating chi squared for words = {}".format(terms))
    observed = find_observed(jeopardy, terms)

    # Now that we have found the observed counts for a few terms,
    # we can compute the expected counts and the chi-squared value.
    chi_squared = calculate_chi_squared(jeopardy, observed)
    for chi_sq in chi_squared:
        print("statistic = {0:.2f} p value = {1:.2f}%".format(
            chi_sq[0], chi_sq[1]*100))


def read_data(filename):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name)
    return df


def clean_data(jeopardy):
    jeopardy.columns = map(str.strip, ['Show Number', ' Air Date', ' Round', ' Category', ' Value',
                                       ' Question', ' Answer'])
    # data normalization
    jeopardy["clean_question"] = jeopardy["Question"].apply(normalize_text)
    jeopardy["clean_answer"] = jeopardy["Answer"].apply(normalize_text)
    jeopardy["clean_value"] = jeopardy["Value"].apply(normalize_values)
    jeopardy["Air Date"] = pd.to_datetime(jeopardy["Air Date"])
    # required for value analysis
    jeopardy["high_value"] = jeopardy.apply(determine_value, axis=1)
    # print(jeopardy.head())
    return jeopardy


if __name__ == '__main__':

    # Read data set
    filename = "jeopardy.csv"
    jeopardy = read_data(filename)

    print("dataset size = {}".format(jeopardy.shape))
    jeopardy.head()

    # clean and normalize data
    jeopardy = clean_data(jeopardy)

    # do the questions contain the answers themselves?
    jeopardy["answer_in_question"] = jeopardy.apply(count_matches, axis=1)
    print("Answer in question = {0:.2f}% ".format(
        jeopardy["answer_in_question"].mean() * 100))

    # ### do questions repeat often? if yes, how often?
    overlap_mean = find_mean_question_overlap(jeopardy)
    print("Question overlap = {0:.2f}% ".format(overlap_mean * 100))

    # count usage analysis. we want to see if a given word is more used in
    # low value or high value question. we will do it for only first 25 words
    # first lets calculate observed values
    terms_used_unique = list(set(terms_used))
    words_to_check = 10
    find_and_print_chi_squared(jeopardy, terms_used_unique[:words_to_check])

    # lets now repeat the chi squared for those words which had 100 apperances
    terms_used_freq = pd.Series(sorted(terms_used)).value_counts()
    terms_used_100_times_or_more = terms_used_freq[terms_used_freq > 100]
    find_and_print_chi_squared(
        jeopardy, terms_used_100_times_or_more.index.tolist())
