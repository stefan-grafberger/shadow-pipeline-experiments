import time

import numpy
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer, label_binarize

from ide.experiments.analysis_utils import get_typo_adder, get_typo_fixer
from ide.experiments.pipeline_utils import create_model, initialize_environment
from ide.utils.utils import get_project_root



def execute():
    initialize_environment()

    scores = {}
    original_pipeline_start = time.time()

    def load_train_data(user_location, tweet_location, included_countries):
        users = pd.read_parquet(user_location)
        users = users[users.country.isin(included_countries)]
        tweets = pd.read_parquet(tweet_location)
        return users.merge(tweets, on='user_id')

    def encode_features():
        #model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
        model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
        embedder = FunctionTransformer(lambda item: model.encode(item))
        preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
        return preprocessor

    user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
    tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
    nl_be = ['NL', 'BE']
    test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

    train = load_train_data(user_location, tweet_location, included_countries=nl_be)
    #train = weak_labeling(train)
    first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
    second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (interest|pleasure|motivation)', regex=True)
    third_regex = ~(train['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
    train['anhedonia'] = ((first_regex | second_regex) & third_regex)
    #
    test = pd.read_parquet(test_location)

    # estimator = Pipeline([
    #     ('features', encode_features()),
    #     ('learner', MyKerasClassifier(build_fn=create_model, epochs=3, batch_size=32, verbose=0))])

    featurizer = encode_features()
    # TODO: What model settings are realistic?
    model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
                                    hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
    encoded_train_data = featurizer.fit_transform(train[['tweet']])
    encoded_train_labels = label_binarize(train['anhedonia'], classes=[True, False])
    encoded_test_data = featurizer.transform(test[['tweet']])
    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
    original_pipeline_preprocessing_end = time.time()

    model.fit(encoded_train_data, encoded_train_labels)
    predicted_test_labels = model.predict(encoded_test_data)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')

    original_pipeline_end = time.time()
    scores["original_pipeline"] = (original_pipeline_end - original_pipeline_start) * 1000
    scores["original_pipeline_preprocess"] = (original_pipeline_preprocessing_end - original_pipeline_start) * 1000
    scores["original_pipeline_eval"] = (original_pipeline_end - original_pipeline_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting corrupting test")

    typo_adder = get_typo_adder()
    corrupted_tweet = typo_adder.fit_transform(test[['tweet']])
    corrupt_diff_mask = corrupted_tweet['tweet'] != test['tweet']
    changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
    if len(changed_indices_corrupt) != 0:
        corrupted_diff = corrupted_tweet.iloc[changed_indices_corrupt].reset_index(drop=True)
        encoded_updated_corrupted_diff = featurizer.transform(corrupted_diff[['tweet']])
    else:
        print("Info: no change in current test set")
    detect_eval_detect_end = time.time()

    diff_start = time.time()
    print(f"keras eval on corrupted test")
    if len(changed_indices_corrupt) != 0:
        predicted_corrupt_labels = predicted_test_labels.copy()
        predicted_corrupt_labels_diff = model.predict(encoded_updated_corrupted_diff)
        predicted_corrupt_labels[changed_indices_corrupt] = predicted_corrupt_labels_diff
        accuracy_corrupt = accuracy_score(predicted_corrupt_labels, encoded_test_labels)
        print(f'Test accuracy is: {accuracy_corrupt}')
    else:
        print("Info: no change in current test set")
    detect_eval_end = time.time()
    scores["detect_eval_1"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_1_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"fix typos in corrupted test")
    if len(changed_indices_corrupt) != 0:
        typo_fixer = get_typo_fixer()
        fixed_corrupted_tweets_diff = typo_fixer.fit_transform(corrupted_diff[['tweet']])
        corrupt_fix_diff_mask = fixed_corrupted_tweets_diff['tweet'] != corrupted_diff['tweet']
        changed_indices_fix_corrupt = changed_indices_corrupt[corrupt_fix_diff_mask]
        fixed_typos_diff = fixed_corrupted_tweets_diff[corrupt_fix_diff_mask].reset_index(drop=True)
        encoded_updated_fixed_diff = featurizer.transform(fixed_typos_diff)
    else:
        print("Info: no change in current test set")
    fix_eval_fix_end = time.time()
    print(f"keras eval on fixed test")
    if len(changed_indices_fix_corrupt) != 0:
        predicted_fix_labels = predicted_corrupt_labels.copy()
        predicted_fix_labels_diff = model.predict(encoded_updated_fixed_diff)
        predicted_fix_labels[changed_indices_fix_corrupt] = predicted_fix_labels_diff
        accuracy_fix_corrupt = accuracy_score(predicted_fix_labels, encoded_test_labels)
        print(f'Test accuracy is: {accuracy_fix_corrupt}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    # Explanation start
    print(f"The original accuracy {accuracy} drops to {accuracy_corrupt or accuracy} when adding typos. However, "
          f"spellchecking can increase the accuracy to {accuracy_fix_corrupt or accuracy} on corrupted data. "
          f"Here are some examples for differing test set records and their preprocessing")
    all_interesting_label_switches_mask = corrupt_diff_mask # | corrupt_fix_diff_mask We only fix now corrupted vals now
    all_interesting_label_switches_indices = numpy.where(all_interesting_label_switches_mask)[0]

    # Unfeaturized
    unfeaturized_original_test_diff = test.iloc[all_interesting_label_switches_indices].reset_index(drop=True)
    unfeaturized_corrupt_test_diff = corrupted_tweet.iloc[all_interesting_label_switches_indices].reset_index(drop=True)
    fixed_corrupted_tweets = corrupted_tweet.copy()
    fixed_corrupted_tweets['tweet'][changed_indices_fix_corrupt] = fixed_typos_diff['tweet']
    unfeaturized_fix_corrupt_test_diff = fixed_corrupted_tweets.iloc[all_interesting_label_switches_indices].reset_index(drop=True)

    # Featurized
    featurized_original_diff = encoded_test_data[all_interesting_label_switches_indices]
    encoded_corrupted_data = encoded_test_data.copy()
    encoded_corrupted_data[corrupt_diff_mask] = encoded_updated_corrupted_diff
    featurized_corrupt_diff = encoded_corrupted_data[all_interesting_label_switches_indices]
    encoded_fix_corrupted_data = encoded_corrupted_data.copy()
    encoded_fix_corrupted_data[changed_indices_fix_corrupt] = encoded_updated_fixed_diff
    featurized_fix_corrupt_diff = encoded_fix_corrupted_data[all_interesting_label_switches_indices]
    # Labels
    predicted_original_labels_diff = predicted_test_labels[all_interesting_label_switches_indices]
    predicted_corrupt_labels_diff = predicted_corrupt_labels[all_interesting_label_switches_indices]
    predicted_fix_corrupt_labels_diff = predicted_fix_labels[all_interesting_label_switches_indices]
    true_labels_diff = encoded_test_labels[all_interesting_label_switches_indices]

    explanation_df = unfeaturized_original_test_diff
    explanation_df['corrupted_tweet'] = unfeaturized_corrupt_test_diff['tweet']
    explanation_df['fix_corrupted_tweet'] = unfeaturized_fix_corrupt_test_diff['tweet']
    explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff))
    explanation_df['featurized_corrupted_tweet'] = pd.Series(list(featurized_corrupt_diff))
    explanation_df['featurized_fix_corrupted_tweet'] = pd.Series(list(featurized_fix_corrupt_diff))
    explanation_df['predicted_label'] = pd.Series(list(predicted_original_labels_diff))
    explanation_df['predicted_corrupt_label'] = pd.Series(list(predicted_corrupt_labels_diff))
    explanation_df['predicted_fix_corrupt_label'] = pd.Series(list(predicted_fix_corrupt_labels_diff))
    explanation_df['true_label'] = pd.Series(list(true_labels_diff))
    print(explanation_df)
    corruption_effect = explanation_df[
        (explanation_df['predicted_label'] == explanation_df['true_label']) &
        (explanation_df['predicted_label'] != explanation_df['predicted_corrupt_label'])]
    print(corruption_effect)
    successfully_fixed = explanation_df[
        (explanation_df['predicted_label'] == explanation_df['true_label']) &
        (explanation_df['predicted_label'] != explanation_df['predicted_corrupt_label']) &
        (explanation_df['predicted_label'] == explanation_df['predicted_fix_corrupt_label'])]
    print(successfully_fixed)
    worsened_prediction = explanation_df[
        (explanation_df['predicted_label'] == explanation_df['true_label']) &
        (explanation_df['predicted_label'] != explanation_df['predicted_fix_corrupt_label'])]
    print(worsened_prediction)
    otherwise_fixed = explanation_df[
        (explanation_df['predicted_label'] != explanation_df['true_label']) &
        (explanation_df['predicted_fix_corrupt_label'] == explanation_df['true_label'])]
    print(otherwise_fixed)

    explanation_end = time.time()
    scores["explanation_1"] = (explanation_end - explanation_start) * 1000
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
