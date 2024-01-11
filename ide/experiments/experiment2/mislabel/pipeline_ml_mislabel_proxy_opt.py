import time

import numpy
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer, label_binarize

from ide.experiments.analysis_utils import get_shapley_value_result
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
    print(f"starting shapley calculation")
    # TODO: What fraction do we want to use?

    # Trying to sample test set to improve performance
    indices = numpy.arange(len(encoded_train_labels))
    numpy.random.shuffle(indices)

    train_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_train_labels) * train_fraction_to_consider)
    train_indices_to_consider = indices[:num_values_to_typo]
    train_data_sample = encoded_train_data[train_indices_to_consider]
    train_label_sample = encoded_train_labels[train_indices_to_consider]

    indices = numpy.arange(len(encoded_test_labels))
    test_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
    test_indices_to_consider = indices[:num_values_to_typo]
    test_data_sample = encoded_test_data[test_indices_to_consider]
    test_label_sample = encoded_test_labels[test_indices_to_consider]

    rows_to_fix, cleaning_batch_size = get_shapley_value_result(train_data_sample, train_label_sample, test_data_sample,
                                                                test_label_sample, train_indices_to_consider)
    unfair_indices = rows_to_fix['train_id']
    selected_shapley_values = rows_to_fix['shapley_value'].reset_index(drop=True)
    detect_eval_end = time.time()
    scores["detect_eval_1"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_detect"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_eval"] = None

    fix_eval_start = time.time()
    print(f"starting retraining and evaluation")
    if len(unfair_indices) != 0:
        modified_encoded_train_labels = encoded_train_labels.copy()
        modified_encoded_train_labels[unfair_indices, :] = 1 - modified_encoded_train_labels[unfair_indices, :]
        fix_eval_fix_end = time.time()
        original_proxy_model = SGDClassifier(loss='log_loss', max_iter=30, n_jobs=1)
        original_proxy_model.fit(encoded_train_data, encoded_train_labels)
        predicted_proxy_test_labels = original_proxy_model.predict(encoded_test_data).reshape(-1, 1)
        accuracy_proxy = accuracy_score(predicted_test_labels, encoded_test_labels)

        modified_model = SGDClassifier(loss='log_loss', max_iter=30, n_jobs=1)
        # modified_model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
        #                                  hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
        modified_model.fit(encoded_train_data, modified_encoded_train_labels)
        modified_predicted_test_labels = modified_model.predict(encoded_test_data).reshape(-1, 1)
        modified_accuracy = accuracy_score(modified_predicted_test_labels, encoded_test_labels)
        print(f'Updated test accuracy is: {modified_accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    print(f"Accuracy overall {accuracy} but likely mislabeled records were detected. A cheap proxy model was used."
          f"The accuracy of the proxy model is {accuracy_proxy}. By flipping the labels of the top "
          f"{cleaning_batch_size} most likely mislabed records, the accuracy was changed to {modified_accuracy}! "
          f"Consider changing the weak labeling to account for these records.")
    all_interesting_label_switches_indices_train = unfair_indices

    # Explanation df train
    # Unfeaturized
    unfeaturized_original_train_diff = train.iloc[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Featurized
    featurized_original_diff = encoded_train_data[all_interesting_label_switches_indices_train]
    regex_1_diff = first_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_2_diff = second_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_3_diff = third_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Labels unencoded
    original_labels_diff = unfeaturized_original_train_diff['anhedonia']
    modified_labels_diff = original_labels_diff.copy()
    modified_labels_diff = ~modified_labels_diff

    # labels encoded
    original_encoded_labels_diff = encoded_train_labels[all_interesting_label_switches_indices_train]
    modified_encoded_labels_diff = modified_encoded_train_labels[all_interesting_label_switches_indices_train]

    train_explanation_df = unfeaturized_original_train_diff[['user_id', 'lang', 'country', 'tweet']]
    train_explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff))
    train_explanation_df['regex_1'] = regex_1_diff
    train_explanation_df['regex_2'] = regex_2_diff
    train_explanation_df['regex_3'] = regex_3_diff
    train_explanation_df['original_label'] = original_labels_diff
    train_explanation_df['modified_label'] = modified_labels_diff
    train_explanation_df['encoded_original_label'] = pd.Series(list(original_encoded_labels_diff))
    train_explanation_df['encoded_modified_label'] = pd.Series(list(modified_encoded_labels_diff))
    train_explanation_df['shapley_value'] = selected_shapley_values

    print(train_explanation_df)

    # Test explanation df
    flipped_predictions_mask = predicted_test_labels != modified_predicted_test_labels
    all_interesting_label_switches_indices_test = numpy.where(flipped_predictions_mask)[0]

    unfeaturized_original_test_diff = test.iloc[all_interesting_label_switches_indices_test].reset_index(drop=True)

    # Featurized
    featurized_original_diff_test = encoded_test_data[all_interesting_label_switches_indices_test]

    # Labels unencoded
    original_true_label_diff_test = unfeaturized_original_test_diff['anhedonia']

    # labels encoded
    original_encoded_labels_diff_test = predicted_test_labels[all_interesting_label_switches_indices_test]
    proxy_encoded_test_labels_diff_test = predicted_proxy_test_labels[all_interesting_label_switches_indices_test]
    modified_encoded_labels_diff_test = modified_predicted_test_labels[all_interesting_label_switches_indices_test]
    true_encoded_labels_diff_test = encoded_test_labels[all_interesting_label_switches_indices_test]

    test_explanation_df = unfeaturized_original_test_diff[['user_id', 'lang', 'country', 'tweet']]
    test_explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff_test))
    test_explanation_df['true_label'] = original_true_label_diff_test
    test_explanation_df['encoded_original_predicted_label'] = pd.Series(list(original_encoded_labels_diff_test))
    test_explanation_df['encoded_proxy_original_predicted_label'] = pd.Series(list(proxy_encoded_test_labels_diff_test))
    test_explanation_df['encoded_proxy_modified_predicted_label'] = pd.Series(list(modified_encoded_labels_diff_test))
    test_explanation_df['encoded_true_label'] = pd.Series(list(true_encoded_labels_diff_test))

    print(test_explanation_df)
    broken_predictions = test_explanation_df[
        (test_explanation_df['encoded_proxy_original_predicted_label'] == test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_proxy_original_predicted_label'] != test_explanation_df['encoded_proxy_modified_predicted_label'])]
    print(broken_predictions)
    successfully_fixed = test_explanation_df[
        (test_explanation_df['encoded_proxy_original_predicted_label'] != test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_proxy_modified_predicted_label'] == test_explanation_df['encoded_true_label'])]
    print(successfully_fixed)

    explanation_end = time.time()
    scores["explanation_1"] = (explanation_end - explanation_start) * 1000

    # Now simulating a pipeline change, the second regex gets updated and the word interest gets removed from it
    original_pipeline_diff_start = time.time()
    print(f"diff preprocessing computation started")
    updated_second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
    changed_predictions = second_regex ^ updated_second_regex
    changed_indices = numpy.where(changed_predictions)[0]
    second_regex_updated = updated_second_regex[changed_indices]
    first_regex_updated = first_regex[changed_indices]
    third_regex_updated = third_regex[changed_indices]
    updated_labels = ((first_regex_updated | second_regex_updated) & third_regex_updated)
    featurized_updated_diff = label_binarize(updated_labels, classes=[True, False])
    encoded_train_labels[changed_indices] = featurized_updated_diff
    original_pipeline_diff_preprocessing_end = time.time()
    print(f"rexecution keras started")
    new_model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
                                hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
    new_model.fit(encoded_train_data, encoded_train_labels)
    accuracy = new_model.score(encoded_test_data, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')
    original_pipeline_diff_end = time.time()
    scores["original_pipeline_diff"] = (original_pipeline_diff_end - original_pipeline_diff_start) * 1000
    scores["original_pipeline_diff_preprocess"] = (original_pipeline_diff_preprocessing_end - original_pipeline_diff_start) * 1000
    scores["original_pipeline_diff_eval"] = (original_pipeline_diff_end - original_pipeline_diff_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting shapley calculation")
    # TODO: What fraction do we want to use?

    # Trying to sample test set to improve performance
    indices = numpy.arange(len(encoded_train_labels))
    numpy.random.shuffle(indices)

    train_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_train_labels) * train_fraction_to_consider)
    train_indices_to_consider = indices[:num_values_to_typo]
    train_data_sample = encoded_train_data[train_indices_to_consider]
    train_label_sample = encoded_train_labels[train_indices_to_consider]

    indices = numpy.arange(len(encoded_test_labels))
    test_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
    test_indices_to_consider = indices[:num_values_to_typo]
    test_data_sample = encoded_test_data[test_indices_to_consider]
    test_label_sample = encoded_test_labels[test_indices_to_consider]

    rows_to_fix, cleaning_batch_size = get_shapley_value_result(train_data_sample, train_label_sample, test_data_sample,
                                                                test_label_sample, train_indices_to_consider)
    unfair_indices = rows_to_fix['train_id']
    selected_shapley_values = rows_to_fix['shapley_value'].reset_index(drop=True)
    detect_eval_end = time.time()
    scores["detect_eval_2"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_detect"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_eval"] = None

    fix_eval_start = time.time()
    print(f"starting retraining and evaluation")
    if len(unfair_indices) != 0:
        modified_encoded_train_labels = encoded_train_labels.copy()
        modified_encoded_train_labels[unfair_indices, :] = 1 - modified_encoded_train_labels[unfair_indices, :]
        fix_eval_fix_end = time.time()
        original_proxy_model = SGDClassifier(loss='log_loss', max_iter=30, n_jobs=1)
        original_proxy_model.fit(encoded_train_data, encoded_train_labels)
        predicted_proxy_test_labels = original_proxy_model.predict(encoded_test_data).reshape(-1, 1)
        accuracy_proxy = accuracy_score(predicted_test_labels, encoded_test_labels)

        modified_model = SGDClassifier(loss='log_loss', max_iter=30, n_jobs=1)
        # modified_model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
        #                                  hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
        modified_model.fit(encoded_train_data, modified_encoded_train_labels)
        modified_predicted_test_labels = modified_model.predict(encoded_test_data).reshape(-1, 1)
        modified_accuracy = accuracy_score(modified_predicted_test_labels, encoded_test_labels)
        print(f'Updated test accuracy is: {modified_accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_2"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_2_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_2_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    print(f"Accuracy overall {accuracy} but likely mislabeled records were detected. A cheap proxy model was used."
          f"The accuracy of the proxy model is {accuracy_proxy}. By flipping the labels of the top "
          f"{cleaning_batch_size} most likely mislabed records, the accuracy was changed to {modified_accuracy}! "
          f"Consider changing the weak labeling to account for these records.")
    all_interesting_label_switches_indices_train = unfair_indices

    # Explanation df train
    # Unfeaturized
    unfeaturized_original_train_diff = train.iloc[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Featurized
    featurized_original_diff = encoded_train_data[all_interesting_label_switches_indices_train]
    regex_1_diff = first_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_2_diff = second_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_3_diff = third_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Labels unencoded
    original_labels_diff = unfeaturized_original_train_diff['anhedonia']
    modified_labels_diff = original_labels_diff.copy()
    modified_labels_diff = ~modified_labels_diff

    # labels encoded
    original_encoded_labels_diff = encoded_train_labels[all_interesting_label_switches_indices_train]
    modified_encoded_labels_diff = modified_encoded_train_labels[all_interesting_label_switches_indices_train]

    train_explanation_df = unfeaturized_original_train_diff[['user_id', 'lang', 'country', 'tweet']]
    train_explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff))
    train_explanation_df['regex_1'] = regex_1_diff
    train_explanation_df['regex_2'] = regex_2_diff
    train_explanation_df['regex_3'] = regex_3_diff
    train_explanation_df['original_label'] = original_labels_diff
    train_explanation_df['modified_label'] = modified_labels_diff
    train_explanation_df['encoded_original_label'] = pd.Series(list(original_encoded_labels_diff))
    train_explanation_df['encoded_modified_label'] = pd.Series(list(modified_encoded_labels_diff))
    train_explanation_df['shapley_value'] = selected_shapley_values

    print(train_explanation_df)

    # Test explanation df
    flipped_predictions_mask = predicted_test_labels != modified_predicted_test_labels
    all_interesting_label_switches_indices_test = numpy.where(flipped_predictions_mask)[0]

    unfeaturized_original_test_diff = test.iloc[all_interesting_label_switches_indices_test].reset_index(drop=True)

    # Featurized
    featurized_original_diff_test = encoded_test_data[all_interesting_label_switches_indices_test]

    # Labels unencoded
    original_true_label_diff_test = unfeaturized_original_test_diff['anhedonia']

    # labels encoded
    original_encoded_labels_diff_test = predicted_test_labels[all_interesting_label_switches_indices_test]
    proxy_encoded_test_labels_diff_test = predicted_proxy_test_labels[all_interesting_label_switches_indices_test]
    modified_encoded_labels_diff_test = modified_predicted_test_labels[all_interesting_label_switches_indices_test]
    true_encoded_labels_diff_test = encoded_test_labels[all_interesting_label_switches_indices_test]

    test_explanation_df = unfeaturized_original_test_diff[['user_id', 'lang', 'country', 'tweet']]
    test_explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff_test))
    test_explanation_df['true_label'] = original_true_label_diff_test
    test_explanation_df['encoded_original_predicted_label'] = pd.Series(list(original_encoded_labels_diff_test))
    test_explanation_df['encoded_proxy_original_predicted_label'] = pd.Series(list(proxy_encoded_test_labels_diff_test))
    test_explanation_df['encoded_proxy_modified_predicted_label'] = pd.Series(list(modified_encoded_labels_diff_test))
    test_explanation_df['encoded_true_label'] = pd.Series(list(true_encoded_labels_diff_test))

    print(test_explanation_df)
    broken_predictions = test_explanation_df[
        (test_explanation_df['encoded_proxy_original_predicted_label'] == test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_proxy_original_predicted_label'] != test_explanation_df['encoded_proxy_modified_predicted_label'])]
    print(broken_predictions)
    successfully_fixed = test_explanation_df[
        (test_explanation_df['encoded_proxy_original_predicted_label'] != test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_proxy_modified_predicted_label'] == test_explanation_df['encoded_true_label'])]
    print(successfully_fixed)

    explanation_end = time.time()
    scores["explanation_2"] = (explanation_end - explanation_start) * 1000
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
