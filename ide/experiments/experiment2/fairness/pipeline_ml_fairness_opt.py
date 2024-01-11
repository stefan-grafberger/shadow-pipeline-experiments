import time
import warnings

import duckdb
import numpy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer, label_binarize
from scikeras.wrappers import KerasClassifier

from ide.experiments.analysis_utils import get_translate_transformer, get_slice_finder_mask_and_slice
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
    print(f"starting slice finding")
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels)
    accuracy_slice = accuracy_score(predicted_test_labels[test_mask], encoded_test_labels[test_mask])
    unfair_indices = numpy.where(test_mask)[0]
    # If this were not a hardcoded example, we would check here if the accuracy difference justifies further analyses
    detect_eval_end = time.time()
    scores["detect_eval_1"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_detect"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_eval"] = None

    fix_eval_start = time.time()
    print(f"starting translating test")
    translate_transformer = get_translate_transformer()
    unfair_diff = test.iloc[unfair_indices].reset_index(drop=True)
    translated_tweets = translate_transformer.fit_transform(unfair_diff[['tweet']])
    translated_diff_mask = translated_tweets['tweet'] != unfair_diff['tweet']
    changed_indices_translated = unfair_indices[translated_diff_mask]
    if len(changed_indices_translated) != 0:
        translated_diff = translated_tweets[translated_diff_mask].reset_index(drop=True)
        encoded_updated_translated_diff = featurizer.transform(translated_diff[['tweet']])
    else:
        print("Info: no change in current test set")
    fix_eval_fix_end = time.time()

    print(f"keras eval on translated test")
    if len(changed_indices_translated) != 0:
        predicted_translated_labels = predicted_test_labels.copy()
        predicted_translated_labels_diff = model.predict(encoded_updated_translated_diff)
        predicted_translated_labels[changed_indices_translated] = predicted_translated_labels_diff
        accuracy_translated_overall = accuracy_score(predicted_translated_labels, encoded_test_labels)
        accuracy_translated_slice = accuracy_score(predicted_translated_labels[test_mask], encoded_test_labels[test_mask])
        print(f'Overall accuracy is: {accuracy_translated_overall}')
        print(f'Slice accuracy is: {accuracy_translated_slice}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    # Explanation start
    # print(f"The original accuracy {accuracy} drops to {accuracy_translat or accuracy} when adding typos. However, "
    #       f"spellchecking can increase the accuracy to {accuracy_fix_translat or accuracy} on translated data."
    #       f"Here are some examples for differing test set records and their preprocessing")
    # TODO: Generate a nice explanation
    print(f"Accuracy overall {accuracy} but accuracy of the slice {top_slice} is only {accuracy_slice}! However, "
          f"adding a translation step can increase the accuracy to an overall accuracy of {accuracy_translated_overall}"
          f" and a slice accuracy of {accuracy_translated_slice} on the translated data.")
    all_interesting_label_switches_mask = test_mask
    all_interesting_label_switches_indices = numpy.where(all_interesting_label_switches_mask)[0]

    # Unfeaturized
    unfeaturized_original_test_diff = test.iloc[all_interesting_label_switches_indices].reset_index(drop=True)
    # unfeaturized_translat_test_diff = unfeaturized_original_test_diff.copy()
    # unfeaturized_translat_test_diff.iloc[changed_indices_translated] = translated_diff
    # unfeaturized_translat_test_diff = unfeaturized_translat_test_diff.iloc[all_interesting_label_switches_indices].reset_index(drop=True)
    # This is faster because all interesting switches are the same as test_mask
    unfeaturized_translat_test_diff = translated_tweets

    # Featurized
    featurized_original_diff = encoded_test_data[all_interesting_label_switches_indices]
    encoded_translated_data = encoded_test_data.copy()
    encoded_translated_data[changed_indices_translated] = encoded_updated_translated_diff
    featurized_translat_diff = encoded_translated_data[all_interesting_label_switches_indices]

    # Labels
    predicted_original_labels_diff = predicted_test_labels[all_interesting_label_switches_indices]
    predicted_translat_labels_diff = predicted_translated_labels[all_interesting_label_switches_indices]
    true_labels_diff = encoded_test_labels[all_interesting_label_switches_indices]

    explanation_df = unfeaturized_original_test_diff
    explanation_df['translated_tweet'] = unfeaturized_translat_test_diff['tweet']
    explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff))
    explanation_df['featurized_translated_tweet'] = pd.Series(list(featurized_translat_diff))
    explanation_df['predicted_label'] = pd.Series(list(predicted_original_labels_diff))
    explanation_df['predicted_translated_label'] = pd.Series(list(predicted_translat_labels_diff))
    explanation_df['true_label'] = pd.Series(list(true_labels_diff))
    print(explanation_df)
    broken_predictions = explanation_df[
        (explanation_df['predicted_label'] == explanation_df['true_label']) &
        (explanation_df['predicted_label'] != explanation_df['predicted_translated_label'])]
    print(broken_predictions)
    successfully_fixed = explanation_df[
        (explanation_df['predicted_label'] != explanation_df['true_label']) &
        (explanation_df['predicted_translated_label'] == explanation_df['true_label'])]
    print(successfully_fixed)
    nothing_changed = explanation_df[
        (explanation_df['predicted_label'] != explanation_df['true_label']) &
        (explanation_df['predicted_translated_label'] != explanation_df['true_label'])]
    print(nothing_changed)
    # TODO: More analyses on explanation df?

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
    new_model = model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
                                    hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
    new_model.fit(encoded_train_data, encoded_train_labels)
    accuracy = new_model.score(encoded_test_data, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')
    original_pipeline_diff_end = time.time()
    scores["original_pipeline_diff"] = (original_pipeline_diff_end - original_pipeline_diff_start) * 1000
    scores["original_pipeline_diff_preprocess"] = (original_pipeline_diff_preprocessing_end - original_pipeline_diff_start) * 1000
    scores["original_pipeline_diff_eval"] = (original_pipeline_diff_end - original_pipeline_diff_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting slice finding")
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels)
    accuracy_slice = accuracy_score(predicted_test_labels[test_mask], encoded_test_labels[test_mask])
    unfair_indices = numpy.where(test_mask)[0]
    # TODO: If this were not a hardcoded example, we would check here if the accuracy difference justifies further analyses
    detect_eval_end = time.time()
    scores["detect_eval_2"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_detect"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_eval"] = None

    fix_eval_start = time.time()
    print(f"starting translating test")
    already_translated_unfeaturized = translated_tweets.copy()
    already_translated_unfeaturized['test_id'] = unfair_indices
    already_translated_unfeaturized["before_translation"] = unfair_diff['tweet']

    already_translated_featurized = pd.DataFrame({'encoded_translated': pd.Series(list(encoded_updated_translated_diff))})
    already_translated_featurized['test_id'] = translated_diff_mask
    # already_translated_featurized_data = encoded_updated_translated_diff
    # already_translated_featurized_ids = pd.DataFrame({'test_id': translated_diff_mask})
    # TODO: It might be better not to use pandas here. Instead, use a combination of numpy and pandas: id in pandas df,
    #  actual data in numpy
    warnings.filterwarnings('ignore')
    # do not run this if not necessary
    unfair_indices_df = pd.DataFrame({"test_id": unfair_indices})
    not_translated_yet = duckdb.sql("""
        SELECT u.test_id
        FROM unfair_indices_df u ANTI JOIN already_translated_unfeaturized a ON u.test_id = a.test_id
    """).df()
    already_translated = duckdb.sql("""
        SELECT u.test_id, a.tweet, a.before_translation
        FROM unfair_indices_df u JOIN already_translated_unfeaturized a ON u.test_id = a.test_id
    """).df()
    unfair_diff = test.iloc[not_translated_yet['test_id']].reset_index(drop=True)
    not_translated_yet['before_translation'] = unfair_diff['tweet']
    if unfair_diff.shape[0] != 0:
        not_translated_yet['tweet'] = translate_transformer.fit_transform(unfair_diff[['tweet']])
    translated_tweets = pd.concat([not_translated_yet, already_translated])
    unfair_indices = translated_tweets['test_id']

    # translated_diff_mask = all_translated_tweets['tweet'] != all_translated_tweets['before_translation']
    # changed_indices_translated = unfair_indices[translated_diff_mask]
    changed_indices_translated_df = duckdb.sql("""
        SELECT test_id, tweet
        FROM translated_tweets
        WHERE tweet != before_translation
    """).df()
    changed_indices_translated = changed_indices_translated_df['test_id']
    if len(changed_indices_translated) != 0:
        # translated_diff = translated_tweets[translated_diff_mask].reset_index(drop=True)
        # encoded_updated_translated_diff = featurizer.transform(translated_diff[['tweet']])
        not_featurized_yet = duckdb.sql("""
            SELECT c.test_id, tweet
            FROM changed_indices_translated_df c ANTI JOIN already_translated_featurized a ON c.test_id = a.test_id
        """).df()
        already_featurized = duckdb.sql("""
            SELECT c.test_id, a.encoded_translated
            FROM changed_indices_translated_df c JOIN already_translated_featurized a ON c.test_id = a.test_id
        """).df()
        not_featurized_yet['encoded_translated'] = pd.Series(list(featurizer.transform(not_featurized_yet[['tweet']])))
        not_featurized_yet = not_featurized_yet[['test_id', 'encoded_translated']]
        all_featurized_df = pd.concat([not_featurized_yet, already_featurized])
        # duckdb converts encoded_translated to float64 list from numpy 32 arrays, this requires a float32 cast
        encoded_updated_translated_diff = numpy.float32(numpy.stack(all_featurized_df['encoded_translated']))
        changed_indices_translated = all_featurized_df['test_id']
    else:
        print("Info: no change in current test set")
    fix_eval_fix_end = time.time()

    print(f"keras eval on translated test")
    if len(changed_indices_translated) != 0:
        predicted_translated_labels = predicted_test_labels.copy()
        predicted_translated_labels_diff = model.predict(encoded_updated_translated_diff)
        predicted_translated_labels[changed_indices_translated] = predicted_translated_labels_diff
        accuracy_translated_overall = accuracy_score(predicted_translated_labels, encoded_test_labels)
        accuracy_translated_slice = accuracy_score(predicted_translated_labels[test_mask], encoded_test_labels[test_mask])
        print(f'Overall accuracy is: {accuracy_translated_overall}')
        print(f'Slice accuracy is: {accuracy_translated_slice}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_2"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_2_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_2_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    # Explanation start
    # print(f"The original accuracy {accuracy} drops to {accuracy_translat or accuracy} when adding typos. However, "
    #       f"spellchecking can increase the accuracy to {accuracy_fix_translat or accuracy} on translated data."
    #       f"Here are some examples for differing test set records and their preprocessing")
    print(f"Accuracy overall {accuracy} but accuracy of the slice {top_slice} is only {accuracy_slice}! However, "
          f"adding a translation step can increase the accuracy to an overall accuracy of {accuracy_translated_overall}"
          f" and a slice accuracy of {accuracy_translated_slice} on the translated data.")
    all_interesting_label_switches_mask = test_mask
    all_interesting_label_switches_indices = numpy.where(all_interesting_label_switches_mask)[0]

    # Unfeaturized
    unfeaturized_original_test_diff = test.iloc[all_interesting_label_switches_indices].reset_index(drop=True)
    # unfeaturized_translat_test_diff = unfeaturized_original_test_diff.copy()
    # unfeaturized_translat_test_diff.iloc[changed_indices_translated] = translated_diff
    # unfeaturized_translat_test_diff = unfeaturized_translat_test_diff.iloc[all_interesting_label_switches_indices].reset_index(drop=True)
    # This is faster because all interesting switches are the same as test_mask
    unfeaturized_translat_test_diff = translated_tweets

    # Featurized
    featurized_original_diff = encoded_test_data[all_interesting_label_switches_indices]
    encoded_translated_data = encoded_test_data.copy()
    encoded_translated_data[changed_indices_translated] = encoded_updated_translated_diff
    featurized_translat_diff = encoded_translated_data[all_interesting_label_switches_indices]

    # Labels
    predicted_original_labels_diff = predicted_test_labels[all_interesting_label_switches_indices]
    predicted_translat_labels_diff = predicted_translated_labels[all_interesting_label_switches_indices]
    true_labels_diff = encoded_test_labels[all_interesting_label_switches_indices]

    explanation_df = unfeaturized_original_test_diff
    explanation_df['translated_tweet'] = unfeaturized_translat_test_diff['tweet']
    explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff))
    explanation_df['featurized_translated_tweet'] = pd.Series(list(featurized_translat_diff))
    explanation_df['predicted_label'] = pd.Series(list(predicted_original_labels_diff))
    explanation_df['predicted_translated_label'] = pd.Series(list(predicted_translat_labels_diff))
    explanation_df['true_label'] = pd.Series(list(true_labels_diff))
    print(explanation_df)
    broken_predictions = explanation_df[
        (explanation_df['predicted_label'] == explanation_df['true_label']) &
        (explanation_df['predicted_label'] != explanation_df['predicted_translated_label'])]
    print(broken_predictions)
    successfully_fixed = explanation_df[
        (explanation_df['predicted_label'] != explanation_df['true_label']) &
        (explanation_df['predicted_translated_label'] == explanation_df['true_label'])]
    print(successfully_fixed)
    nothing_changed = explanation_df[
        (explanation_df['predicted_label'] != explanation_df['true_label']) &
        (explanation_df['predicted_translated_label'] != explanation_df['true_label'])]
    print(nothing_changed)

    explanation_end = time.time()
    scores["explanation_2"] = (explanation_end - explanation_start) * 1000
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
