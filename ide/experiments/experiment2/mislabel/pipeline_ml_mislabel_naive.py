import time

import numpy
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
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

    def load_train_data(user_location, tweet_location, included_countries):
        users = pd.read_parquet(user_location)
        users = users[users.country.isin(included_countries)]
        tweets = pd.read_parquet(tweet_location)
        return users.merge(tweets, on='user_id')

    def encode_features():
        # model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
        model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
        embedder = FunctionTransformer(lambda item: model.encode(item))
        preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
        return preprocessor

    user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
    tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
    nl_be = ['NL', 'BE']
    test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

    train = load_train_data(user_location, tweet_location, included_countries=nl_be)
    # train = weak_labeling(train)
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
    detect_eval_detect_end = time.time()
    model.fit(encoded_train_data, encoded_train_labels)
    predicted_test_labels = model.predict(encoded_test_data)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')
    detect_eval_end = time.time()
    scores["detect_eval_1"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_1_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"starting retraining and evaluation")
    if len(unfair_indices) != 0:
        def load_train_data(user_location, tweet_location, included_countries):
            users = pd.read_parquet(user_location)
            users = users[users.country.isin(included_countries)]
            tweets = pd.read_parquet(tweet_location)
            return users.merge(tweets, on='user_id')

        def encode_features():
            # model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
            model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
            embedder = FunctionTransformer(lambda item: model.encode(item))
            preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
            return preprocessor

        user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
        tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
        nl_be = ['NL', 'BE']
        test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

        train = load_train_data(user_location, tweet_location, included_countries=nl_be)
        # train = weak_labeling(train)
        first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
        second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (interest|pleasure|motivation)',
                                                   regex=True)
        third_regex = ~(
            train['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
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
        encoded_train_labels[unfair_indices, :] = 1 - encoded_train_labels[unfair_indices, :]

        fix_eval_fix_end = time.time()
        model.fit(encoded_train_data, encoded_train_labels)
        predicted_test_labels = model.predict(encoded_test_data)
        accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
        print(f'Updated test accuracy is: {accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    scores["explanation_1"] = None

    # Pipeline change starting here
    original_pipeline_start = time.time()

    def load_train_data(user_location, tweet_location, included_countries):
        users = pd.read_parquet(user_location)
        users = users[users.country.isin(included_countries)]
        tweets = pd.read_parquet(tweet_location)
        return users.merge(tweets, on='user_id')

    def encode_features():
        # model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
        model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
        embedder = FunctionTransformer(lambda item: model.encode(item))
        preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
        return preprocessor

    user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
    tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
    nl_be = ['NL', 'BE']
    test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

    train = load_train_data(user_location, tweet_location, included_countries=nl_be)
    # train = weak_labeling(train)
    first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
    second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
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
    scores["original_pipeline_diff"] = (original_pipeline_end - original_pipeline_start) * 1000
    scores["original_pipeline_diff_preprocess"] = (original_pipeline_preprocessing_end - original_pipeline_start) * 1000
    scores["original_pipeline_diff_eval"] = (original_pipeline_end - original_pipeline_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting shapley calculation")

    def load_train_data(user_location, tweet_location, included_countries):
        users = pd.read_parquet(user_location)
        users = users[users.country.isin(included_countries)]
        tweets = pd.read_parquet(tweet_location)
        return users.merge(tweets, on='user_id')

    def encode_features():
        # model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
        model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
        embedder = FunctionTransformer(lambda item: model.encode(item))
        preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
        return preprocessor

    user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
    tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
    nl_be = ['NL', 'BE']
    test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

    train = load_train_data(user_location, tweet_location, included_countries=nl_be)
    # train = weak_labeling(train)
    first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
    second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
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
    detect_eval_detect_end = time.time()
    model.fit(encoded_train_data, encoded_train_labels)
    predicted_test_labels = model.predict(encoded_test_data)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')
    detect_eval_end = time.time()
    scores["detect_eval_2"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_2_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"starting retraining and evaluation")
    if len(unfair_indices) != 0:
        def load_train_data(user_location, tweet_location, included_countries):
            users = pd.read_parquet(user_location)
            users = users[users.country.isin(included_countries)]
            tweets = pd.read_parquet(tweet_location)
            return users.merge(tweets, on='user_id')

        def encode_features():
            # model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
            model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
            embedder = FunctionTransformer(lambda item: model.encode(item))
            preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
            return preprocessor

        user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
        tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
        nl_be = ['NL', 'BE']
        test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

        train = load_train_data(user_location, tweet_location, included_countries=nl_be)
        # train = weak_labeling(train)
        first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
        second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
        third_regex = ~(
            train['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
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
        encoded_train_labels[unfair_indices, :] = 1 - encoded_train_labels[unfair_indices, :]

        fix_eval_fix_end = time.time()
        model.fit(encoded_train_data, encoded_train_labels)
        predicted_test_labels = model.predict(encoded_test_data)
        accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
        print(f'Updated test accuracy is: {accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_2"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_2_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_2_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    scores["explanation_2"] = None
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
