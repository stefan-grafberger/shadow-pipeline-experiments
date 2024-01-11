import os
import time

import numpy
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, label_binarize
from xgboost import XGBClassifier

from ide.experiments.pipeline_utils import create_model
from ide.utils.utils import get_project_root

os.environ["TOKENIZERS_PARALLELISM"] = "False"


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

initial_start = time.time()

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
model = model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
                                hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
encoded_train_data = featurizer.fit_transform(train[['tweet']])
encoded_train_labels = label_binarize(train['anhedonia'], classes=[True, False])
model.fit(encoded_train_data, encoded_train_labels)

encoded_test_data = featurizer.transform(test[['tweet']])
encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
accuracy = model.score(encoded_test_data, encoded_test_labels)
print(f'Test accuracy is: {accuracy}')

initial_end = time.time()
print(f"initial: {(initial_end - initial_start) * 1000}")

diff_start = time.time()
print(f"recompute all train labels started")
updated_second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
updated_labels = ((first_regex | updated_second_regex) & third_regex)
featurized_updated_labels = label_binarize(updated_labels, classes=[True, False])
diff_end = time.time()
print(f"recompute all train labels: {(diff_end - diff_start) * 1000}")

diff_start = time.time()
print(f"diff computation started")
updated_second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
changed_predictions = second_regex ^ updated_second_regex
changed_indices = numpy.where(changed_predictions)[0]
second_regex_updated = updated_second_regex[changed_indices]
first_regex_updated = first_regex[changed_indices]
third_regex_updated = third_regex[changed_indices]
updated_labels = ((first_regex_updated | second_regex_updated) & third_regex_updated)
featurized_updated_diff = label_binarize(updated_labels, classes=[True, False])
encoded_train_labels[changed_indices] = featurized_updated_diff
diff_end = time.time()
print(f"diff computation: {(diff_end - diff_start) * 1000}")

assert numpy.array_equal(encoded_train_labels, featurized_updated_labels)

retrain_start = time.time()
print(f"rexecution keras started")
new_model = model = KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
                                hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
new_model.fit(encoded_train_data, encoded_train_labels)
accuracy = new_model.score(encoded_test_data, encoded_test_labels)
print(f'Test accuracy is: {accuracy}')
retrain_end = time.time()
print(f"keras training: {(retrain_end - retrain_start) * 1000}")

retrain_start = time.time()
print(f"rexecution expensive keras started")
new_model = model = KerasClassifier(model=create_model, epochs=20, batch_size=1, verbose=0,
                                hidden_layer_sizes=(27, 27,), loss="binary_crossentropy")
new_model.fit(encoded_train_data, encoded_train_labels)
accuracy = new_model.score(encoded_test_data, encoded_test_labels)
print(f'Test accuracy is: {accuracy}')
retrain_end = time.time()
print(f"keras expensive training: {(retrain_end - retrain_start) * 1000}")


retrain_start = time.time()
print(f"rexecution sgd started")
new_model = SGDClassifier(loss='log_loss', max_iter=30, n_jobs=1)
new_model.fit(encoded_train_data, encoded_train_labels)
accuracy = new_model.score(encoded_test_data, encoded_test_labels)
print(f'Test accuracy is: {accuracy}')
retrain_end = time.time()
print(f"sgd training: {(retrain_end - retrain_start) * 1000}")

retrain_start = time.time()
print(f"rexecution xgb started")
new_model = XGBClassifier(max_depth=12, tree_method='hist', n_jobs=1)
new_model.fit(encoded_train_data, encoded_train_labels)
accuracy = new_model.score(encoded_test_data, encoded_test_labels)
print(f'Test accuracy is: {accuracy}')
retrain_end = time.time()
print(f"xgb training: {(retrain_end - retrain_start) * 1000}")

retrain_start = time.time()
print(f"rexecution kn started")
new_model = KNeighborsClassifier()
new_model.fit(encoded_train_data, encoded_train_labels.ravel())
accuracy = new_model.score(encoded_test_data, encoded_test_labels)
print(f'Test accuracy is: {accuracy}')
retrain_end = time.time()
print(f"kn training: {(retrain_end - retrain_start) * 1000}")
