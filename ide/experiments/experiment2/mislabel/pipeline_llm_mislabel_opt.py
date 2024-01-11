import time
from functools import partial

import duckdb
import numpy
import pandas
import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from ide.experiments.analysis_utils import wait_llm_call, get_shapley_value_result
from ide.experiments.pipeline_utils import get_langchain_rag_binary_classification_with_retrieval_tracking, \
    initialize_environment
from ide.utils.utils import get_project_root

def execute():
    initialize_environment()
    scores = {}
    original_pipeline_start = time.time()

    boolean_dictionary = {True: 'anhedonia', False: 'regular'}

    def load_train_data(user_location, tweet_location, included_countries):
        users = pd.read_parquet(user_location)
        users = users[users.country.isin(included_countries)]
        tweets = pd.read_parquet(tweet_location)
        return users.merge(tweets, on='user_id')

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
    # Addition to help the llm
    train['label'] = train['anhedonia'].replace(boolean_dictionary)
    #
    test = pd.read_parquet(test_location)

    # estimator = Pipeline([
    #     ('features', encode_features()),
    #     ('learner', MyKerasClassifier(build_fn=create_model, epochs=10, batch_size=1, verbose=0))])

    embedding_func = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # TODO: Is this a good idea? Or should we try to reuse them manually as much as possible and recompute in some cases?
    embedding_func = CacheBackedEmbeddings.from_bytes_store(embedding_func, InMemoryByteStore())
    # Required for optimization
    train['_metadata_ids'] = list(range(train.shape[0])) # list(map(str, range(train.shape[0])))
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(),
                                    metadatas=train[['label', '_metadata_ids']].to_dict('records'),
                                    embedding=embedding_func,
                                    ids=list(map(str, range(train.shape[0]))))
    # remember to reset this if we want to update the index on more than the first test set prediction call
    prediction_counter = [0]
    retrieval_index = numpy.zeros((test.shape[0], 4), dtype=int)
    rag_chain = get_langchain_rag_binary_classification_with_retrieval_tracking(
        list(boolean_dictionary.values()), vectorstore.as_retriever(), retrieval_index, prediction_counter)

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
    original_pipeline_preprocessing_end = time.time()
    predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, test['tweet'].tolist()), test))
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')

    original_pipeline_end = time.time()
    scores["original_pipeline"] = (original_pipeline_end - original_pipeline_start) * 1000
    scores["original_pipeline_preprocess"] = (original_pipeline_preprocessing_end - original_pipeline_start) * 1000
    scores["original_pipeline_eval"] = (original_pipeline_end - original_pipeline_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting shapley calculation")

    # Trying to sample test set to improve performance
    indices = numpy.arange(train.shape[0])
    numpy.random.shuffle(indices)

    # TODO: What fractions do we want to use?
    train_fraction_to_consider = 1.
    num_values_to_typo = int(train.shape[0] * train_fraction_to_consider)
    train_indices_to_consider = indices[:num_values_to_typo]
    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    reverse_train_index = numpy.zeros(num_values_to_typo, dtype=int)
    reverse_train_index[train_indices_to_consider] = numpy.arange(0, num_values_to_typo, dtype=int)
    train_data_sample = numpy.array(vectorstore.get(ids=list(map(str, train_indices_to_consider)), include=["embeddings"])['embeddings'])
    train_label_sample = label_binarize(train['anhedonia'][train_indices_to_consider], classes=[True, False])

    indices = numpy.arange(len(encoded_test_labels))
    test_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
    test_indices_to_consider = indices[:num_values_to_typo]
    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    reverse_test_index = numpy.zeros(num_values_to_typo, dtype=int)
    reverse_test_index[test_indices_to_consider] = numpy.arange(0, num_values_to_typo, dtype=int)
    test_data_sample = numpy.array(embedding_func.embed_documents(test['tweet'][test_indices_to_consider].tolist()))
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
        modified_encoded_train_labels = 1 - train_label_sample[reverse_train_index[unfair_indices]]
        # Compute the non-featurized labels
        new_unfeaturized_labels = train[['anhedonia']].iloc[unfair_indices, :].reset_index(drop=True)
        new_unfeaturized_labels['anhedonia'] = ~new_unfeaturized_labels['anhedonia']
        # Addition to help the llm
        new_unfeaturized_labels['label'] = new_unfeaturized_labels['anhedonia'].replace(boolean_dictionary)

        # Update the labels in the vectorstore
        vectorstore_ids = list(map(str, unfair_indices))
        old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "documents", "metadatas"])
        documents = old_entries['documents']
        embeddings = old_entries['embeddings']
        old_metadata = old_entries['metadatas']
        new_metadata = new_unfeaturized_labels[['label']].to_dict('records')
        vectorstore._collection.update(vectorstore_ids, embeddings, new_metadata, documents)
        fix_eval_fix_end = time.time()

        pandas_retrieval_index_df = pandas.DataFrame(retrieval_index, columns = ['train_retrieved_1', 'train_retrieved_2',
                                                                                 'train_retrieved_3', 'train_retrieved_4'])
        pandas_retrieval_index_df['prediction_id'] = list(range(test.shape[0]))
        changed_df = rows_to_fix[['train_id']]
        # TODO: Is this faster than using langchain caching for input/output pairs?
        # all_predictions_to_rerun = pandas.concat([
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_1')['prediction_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_2')['prediction_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_3')['prediction_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_4')['prediction_id'],
        # ]).drop_duplicates().reset_index(drop=True)
        all_predictions_to_rerun = duckdb.query("""
            SELECT DISTINCT prediction_id
            FROM changed_df c JOIN pandas_retrieval_index_df p 
            ON c.train_id = train_retrieved_1 
            OR c.train_id = train_retrieved_2 
            OR c.train_id = train_retrieved_3 
            OR c.train_id = train_retrieved_4 
        """).fetchnumpy()['prediction_id']
        modified_predicted_test_labels = predicted_test_labels.copy()
        llm_input = test['tweet'][all_predictions_to_rerun].tolist()
        modified_predictions_diff = numpy.array(wait_llm_call(partial(rag_chain.batch, llm_input), llm_input))
        modified_predicted_test_labels[all_predictions_to_rerun] = modified_predictions_diff
        modified_accuracy = accuracy_score(modified_predicted_test_labels, encoded_test_labels)

        # Set the labels back in the vectorstore
        # TODO: Do we want to have that as part of our measured runtime?
        vectorstore._collection.update(vectorstore_ids, embeddings, old_metadata, documents)
        print(f'Updated test accuracy is: {modified_accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    print(f"Accuracy overall {accuracy} but likely mislabeled records were detected. By flipping the labels of the top "
          f"{cleaning_batch_size} most likely mislabed records, the accuracy was changed to {modified_accuracy}! "
          f"Consider changing the weak labeling to account for these records.")
    all_interesting_label_switches_indices_train = unfair_indices

    # Explanation df train
    # Unfeaturized
    unfeaturized_original_train_diff = train.iloc[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Featurized
    featurized_original_diff = train_data_sample[reverse_train_index[all_interesting_label_switches_indices_train]]
    regex_1_diff = first_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_2_diff = second_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_3_diff = third_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Labels unencoded
    original_labels_diff = unfeaturized_original_train_diff['anhedonia']
    modified_labels_diff = original_labels_diff.copy()
    modified_labels_diff = ~modified_labels_diff

    # labels encoded
    original_encoded_labels_diff = train_label_sample[reverse_train_index[all_interesting_label_switches_indices_train]]
    modified_encoded_labels_diff = modified_encoded_train_labels

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
    featurized_original_diff_test = test_data_sample[reverse_test_index[all_interesting_label_switches_indices_test]]

    # Labels unencoded
    original_true_label_diff_test = unfeaturized_original_test_diff['anhedonia']

    # labels encoded
    original_encoded_labels_diff_test = predicted_test_labels[all_interesting_label_switches_indices_test]
    modified_encoded_labels_diff_test = modified_predicted_test_labels[all_interesting_label_switches_indices_test]
    true_encoded_labels_diff_test = encoded_test_labels[all_interesting_label_switches_indices_test]

    test_explanation_df = unfeaturized_original_test_diff[['user_id', 'lang', 'country', 'tweet']]
    test_explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff_test))
    test_explanation_df['true_label'] = original_true_label_diff_test
    test_explanation_df['encoded_original_predicted_label'] = pd.Series(list(original_encoded_labels_diff_test))
    test_explanation_df['encoded_modified_predicted_label'] = pd.Series(list(modified_encoded_labels_diff_test))
    test_explanation_df['encoded_true_label'] = pd.Series(list(true_encoded_labels_diff_test))

    print(test_explanation_df)
    broken_predictions = test_explanation_df[
        (test_explanation_df['encoded_original_predicted_label'] == test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_original_predicted_label'] != test_explanation_df['encoded_modified_predicted_label'])]
    print(broken_predictions)
    successfully_fixed = test_explanation_df[
        (test_explanation_df['encoded_original_predicted_label'] != test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_modified_predicted_label'] == test_explanation_df['encoded_true_label'])]
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

    train['anhedonia'].iloc[changed_indices] = updated_labels
    # Addition to help the llm
    train['label'].iloc[changed_indices] = train['anhedonia'].iloc[changed_indices].replace(boolean_dictionary)

    # Update the labels in the vectorstore
    vectorstore_ids = list(map(str, changed_indices))
    old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "documents", "metadatas"])
    documents = old_entries['documents']
    embeddings = old_entries['embeddings']
    old_metadata = old_entries['metadatas']
    new_metadata = train[['label', '_metadata_ids']].iloc[changed_indices].to_dict('records')
    vectorstore._collection.update(vectorstore_ids, embeddings, new_metadata, documents)
    original_pipeline_diff_preprocessing_end = time.time()
    print(f"rexecution keras started")
    # We can reuse pandas_retrieval_index_df
    changed_indices_df = pandas.DataFrame({'train_id': changed_indices})
    # TODO: Is this faster than using langchain caching for input/output pairs?
    # all_predictions_to_rerun = pandas.concat([
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_1')['prediction_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_2')['prediction_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_3')['prediction_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_4')['prediction_id'],
    # ]).drop_duplicates().reset_index(drop=True)
    all_predictions_to_rerun = duckdb.query("""
        SELECT DISTINCT prediction_id
        FROM changed_indices_df c JOIN pandas_retrieval_index_df p 
        ON c.train_id = train_retrieved_1 
        OR c.train_id = train_retrieved_2 
        OR c.train_id = train_retrieved_3 
        OR c.train_id = train_retrieved_4 
    """).fetchnumpy()['prediction_id']
    llm_input = test['tweet'][all_predictions_to_rerun].tolist()
    modified_predictions_diff = numpy.array(wait_llm_call(partial(rag_chain.batch, llm_input), llm_input)
                                            ).reshape(-1, 1)
    predicted_test_labels[all_predictions_to_rerun] = modified_predictions_diff
    modified_accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')
    original_pipeline_diff_end = time.time()
    scores["original_pipeline_diff"] = (original_pipeline_diff_end - original_pipeline_diff_start) * 1000
    scores["original_pipeline_diff_preprocess"] = (original_pipeline_diff_preprocessing_end - original_pipeline_diff_start) * 1000
    scores["original_pipeline_diff_eval"] = (original_pipeline_diff_end - original_pipeline_diff_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting shapley calculation")

    # Trying to sample test set to improve performance
    indices = numpy.arange(train.shape[0])
    numpy.random.shuffle(indices)

    # TODO: What fractions do we want to use?
    train_fraction_to_consider = 1.
    num_values_to_typo = int(train.shape[0] * train_fraction_to_consider)
    train_indices_to_consider = indices[:num_values_to_typo]
    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    reverse_train_index = numpy.zeros(num_values_to_typo, dtype=int)
    reverse_train_index[train_indices_to_consider] = numpy.arange(0, num_values_to_typo, dtype=int)
    train_data_sample = numpy.array(vectorstore.get(ids=list(map(str, train_indices_to_consider)), include=["embeddings"])['embeddings'])
    train_label_sample = label_binarize(train['anhedonia'][train_indices_to_consider], classes=[True, False])

    indices = numpy.arange(len(encoded_test_labels))
    test_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
    test_indices_to_consider = indices[:num_values_to_typo]
    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    reverse_test_index = numpy.zeros(num_values_to_typo, dtype=int)
    reverse_test_index[test_indices_to_consider] = numpy.arange(0, num_values_to_typo, dtype=int)
    test_data_sample = numpy.array(embedding_func.embed_documents(test['tweet'][test_indices_to_consider].tolist()))
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
        modified_encoded_train_labels = 1 - train_label_sample[reverse_train_index[unfair_indices]]
        # Compute the non-featurized labels
        new_unfeaturized_labels = train[['anhedonia']].iloc[unfair_indices, :].reset_index(drop=True)
        new_unfeaturized_labels['anhedonia'] = ~new_unfeaturized_labels['anhedonia']
        # Addition to help the llm
        new_unfeaturized_labels['label'] = new_unfeaturized_labels['anhedonia'].replace(boolean_dictionary)

        # Update the labels in the vectorstore
        vectorstore_ids = list(map(str, unfair_indices))
        old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "documents", "metadatas"])
        documents = old_entries['documents']
        embeddings = old_entries['embeddings']
        old_metadata = old_entries['metadatas']
        new_metadata = new_unfeaturized_labels[['label']].to_dict('records')
        vectorstore._collection.update(vectorstore_ids, embeddings, new_metadata, documents)
        fix_eval_fix_end = time.time()

        # TODO: Use an index not to rerun all predictions.
        pandas_retrieval_index_df = pandas.DataFrame(retrieval_index, columns = ['train_retrieved_1', 'train_retrieved_2',
                                                                                 'train_retrieved_3', 'train_retrieved_4'])
        pandas_retrieval_index_df['prediction_id'] = list(range(test.shape[0]))
        changed_df = rows_to_fix[['train_id']]
        # TODO: Is this faster than using langchain caching for input/output pairs?
        # all_predictions_to_rerun = pandas.concat([
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_1')['prediction_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_2')['prediction_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_3')['prediction_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_4')['prediction_id'],
        # ]).drop_duplicates().reset_index(drop=True)
        all_predictions_to_rerun = duckdb.query("""
            SELECT DISTINCT prediction_id
            FROM changed_df c JOIN pandas_retrieval_index_df p 
            ON c.train_id = train_retrieved_1 
            OR c.train_id = train_retrieved_2 
            OR c.train_id = train_retrieved_3 
            OR c.train_id = train_retrieved_4 
        """).fetchnumpy()['prediction_id']
        # TODO: join them to find records we need to rerun predictions for
        #  Then only rerun predictions for them
        modified_predicted_test_labels = predicted_test_labels.copy()
        llm_input = test['tweet'][all_predictions_to_rerun].tolist()
        modified_predictions_diff = numpy.array(wait_llm_call(partial(rag_chain.batch, llm_input), llm_input))
        modified_predicted_test_labels[all_predictions_to_rerun] = modified_predictions_diff
        modified_accuracy = accuracy_score(modified_predicted_test_labels, encoded_test_labels)

        # Set the labels back in the vectorstore
        # TODO: Do we want to have that as part of our measured runtime?
        vectorstore._collection.update(vectorstore_ids, embeddings, old_metadata, documents)
        print(f'Updated test accuracy is: {modified_accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_2"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_2_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_2_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    print(f"Accuracy overall {accuracy} but likely mislabeled records were detected. By flipping the labels of the top "
          f"{cleaning_batch_size} most likely mislabed records, the accuracy was changed to {modified_accuracy}! "
          f"Consider changing the weak labeling to account for these records.")
    all_interesting_label_switches_indices_train = unfair_indices

    # Explanation df train
    # Unfeaturized
    unfeaturized_original_train_diff = train.iloc[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Featurized
    featurized_original_diff = train_data_sample[reverse_train_index[all_interesting_label_switches_indices_train]]
    regex_1_diff = first_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_2_diff = second_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)
    regex_3_diff = third_regex[all_interesting_label_switches_indices_train].reset_index(drop=True)

    # Labels unencoded
    original_labels_diff = unfeaturized_original_train_diff['anhedonia']
    modified_labels_diff = original_labels_diff.copy()
    modified_labels_diff = ~modified_labels_diff

    # labels encoded
    original_encoded_labels_diff = train_label_sample[reverse_train_index[all_interesting_label_switches_indices_train]]
    modified_encoded_labels_diff = modified_encoded_train_labels

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
    featurized_original_diff_test = test_data_sample[reverse_test_index[all_interesting_label_switches_indices_test]]

    # Labels unencoded
    original_true_label_diff_test = unfeaturized_original_test_diff['anhedonia']

    # labels encoded
    original_encoded_labels_diff_test = predicted_test_labels[all_interesting_label_switches_indices_test]
    modified_encoded_labels_diff_test = modified_predicted_test_labels[all_interesting_label_switches_indices_test]
    true_encoded_labels_diff_test = encoded_test_labels[all_interesting_label_switches_indices_test]

    test_explanation_df = unfeaturized_original_test_diff[['user_id', 'lang', 'country', 'tweet']]
    test_explanation_df['featurized_tweet'] = pd.Series(list(featurized_original_diff_test))
    test_explanation_df['true_label'] = original_true_label_diff_test
    test_explanation_df['encoded_original_predicted_label'] = pd.Series(list(original_encoded_labels_diff_test))
    test_explanation_df['encoded_modified_predicted_label'] = pd.Series(list(modified_encoded_labels_diff_test))
    test_explanation_df['encoded_true_label'] = pd.Series(list(true_encoded_labels_diff_test))

    print(test_explanation_df)
    broken_predictions = test_explanation_df[
        (test_explanation_df['encoded_original_predicted_label'] == test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_original_predicted_label'] != test_explanation_df['encoded_modified_predicted_label'])]
    print(broken_predictions)
    successfully_fixed = test_explanation_df[
        (test_explanation_df['encoded_original_predicted_label'] != test_explanation_df['encoded_true_label']) &
        (test_explanation_df['encoded_modified_predicted_label'] == test_explanation_df['encoded_true_label'])]
    print(successfully_fixed)

    explanation_end = time.time()
    scores["explanation_2"] = (explanation_end - explanation_start) * 1000
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
