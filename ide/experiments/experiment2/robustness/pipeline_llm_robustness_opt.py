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

from ide.experiments.analysis_utils import wait_llm_call, get_typo_adder, get_typo_fixer
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
    print(f"starting corrupting test")
    typo_adder = get_typo_adder()
    corrupted_tweet = typo_adder.fit_transform(test[['tweet']])
    corrupt_diff_mask = corrupted_tweet['tweet'] != test['tweet']
    changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
    if len(changed_indices_corrupt) != 0:
        corrupted_diff = corrupted_tweet.iloc[changed_indices_corrupt].reset_index(drop=True)
    else:
        print("Info: no change in current test set")
    detect_eval_detect_end = time.time()
    print(f"llm eval on corrupted test")
    if len(changed_indices_corrupt) != 0:
        predicted_corrupt_labels = predicted_test_labels.copy()
        predicted_corrupt_labels_diff_rerun = numpy.array(wait_llm_call(partial(
            rag_chain.batch, corrupted_diff['tweet'].tolist()), corrupted_diff))
        predicted_corrupt_labels[changed_indices_corrupt] = predicted_corrupt_labels_diff_rerun
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
    else:
        print("Info: no change in current test set")
    fix_eval_fix_end = time.time()
    print(f"llm eval on fixed test")
    if len(changed_indices_fix_corrupt) != 0:
        predicted_fix_labels = predicted_corrupt_labels.copy()
        predicted_fix_labels_diff_rerun = numpy.array(wait_llm_call(partial(
            rag_chain.batch, fixed_typos_diff['tweet'].tolist()), fixed_typos_diff))
        predicted_fix_labels[changed_indices_fix_corrupt] = predicted_fix_labels_diff_rerun
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
          f"spellchecking can increase the accuracy to {accuracy_fix_corrupt or accuracy} on corrupted data."
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
    # for train: vectorstore.get(ids=list(map(str, all_interesting_label_switches_indices)), include=["embeddings"]).embeddings
    test["featurized_tweet"] = False
    test['featurized_tweet'][all_interesting_label_switches_indices] = pd.Series(embedding_func.embed_documents(
        test["tweet"][all_interesting_label_switches_indices].tolist()))
    featurized_original_diff = test['featurized_tweet'][all_interesting_label_switches_indices]
    test['featurized_corrupted_tweet'] = test["featurized_tweet"]
    test['featurized_corrupted_tweet'][corrupt_diff_mask] = pd.Series(embedding_func.embed_documents(corrupted_diff['tweet'].tolist()))
    featurized_corrupt_diff = test['featurized_corrupted_tweet'][all_interesting_label_switches_indices]
    test['encoded_fix_corrupted_data'] = test['featurized_corrupted_tweet']
    test['encoded_fix_corrupted_data'][changed_indices_fix_corrupt] =  pd.Series(embedding_func.embed_documents(fixed_typos_diff['tweet'].tolist()))
    featurized_fix_corrupt_diff = test['encoded_fix_corrupted_data'][all_interesting_label_switches_indices]
    # Labels
    predicted_original_labels_diff = predicted_test_labels[all_interesting_label_switches_indices]
    predicted_corrupt_labels_diff = predicted_corrupt_labels[all_interesting_label_switches_indices]
    predicted_fix_corrupt_labels_diff = predicted_fix_labels[all_interesting_label_switches_indices]
    true_labels_diff = encoded_test_labels[all_interesting_label_switches_indices]

    explanation_df = unfeaturized_original_test_diff
    explanation_df['corrupted_tweet'] = unfeaturized_corrupt_test_diff['tweet']
    explanation_df['fix_corrupted_tweet'] = unfeaturized_fix_corrupt_test_diff['tweet']
    explanation_df['featurized_tweet'] = featurized_original_diff
    explanation_df['featurized_corrupted_tweet'] = featurized_corrupt_diff
    explanation_df['featurized_fix_corrupted_tweet'] = featurized_fix_corrupt_diff
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
    pandas_retrieval_index_df = pandas.DataFrame(retrieval_index, columns = ['train_retrieved_1', 'train_retrieved_2',
                                                                             'train_retrieved_3', 'train_retrieved_4'])
    pandas_retrieval_index_df['prediction_id'] = list(range(test.shape[0]))
    changed_indices_df = pandas.DataFrame({'train_id': changed_indices})
    # TODO: Is this faster than using langchain caching for input/output pairs?
    # all_predictions_to_rerun = pandas.concat([
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_1')['prediction_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_2')['prediction_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_3')['prediction_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_4')['prediction_id'],
    # ]).drop_duplicates().reset_index(drop=True)
    all_predictions_to_rerun_df = duckdb.query("""
        SELECT DISTINCT prediction_id
        FROM changed_indices_df c JOIN pandas_retrieval_index_df p 
        ON c.train_id = train_retrieved_1 
        OR c.train_id = train_retrieved_2 
        OR c.train_id = train_retrieved_3 
        OR c.train_id = train_retrieved_4 
    """).df()
    all_predictions_to_rerun = all_predictions_to_rerun_df['prediction_id']
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
    print(f"starting corrupting test")
    detect_eval_detect_end = time.time()
    print(f"llm eval on corrupted test")
    if len(changed_indices_corrupt) != 0:
        predicted_corrupt_labels = predicted_test_labels.copy()
        changed_indices_corrupt_df = pandas.DataFrame({'prediction_id': changed_indices_corrupt})
        indices_affected_by_pipeline_change = duckdb.query("""
            SELECT c.prediction_id
            FROM changed_indices_corrupt_df c JOIN all_predictions_to_rerun_df a 
            ON c.prediction_id = a.prediction_id
        """).fetchnumpy()['prediction_id']
        predicted_corrupt_labels[changed_indices_corrupt] = predicted_corrupt_labels_diff_rerun
        llm_input = corrupted_tweet['tweet'].iloc[indices_affected_by_pipeline_change].tolist()
        predicted_corrupt_labels_diff_rerun_diff = numpy.array(wait_llm_call(partial(rag_chain.batch, llm_input),
                                                                             llm_input)).reshape(-1, 1)
        predicted_corrupt_labels[indices_affected_by_pipeline_change] = predicted_corrupt_labels_diff_rerun_diff
        accuracy_corrupt = accuracy_score(predicted_corrupt_labels, encoded_test_labels)
        print(f'Test accuracy is: {accuracy_corrupt}')
    else:
        print("Info: no change in current test set")
    detect_eval_end = time.time()
    scores["detect_eval_2"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_2_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"fix typos in corrupted test")
    fix_eval_fix_end = time.time()
    print(f"llm eval on fixed test")
    if len(changed_indices_fix_corrupt) != 0:
        predicted_fix_labels = predicted_corrupt_labels.copy()
        changed_indices_fix_corrupt_df = pandas.DataFrame({'prediction_id': changed_indices_fix_corrupt})
        indices_affected_by_pipeline_change = duckdb.query("""
                SELECT c.prediction_id
                FROM changed_indices_fix_corrupt_df c JOIN all_predictions_to_rerun_df a 
                ON c.prediction_id = a.prediction_id
            """).fetchnumpy()['prediction_id']
        predicted_fix_labels[changed_indices_fix_corrupt] = predicted_fix_labels_diff_rerun
        llm_input = corrupted_tweet['tweet'].iloc[indices_affected_by_pipeline_change].tolist()
        predicted_fix_labels_diff_rerun_diff = numpy.array(wait_llm_call(partial(
            rag_chain.batch, llm_input),llm_input)).reshape(-1, 1)
        predicted_fix_labels[indices_affected_by_pipeline_change] = predicted_fix_labels_diff_rerun_diff

        accuracy_fix_corrupt = accuracy_score(predicted_fix_labels, encoded_test_labels)
        print(f'Test accuracy is: {accuracy_fix_corrupt}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_2"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_2_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_2_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    explanation_start = time.time()
    print(f"generating explanations")
    # Explanation start
    print(f"The original accuracy {accuracy} drops to {accuracy_corrupt or accuracy} when adding typos. However, "
          f"spellchecking can increase the accuracy to {accuracy_fix_corrupt or accuracy} on corrupted data."
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
    # for train: vectorstore.get(ids=list(map(str, all_interesting_label_switches_indices)), include=["embeddings"]).embeddings
    test["featurized_tweet"] = False
    test['featurized_tweet'][all_interesting_label_switches_indices] = pd.Series(embedding_func.embed_documents(
        test["tweet"][all_interesting_label_switches_indices].tolist()))
    featurized_original_diff = test['featurized_tweet'][all_interesting_label_switches_indices]
    test['featurized_corrupted_tweet'] = test["featurized_tweet"]
    test['featurized_corrupted_tweet'][corrupt_diff_mask] = pd.Series(embedding_func.embed_documents(corrupted_diff['tweet'].tolist()))
    featurized_corrupt_diff = test['featurized_corrupted_tweet'][all_interesting_label_switches_indices]
    test['encoded_fix_corrupted_data'] = test['featurized_corrupted_tweet']
    test['encoded_fix_corrupted_data'][changed_indices_fix_corrupt] =  pd.Series(embedding_func.embed_documents(fixed_typos_diff['tweet'].tolist()))
    featurized_fix_corrupt_diff = test['encoded_fix_corrupted_data'][all_interesting_label_switches_indices]
    # Labels
    predicted_original_labels_diff = predicted_test_labels[all_interesting_label_switches_indices]
    predicted_corrupt_labels_diff = predicted_corrupt_labels[all_interesting_label_switches_indices]
    predicted_fix_corrupt_labels_diff = predicted_fix_labels[all_interesting_label_switches_indices]
    true_labels_diff = encoded_test_labels[all_interesting_label_switches_indices]

    explanation_df = unfeaturized_original_test_diff
    explanation_df['corrupted_tweet'] = unfeaturized_corrupt_test_diff['tweet']
    explanation_df['fix_corrupted_tweet'] = unfeaturized_fix_corrupt_test_diff['tweet']
    explanation_df['featurized_tweet'] = featurized_original_diff
    explanation_df['featurized_corrupted_tweet'] = featurized_corrupt_diff
    explanation_df['featurized_fix_corrupted_tweet'] = featurized_fix_corrupt_diff
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
    scores["explanation_2"] = (explanation_end - explanation_start) * 1000
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
