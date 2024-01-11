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

from ide.experiments.analysis_utils import get_translate_transformer, wait_llm_call, get_slice_finder_mask_and_slice
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
    print(f"starting slice finding")
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels)
    accuracy_slice = accuracy_score(predicted_test_labels[test_mask], encoded_test_labels[test_mask])
    unfair_indices = numpy.where(test_mask)[0]
    # TODO: If this were not a hardcoded example, we would check here if the accuracy difference justifies further analyses
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
    else:
        print("Info: no change in current test set")
    fix_eval_fix_end = time.time()

    print(f"llm eval on translated test")
    if len(changed_indices_translated) != 0:
        predicted_translated_labels = predicted_test_labels.copy()

        # we need to create a second retrieval index, instead of resetting the index here we could also create a new
        #  langchain pipeline, but maybe this is quicker?
        old_prediction_counter_value = prediction_counter[0]
        prediction_counter = [0]
        old_retrieval_index = retrieval_index.copy()
        retrieval_index = numpy.zeros((translated_diff.shape[0], 4), dtype=int)

        # get predictions of translated tweets and extract retrieval index
        predicted_translated_labels_diff = numpy.array(wait_llm_call(partial(
            rag_chain.batch, translated_diff['tweet'].tolist()), translated_diff))
        pandas_translated_retrieval_index_df = pandas.DataFrame(retrieval_index,
                                                                columns=['train_retrieved_1', 'train_retrieved_2',
                                                                         'train_retrieved_3', 'train_retrieved_4'])
        pandas_translated_retrieval_index_df['test_id'] = changed_indices_translated

        # Reset retrieval index again as if nothing happened not in the original pipeline
        retrieval_index = old_retrieval_index
        prediction_counter[0] = old_prediction_counter_value

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
    test["featurized_tweet"] = False
    test['featurized_tweet'][all_interesting_label_switches_indices] = pd.Series(embedding_func.embed_documents(
        test["tweet"][all_interesting_label_switches_indices].tolist()))
    featurized_original_diff = test['featurized_tweet'][all_interesting_label_switches_indices]
    test['featurized_translated_tweet'] = test["featurized_tweet"]
    test['featurized_translated_tweet'][changed_indices_translated] = pd.Series(embedding_func.embed_documents(translated_diff['tweet'].tolist()))
    featurized_translat_diff = test['featurized_translated_tweet'][all_interesting_label_switches_indices]

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
    pandas_retrieval_index_df['test_id'] = list(range(test.shape[0]))
    changed_indices_df = pandas.DataFrame({'train_id': changed_indices})
    # TODO: Is this faster than using langchain caching for input/output pairs?
    # all_predictions_to_rerun = pandas.concat([
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_1')['test_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_2')['test_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_3')['test_id'],
    #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_4')['test_id'],
    # ]).drop_duplicates().reset_index(drop=True)
    all_predictions_to_rerun_df = duckdb.query("""
        SELECT DISTINCT test_id
        FROM changed_indices_df c JOIN pandas_retrieval_index_df p 
        ON c.train_id = train_retrieved_1 
        OR c.train_id = train_retrieved_2 
        OR c.train_id = train_retrieved_3 
        OR c.train_id = train_retrieved_4 
    """).df()
    all_predictions_to_rerun = all_predictions_to_rerun_df['test_id']
    llm_inut = test['tweet'][all_predictions_to_rerun].tolist()
    modified_predictions_diff = numpy.array(wait_llm_call(partial(rag_chain.batch, llm_inut), llm_inut)
                                            ).reshape(-1, 1)
    predicted_test_labels[all_predictions_to_rerun] = modified_predictions_diff
    modified_accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
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
    else:
        not_translated_yet['tweet'] = None  # Just so duckdb can find the column even if there are no changes
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
        translated_diff = changed_indices_translated_df
    else:
        print("Info: no change in current test set")
    fix_eval_fix_end = time.time()

    print(f"llm eval on translated test")
    if len(changed_indices_translated) != 0:
        predicted_translated_labels_rerun = predicted_test_labels.copy()
        # First replace modified test set predictions with the preds for the now (maybe cache-)translated tweets that are
        # unfair. We use the old translated predictions independent of if they are still up-to-date
        predicted_translated_labels_rerun[changed_indices_translated] = predicted_translated_labels[changed_indices_translated]
        # Then, recompute all predictions that might be outdated because of the label changes, using the retrieval index
        # TODO: Is this faster than using langchain caching for input/output pairs?
        # all_predictions_to_rerun = pandas.concat([
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_1')['test_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_2')['test_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_3')['test_id'],
        #     changed_df.merge(pandas_retrieval_index_df, left_on='train_id', right_on='train_retrieved_4')['test_id'],
        # ]).drop_duplicates().reset_index(drop=True)
        all_translation_predictions_to_rerun_df = duckdb.query("""
            -- These are the ones where the prediction needs to be updated because they retrieve rows changed by 
            -- the updated labeling
            SELECT p.test_id, a.tweet
            FROM changed_indices_df c JOIN pandas_translated_retrieval_index_df p
            ON c.train_id = train_retrieved_1 
            OR c.train_id = train_retrieved_2 
            OR c.train_id = train_retrieved_3 
            OR c.train_id = train_retrieved_4 
            JOIN already_translated a
            ON a.test_id = p.test_id
            UNION
            -- These are the records that we are running predictions on for the first time
            SELECT test_id, tweet 
            FROM not_translated_yet
            WHERE tweet != before_translation
        """).df()
        all_translation_predictions_to_rerun = all_translation_predictions_to_rerun_df['test_id']
        tweets_to_translate = all_translation_predictions_to_rerun_df['tweet']
        predicted_translated_labels_diff_rerun = numpy.array(wait_llm_call(partial(
            rag_chain.batch, tweets_to_translate.tolist()), tweets_to_translate)).reshape(-1, 1)
        predicted_translated_labels[all_translation_predictions_to_rerun] = predicted_translated_labels_diff_rerun
        predicted_translated_labels = predicted_translated_labels_rerun # Now we can discard the old results
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
    test["featurized_tweet"] = False
    test['featurized_tweet'][all_interesting_label_switches_indices] = pd.Series(embedding_func.embed_documents(
        test["tweet"][all_interesting_label_switches_indices].tolist()))
    featurized_original_diff = test['featurized_tweet'][all_interesting_label_switches_indices]
    test['featurized_translated_tweet'] = test["featurized_tweet"]
    test['featurized_translated_tweet'][changed_indices_translated] = pd.Series(embedding_func.embed_documents(translated_diff['tweet'].tolist()))
    featurized_translat_diff = test['featurized_translated_tweet'][all_interesting_label_switches_indices]

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
    scores["explanation_2"] = (explanation_end - explanation_start) * 1000
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
