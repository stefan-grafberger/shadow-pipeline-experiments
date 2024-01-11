import time
from functools import partial

import numpy
import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from ide.experiments.analysis_utils import get_translate_transformer, wait_llm_call, get_slice_finder_mask_and_slice
from ide.experiments.pipeline_utils import get_langchain_rag_binary_classification, initialize_environment
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func,
                                    ids=list(map(str, range(train.shape[0]))))
    rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

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
        predicted_translated_labels_diff = numpy.array(wait_llm_call(partial(
            rag_chain.batch, translated_diff['tweet'].tolist()), translated_diff))
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
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
