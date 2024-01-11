import time
from functools import partial

import numpy
import pandas as pd
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func)
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
    # train = weak_labeling(train)
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func)
    rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
    detect_eval_detect_end = time.time()
    predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, test['tweet'].tolist()), test))
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    accuracy_slice = accuracy_score(predicted_test_labels[test_mask], encoded_test_labels[test_mask])
    # TODO: If this were not a hardcoded example, we would check here if the accuracy difference justifies further analyses
    detect_eval_end = time.time()
    scores["detect_eval_1"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_1_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"starting translating test")
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
    # train = weak_labeling(train)
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func)
    rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])

    translate_transformer = get_translate_transformer()
    translated_tweets = translate_transformer.fit_transform(test[['tweet']])
    fix_eval_fix_end = time.time()

    print(f"llm eval on translated test")
    predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, translated_tweets['tweet'].tolist()), test))
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels,
                                                           assert_bengali=False)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000


    scores["explanation_1"] = None

    # Pipeline change starting here
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
    # train = weak_labeling(train)
    first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
    second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func)
    rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
    original_pipeline_preprocessing_end = time.time()
    predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, test['tweet'].tolist()), test))
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    print(f'Test accuracy is: {accuracy}')

    original_pipeline_end = time.time()
    scores["original_pipeline_diff"] = (original_pipeline_end - original_pipeline_start) * 1000
    scores["original_pipeline_diff_preprocess"] = (original_pipeline_preprocessing_end - original_pipeline_start) * 1000
    scores["original_pipeline_diff_eval"] = (original_pipeline_end - original_pipeline_preprocessing_end) * 1000

    detect_eval_start = time.time()
    print(f"starting slice finding")
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
    # train = weak_labeling(train)
    first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
    second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func)
    rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
    detect_eval_detect_end = time.time()
    predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, test['tweet'].tolist()), test))
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    accuracy_slice = accuracy_score(predicted_test_labels[test_mask], encoded_test_labels[test_mask])
    # TODO: If this were not a hardcoded example, we would check here if the accuracy difference justifies further analyses
    detect_eval_end = time.time()
    scores["detect_eval_2"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_2_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_2_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"starting translating test")
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
    # train = weak_labeling(train)
    first_regex = train['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
    second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True)
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
    vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                    embedding=embedding_func)
    rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])

    translate_transformer = get_translate_transformer()
    translated_tweets = translate_transformer.fit_transform(test[['tweet']])
    fix_eval_fix_end = time.time()

    print(f"llm eval on translated test")
    predicted_test_labels = numpy.array(
        wait_llm_call(partial(rag_chain.batch, translated_tweets['tweet'].tolist()), test))
    top_slice, test_mask = get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels,
                                                           assert_bengali=False)
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    fix_eval_end = time.time()
    scores["fix_eval_2"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_2_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_2_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    scores["explanation_2"] = None
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
    # TODO: Here, the slice finder can only be applied after model eval, so it is part of the model eval here.
    #  Is this okay? Or should we measure this time separately and add it to the other time?
