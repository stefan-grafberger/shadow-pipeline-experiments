import time
from functools import partial

import numpy
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from ide.experiments.analysis_utils import wait_llm_call, get_shapley_value_result
from ide.experiments.pipeline_utils import initialize_environment, get_langchain_rag_binary_classification
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
    # remember to reset this if we want to update the index on more than the first test set prediction call
    rag_chain = get_langchain_rag_binary_classification(
        list(boolean_dictionary.values()), vectorstore.as_retriever())

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
                                    embedding=embedding_func,
                                    ids=list(map(str, range(train.shape[0]))))
    rag_chain = get_langchain_rag_binary_classification(
        list(boolean_dictionary.values()), vectorstore.as_retriever())

    encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])

    # Trying to sample test set to improve performance
    indices = numpy.arange(train.shape[0])
    numpy.random.shuffle(indices)

    # TODO: What fractions do we want to use?
    train_fraction_to_consider = 1.
    num_values_to_typo = int(train.shape[0] * train_fraction_to_consider)
    train_indices_to_consider = indices[:num_values_to_typo]
    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    train_data_sample = numpy.array(vectorstore.get(ids=list(map(str, train_indices_to_consider)), include=["embeddings"])['embeddings'])
    train_label_sample = label_binarize(train['anhedonia'][train_indices_to_consider], classes=[True, False])

    indices = numpy.arange(len(encoded_test_labels))
    test_fraction_to_consider = 1.
    num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
    test_indices_to_consider = indices[:num_values_to_typo]
    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    test_data_sample = numpy.array(embedding_func.embed_documents(test['tweet'][test_indices_to_consider].tolist()))
    test_label_sample = encoded_test_labels[test_indices_to_consider]

    rows_to_fix, cleaning_batch_size = get_shapley_value_result(train_data_sample, train_label_sample, test_data_sample,
                                                                test_label_sample, train_indices_to_consider)
    unfair_indices = rows_to_fix['train_id']
    selected_shapley_values = rows_to_fix['shapley_value'].reset_index(drop=True)
    detect_eval_detect_end = time.time()
    predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, test['tweet'].tolist()), test))
    accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
    detect_eval_end = time.time()
    scores["detect_eval_1"] = (detect_eval_end - detect_eval_start) * 1000
    scores["detect_eval_1_detect"] = (detect_eval_detect_end - detect_eval_start) * 1000
    scores["detect_eval_1_eval"] = (detect_eval_end - detect_eval_detect_end) * 1000

    fix_eval_start = time.time()
    print(f"starting retraining and evaluation")
    if len(unfair_indices) != 0:
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
        second_regex = train['tweet'].str.contains('(lose|losing|lost).{0,15} (interest|pleasure|motivation)',
                                                   regex=True)
        third_regex = ~(
            train['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
        train['anhedonia'] = ((first_regex | second_regex) & third_regex)

        test = pd.read_parquet(test_location)

        # estimator = Pipeline([
        #     ('features', encode_features()),
        #     ('learner', MyKerasClassifier(build_fn=create_model, epochs=10, batch_size=1, verbose=0))])

        encoded_test_labels = label_binarize(test['anhedonia'], classes=[True, False])
        # This assumes the user, e.g., copy and pastes the likely mislabed rows in-between executions
        # Compute the non-featurized labels
        train[['anhedonia']].iloc[unfair_indices, :] = ~train[['anhedonia']].iloc[unfair_indices, :]
        # Addition to help the llm
        train['label'] = train['anhedonia'].replace(boolean_dictionary)

        embedding_func = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                        embedding=embedding_func)
        # remember to reset this if we want to update the index on more than the first test set prediction call
        rag_chain = get_langchain_rag_binary_classification(
            list(boolean_dictionary.values()), vectorstore.as_retriever())
        fix_eval_fix_end = time.time()
        llm_input = test['tweet'].tolist()
        predicted_test_labels = numpy.array(wait_llm_call(partial(rag_chain.batch, llm_input), llm_input))

        modified_accuracy = accuracy_score(predicted_test_labels, encoded_test_labels)
        print(f'Updated test accuracy is: {modified_accuracy}')
    else:
        print("Info: no change in current test set")
    fix_eval_end = time.time()
    scores["fix_eval_1"] = (fix_eval_end - fix_eval_start) * 1000
    scores["fix_eval_1_fix"] = (fix_eval_fix_end - fix_eval_start) * 1000
    scores["fix_eval_1_eval"] = (fix_eval_end - fix_eval_fix_end) * 1000

    scores["explanation_1"] = None
    return scores

if __name__ == "__main__":
    scores = execute()
    print(scores)
