"""
Some useful utils for the project
"""
import os
import getpass
from inspect import cleandoc

import numpy
from keras import Input
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from ide.experiments.gensim_wrapper import W2VTransformer
from ide.utils.utils import get_project_root


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""

    def partial_fit(self, X):
        # pylint: disable=useless-super-delegation
        super().partial_fit([X])

    def fit(self, X, y=None):
        X = X.iloc[:, 0].tolist()
        return super().fit([X], y)

    def transform(self, words):
        words = words.iloc[:, 0].tolist()
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        vectors = []
        for word in words:
            if word in self.gensim_model.wv:
                vectors.append(self.gensim_model.wv[word])
            else:
                vectors.append(numpy.zeros(self.size))
        return numpy.reshape(numpy.array(vectors), (len(words), self.size))


def create_model(meta, hidden_layer_sizes):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(Dense(hidden_layer_size, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

def get_langchain_rag_binary_classification(classes, retriever):
    # Prompt template taken from skllm
    FEW_SHOT_CLF_PROMPT_TEMPLATE = cleandoc(f"""
        You will be provided with the following information:
        1. An arbitrary text sample. The sample is delimited with triple backticks.
        2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
        3. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data.
        
        Perform the following tasks:
        1. Identify to which category the provided text belongs to with the highest probability.
        2. Assign the provided text to that category.
        3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.
        
        List of categories: {classes}
        
        Training data:
        {{context}}
        
        Text sample: ```{{question}}```
        
        Your JSON response:
        """)
    prompt = ChatPromptTemplate.from_template(FEW_SHOT_CLF_PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        retrieved_formatted = "\n\n".join(
            f"```{doc.page_content}```\nassigned category: ['{doc.metadata['label']}']" for doc in
            docs)  # Here we can also use label!
        # TODO: Create a copy of these functions and then modify the metadata in the llm mislabel pipeline to
        #  also contain the document id in the metadata. We don't have access to the id here, but as metadata
        #  we can access a static variable with the index datastructure
        #  We can infer the query id based on the order that we call the langchain with, I guess these are not
        #  randomly shuffled or anything like that. But we should verify this.
        #  However, we might not always want to have this additional id assign overhead, but maybe its fine
        return retrieved_formatted

    def format_response(json_object):
        if not isinstance(json_object, dict) or not "label" in json_object:
            return random.choice([0, 1])
        assigned_label = json_object["label"]
        if assigned_label not in classes:
            return random.choice([0, 1])
        result = label_binarize([assigned_label], classes=classes)
        return result[0]

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | JsonOutputParser()
            | format_response
    )
    return rag_chain


def get_langchain_rag_binary_classification_with_retrieval_tracking(classes, retriever, retrieval_index,
                                                                    prediction_counter):
    # Prompt template taken from skllm
    FEW_SHOT_CLF_PROMPT_TEMPLATE = cleandoc(f"""
        You will be provided with the following information:
        1. An arbitrary text sample. The sample is delimited with triple backticks.
        2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
        3. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data.

        Perform the following tasks:
        1. Identify to which category the provided text belongs to with the highest probability.
        2. Assign the provided text to that category.
        3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

        List of categories: {classes}

        Training data:
        {{context}}

        Text sample: ```{{question}}```

        Your JSON response:
        """)
    prompt = ChatPromptTemplate.from_template(FEW_SHOT_CLF_PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    def format_docs(docs):
        retrieved_formatted = "\n\n".join(
            f"```{doc.page_content}```\nassigned category: ['{doc.metadata['label']}']" for doc in
            docs)  # Here we can also use label!
        # TODO: Create a copy of these functions and then modify the metadata in the llm mislabel pipeline to
        #  also contain the document id in the metadata. We don't have access to the id here, but as metadata
        #  we can access a static variable with the index datastructure
        #  We can infer the query id based on the order that we call the langchain with, I guess these are not
        #  randomly shuffled or anything like that. But we should verify this.
        #  However, we might not always want to have this additional id assign overhead, but maybe its fine
        if prediction_counter[0] < retrieval_index.shape[0]:
            retrieval_index[prediction_counter[0], :] = [doc.metadata['_metadata_ids'] for doc in docs]
            prediction_counter[0] += 1
        return retrieved_formatted

    def format_response(json_object):
        if not isinstance(json_object, dict) or not "label" in json_object:
            return random.choice([0, 1])
        assigned_label = json_object["label"]
        if assigned_label not in classes:
            return random.choice([0, 1])
        result = label_binarize([assigned_label], classes=classes)
        return result[0]

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | JsonOutputParser()
            | format_response
    )
    return rag_chain

def initialize_environment():
    seed = 42
    numpy.random.seed(seed)
    random.seed(seed)
    set_llm_cache(SQLiteCache(database_path=f"{str(get_project_root())}/offline/.langchain.db"))
    # os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    os.environ["OPENAI_API_KEY"] = "not_required_because_of_sqlite_caching"
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
