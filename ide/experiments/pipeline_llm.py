import time
from functools import partial

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from ide.experiments.analysis_utils import wait_llm_call
from ide.experiments.pipeline_utils import get_langchain_rag_binary_classification, initialize_environment
from ide.utils.utils import get_project_root

initialize_environment()

initial_start = time.time()

boolean_dictionary = {True: 'anhedonia', False: 'regular'}

def load_train_data(user_location, tweet_location, included_countries):
    users = pd.read_parquet(user_location)
    users = users[users.country.isin(included_countries)]
    tweets = pd.read_parquet(tweet_location)
    return users.merge(tweets, on='user_id')


def weak_labeling(data):
    data['anhedonia'] = ((data['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
                          | data['tweet'].str.contains('(lose|losing|lost).{0,15} (interest|pleasure|motivation)', regex=True))
                         & ~(data['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
                         )
    # Addition to help the llm
    data['label'] = data['anhedonia'].replace(boolean_dictionary)
    return data

user_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/users.pqt'
tweet_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/tweets.pqt'
nl_be = ['NL', 'BE']
test_location = f'{str(get_project_root())}/ide/experiments/datasets/anhedonia/expert_labeled.pqt'

train = load_train_data(user_location, tweet_location, included_countries=nl_be)
train = weak_labeling(train)
test = pd.read_parquet(test_location)

vectorstore = Chroma.from_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
                                embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
                                #ids=list(map(str, range(train.shape[0])))
                                )
# This is how we can update documents in the vectorstore by giving them ids and using those for replacements
# train['label'] = (~train['anhedonia']).replace(boolean_dictionary)
# vectorstore.add_texts(texts=train['tweet'].tolist(), metadatas=train[['label']].to_dict('records'),
#                       ids=list(map(str, range(train.shape[0]))))

rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

y_predicted = wait_llm_call(partial(rag_chain.batch, test['tweet'].tolist()), test)
y_test_binarized = label_binarize(test['anhedonia'], classes=[True, False])
accuracy = accuracy_score(y_predicted, y_test_binarized)
print(f'Test accuracy is: {accuracy}')

initial_end = time.time()
print(f"initial: {(initial_end - initial_start) * 1000}")
