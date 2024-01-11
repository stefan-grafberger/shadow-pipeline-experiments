import time

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from sqlalchemy import text


class CachedTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func_transformer, database_path: str = ".function_transformer_cache.db", cache_table='api_cache'):
        self.func_transformer = func_transformer
        self.cache_db = database_path
        self.cache_table = cache_table
        self.engine = create_engine(f'sqlite:///{self.cache_db}')

        # Create cache table if not exists
        with self.engine.connect() as conn:
            conn.execute(text(f'''
                CREATE TABLE IF NOT EXISTS {self.cache_table} (
                    input TEXT PRIMARY KEY,
                    output TEXT
                )
            '''))

    def fit(self, X, y=None):
        self.func_transformer.fit(X, y)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame) and len(list(X.columns)) == 1
        # this needs a wait because it is cache-only in our experiments
        # LLM throughput estimate:
        # groq mixtral: https://groq.com/?model_id=mixtral-8x7b-32768
        # 500T/s
        # For mistral: "A word is generally 2-3 tokens" https://replicate.com/mistralai/mistral-7b-instruct-v0.1/api
        # Let us calculate with 3 to be conservative
        # how many words do we have here per tweet?
        # avg_word_count = sum(map(lambda x: len(x.split()), test['tweet'].tolist())) / len(test['tweet'].tolist())
        # print(avg_word_count)
        # # 12.95
        # calculation: (batch_size * 13 * 3) / 500
        # translation estimate
        # However, specialised translation models are faster than more general LLMs, use a higher throughput formula
        # CTranslate 2 is an example of a fast translation library for efficient inference
        # They include some benchmark numbers here: https://github.com/OpenNMT/CTranslate2?tab=readme-ov-file#benchmarks
        # The reported GPU token/s numbers are between 6634 and 10990
        # Let us maybe be a bit more conservative and use 6.000 tokens/s.
        # https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090
        # This is the tokenizer code used in their benchmark:
        # pyonmttok.Tokenizer("none", sp_model_path="/model/sentencepiece.model")
        # For SentencePience, it seems to heavily depend on the language how many tokens per word are used
        # For common languages, 2-3, for rare ones, 4+. However, in most of our tweets, the language is English.
        # https://ddimri.medium.com/sentencepiece-the-nlp-architects-tool-for-building-bridges-between-languages-7a0b8ae53130
        # There are only a few rare language snippets in there.
        # But let's say 4 tokens/s here
        # But, e.g., DeepL says they can translate a million words in under a second, so we could also use higher
        # numbers, even with self-deployed solutions parallelization can be used
        # https://slator.com/linguees-founder-launches-deepl-attempt-challenge-google-translate/
        realistic_wait_time_calculation = (X.shape[0] * 13 * 4) / 6000
        translation_start = time.time()


        # Transform X using cache
        # TODO: Do we need to implement this more efficiently by doing batch updates to disk?
        with self.engine.connect() as conn:
            cached_data = pd.read_sql(text(f'SELECT * FROM {self.cache_table}'), conn)
        cached_dict = dict(zip(cached_data['input'], cached_data['output']))
        transformed_data = []
        for x in X.iloc[:, 0].tolist():
            input_str = str(x)
            if input_str in cached_dict:
                transformed_data.append(cached_dict[input_str])
            else:
                output = self.func_transformer.transform([x])[0]
                transformed_data.append(output)
                # Update cache
                self._update_cache(input_str, output)
        X.iloc[:, 0] = transformed_data

        translation_end = time.time()
        additional_sleep = max(realistic_wait_time_calculation - (translation_end - translation_start) / 1000,  0)
        print(f"Sleeping an additional {additional_sleep}s to simulate real API call when cache was hit "
              f"({realistic_wait_time_calculation} - {(translation_end - translation_start) / 1000})!")
        time.sleep(additional_sleep)
        return X

    def _update_cache(self, input_str, output):
        with self.engine.connect() as conn:
            conn.execute(text(f'''
                INSERT OR REPLACE INTO {self.cache_table} (input, output)
                VALUES (:input, :output)
            '''), {'input': input_str, 'output': output})
