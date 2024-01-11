import time
import warnings

import duckdb
import numpy
import pandas
import pandas as pd
from autocorrect import Speller
from deep_translator import GoogleTranslator
from numba import njit, prange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sliceline import Slicefinder
from textblob import TextBlob

from ide.experiments.cached_text_transformer import CachedTextTransformer
from ide.utils.utils import get_project_root


class MislabelCleaner:
    """
    Mislabel ErrorType
    """

    @staticmethod
    def get_shapley_values(train_data, train_labels):
        """See https://arxiv.org/abs/2204.11131"""
        if isinstance(train_labels, pandas.Series):
            train_labels = train_labels.to_numpy()
        k = 10
        assert k > (len(train_labels) * 0.2)
        train_data, test_data, train_labels, test_label = train_test_split(train_data, train_labels, test_size=k)
        shapley_values = MislabelCleaner._compute_shapley_values(train_data, train_labels, test_data, test_label, k)
        greater_zero = shapley_values >= 0.0
        return shapley_values

    # removed cache=True because of https://github.com/numba/numba/issues/4908 need a workaround soon
    @staticmethod
    @njit(fastmath=True, parallel=True, cache=True)
    def _compute_shapley_values(X_train, y_train, X_test, y_test, K=1):
        # pylint: disable=invalid-name,too-many-locals
        """Compute approximate shapley values as presented in the DataScope paper. Here, we only do it for the
        estimator input data though and not for the input data of the surrounding pipeline.
        """
        N = len(X_train)
        M = len(X_test)
        result = numpy.zeros(N, dtype=numpy.float32)

        for j in prange(M):  # pylint: disable=not-an-iterable
            score = numpy.zeros(N, dtype=numpy.float32)
            dist = numpy.zeros(N, dtype=numpy.float32)
            div_range = numpy.arange(1.0, N)
            div_min = numpy.minimum(div_range, K)
            for i in range(N):
                dist[i] = numpy.sqrt(numpy.sum(numpy.square(X_train[i] - X_test[j])))
            indices = numpy.argsort(dist)
            y_sorted = y_train[indices]
            eq_check = (y_sorted == y_test[j]) * 1.0
            diff = - 1 / K * (eq_check[1:] - eq_check[:-1])
            diff /= div_range
            diff *= div_min
            score[indices[:-1]] = diff
            score[indices[-1]] = eq_check[-1] / N
            score[indices] += numpy.sum(score[indices]) - numpy.cumsum(score[indices])
            result += score / M

        return result

def get_translate_transformer():
    global translator, translate_transformer
    translator = GoogleTranslator(source='auto', target='en')

    # translator = MyMemoryTranslator(source='auto', target='en-US')
    # Could also use HuggingFace, but then it would be even slower probably
    # https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
    def translate(df):
        # df['tweet'] = df['tweet'].map(lambda txt: translator.translate(txt))
        if isinstance(df, pd.DataFrame):
            df['tweet'] = translator.translate_batch(df['tweet'].to_list())
        else:
            df = translator.translate_batch(df)
        # TODO: Is this fast enough?
        return df

    warnings.filterwarnings('ignore')
    translate_transformer = FunctionTransformer(translate)
    translate_transformer = CachedTextTransformer(translate_transformer,
                                                  database_path=f"{str(get_project_root())}/offline"
                                                                f"/.function_transformer_cache.db")
    return translate_transformer

def wait_llm_call(partial, test_df_for_size_calculation):
    # this needs a wait because it is cache-only in our experiments
    # groq mixtral: https://groq.com/?model_id=mixtral-8x7b-32768
    # 500T/s
    # For mistral: "A word is generally 2-3 tokens" https://replicate.com/mistralai/mistral-7b-instruct-v0.1/api
    # Let us calculate with 3 to be conservative
    # how many words do we have here?
    # avg_word_count = sum(map(lambda x: len(x.split()), test['tweet'].tolist())) / len(test['tweet'].tolist())
    # print(avg_word_count)
    # 12.95
    # calculation: (batch_size * 13 * 3) / 500
    # TODO: slow/fast LLM at some point in the future?
    if isinstance(test_df_for_size_calculation, pd.DataFrame):
        test_size = test_df_for_size_calculation.shape[0]
    elif isinstance(test_df_for_size_calculation, list):
        test_size = len(test_df_for_size_calculation)
    else:
        test_size = test_df_for_size_calculation.size
    realistic_wait_time_calculation = (test_size * 13 * 3) / 500
    langchain_start = time.time()
    result = partial()
    langchain_end = time.time()
    time.sleep(max(realistic_wait_time_calculation - (langchain_end - langchain_start) / 1000, 0))
    additional_sleep = max(realistic_wait_time_calculation - (langchain_end - langchain_start) / 1000, 0)
    print(f"Sleeping an additional {additional_sleep}s to simulate real API call when cache was hit "
          f"({realistic_wait_time_calculation} - {(langchain_end - langchain_start) / 1000})!")
    # langchain batch using ChatGPT 3: 29,649606943130493s
    # estimated wait time for it: 7.8s

    return result

def get_slice_finder_mask_and_slice(test, encoded_test_labels, predicted_test_labels, assert_bengali=True):
    # TODO: In the documentation, it is only used on train and with known float loss
    sf = Slicefinder(
        alpha=0.95,
        k=1,
        max_l=2,
        min_sup=1,
        verbose=True,
    )

    sf_result = sf.fit(test[['lang', 'country']], (encoded_test_labels != predicted_test_labels).reshape(-1, ))
    top_slice = sf_result.top_slices_[0]
    # Make sure that our experiment consistently hits the case where something needs to be translated
    if assert_bengali is True:
        assert top_slice[0] == 'bengali'
    elif assert_bengali is False:
        assert top_slice[0] != 'bengali'
    if top_slice[1] is not None:
        test_mask = (test["lang"] == top_slice[0]) & (test['country'] == top_slice[1])
    else:
        test_mask = test["lang"] == top_slice[0]
    return top_slice, test_mask

def get_shapley_value_result(train_data_sample, train_label_sample, test_data_sample, test_label_sample,
                             train_indices):
    shapley_values = MislabelCleaner._compute_shapley_values(train_data_sample, numpy.squeeze(train_label_sample),
                                                             test_data_sample,
                                                             numpy.squeeze(test_label_sample))
    df_with_id_and_shapley_value = pd.DataFrame(
        {"train_id": train_indices, "shapley_value": shapley_values})

    # The numba compilation takes a while, caching for this function is important. Need to make sure when running benchmarks
    #  that we discard the first run if it is not cached
    # shapley_values = MislabelCleaner._compute_shapley_values(encoded_train_data, numpy.squeeze(encoded_train_labels), encoded_test_data,
    #                                                          numpy.squeeze(encoded_test_labels))
    # df_with_id_and_shapley_value = pd.DataFrame(
    #         {"id": list(range(train.shape[0])), "shapley_value": shapley_values})
    cleaning_batch_size = 20
    rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
    return rows_to_fix, cleaning_batch_size

def get_typo_adder():
    fraction_to_typo = 0.1

    def add_typos(df):
        indices = numpy.arange(len(df))
        numpy.random.shuffle(indices)
        num_values_to_typo = int(len(df) * fraction_to_typo)
        indices_to_typo = indices[:num_values_to_typo]
        # df.loc[indices_to_typo, 'tweet'] = df.loc[indices_to_typo, 'tweet'].apply(lambda txt: typo_augmenter.augment(txt)[0])
        data_to_corrupt = df[['tweet']].iloc[indices_to_typo]
        data_to_corrupt['row_id'] = list(range(data_to_corrupt.shape[0]))
        # corrupted_data = duckdb.query("""
        #     SELECT regexp_replace(tweet, '')
        #     FROM data_to_corrupt
        # """).df()['tweet']
        corrupted_data = duckdb.query("""
                SELECT
                CASE
                WHEN random() < 0.3 THEN (
                    SELECT
                        REPLACE(
                            REPLACE(
                                REPLACE(
                                    REPLACE(tweet, 'n', 'm'),
                                'b', 'v'),
                            't', 'r'),
                        'o', 'p'),
                )
                -- Introduce substitution errors
                WHEN random() < 0.3 THEN (
                    SELECT
                        STRING_AGG(
                            CASE
                                WHEN random() < 0.03 THEN chr(65 + (abs(ASCII(character)) + CAST(random() * 25 AS INT)) % 26)  -- Substitute with random character
                                ELSE character
                            END, ''
                        )
                    FROM
                        UNNEST(SPLIT(tweet, '')) AS t(character)
                )
                -- Introduce insertion errors
                WHEN random() < 0.3 THEN (
                    SELECT
                        STRING_AGG(
                            CASE WHEN random() < 0.02 THEN CONCAT(character, chr(65 + (abs(ASCII(character)) + CAST(random() * 25 AS INT)) % 26))
                            ELSE character END,
                            ''
                        )
                    FROM
                        UNNEST(SPLIT(tweet, '')) AS t(character)
                )
                -- Introduce deletion errors
                ELSE (
                    SELECT
                        STRING_AGG(
                            character,
                            ''
                        )
                    FROM
                        UNNEST(SPLIT(tweet, '')) AS t(character)
                    WHERE
                        random() > 0.02
                )
                -- Introduce transposition errors
                -- WHEN random() < 0.1 THEN (
                --     SELECT
                --         STRING_AGG(
                --             CONCAT(
                --                 LEAST(character1, character2),
                --                 GREATEST(character1, character2)
                --             ),
                --             ''
                --         )
                --     FROM
                --         UNNEST(SPLIT(tweet, '')) AS t(character1)
                --     LEFT JOIN
                --         UNNEST(SPLIT(tweet, '')) AS u(character2)
                --     ON
                --         random() < 0.001
                -- )
                -- ELSE tweet
            END AS tweet
            FROM data_to_corrupt
            ORDER BY row_id
            """).df()['tweet']
        df['tweet'].iloc[indices_to_typo] = corrupted_data
        return df

    warnings.filterwarnings('ignore')
    typo_adder = FunctionTransformer(add_typos)

    # typo_transformation = WordSwapQWERTY(random_one=False)
    # typo_transformation = CompositeTransformation(
    #     [WordSwapRandomCharacterDeletion(), WordSwapQWERTY()]
    # )
    # typo_transformation = WordSwapRandomCharacterSubstitution(random_one=True)
    # typo_augmenter = Augmenter(transformation=typo_transformation, fast_augment=True, transformations_per_example=4,
    #                            pct_words_to_swap=0.8)
    # def add_typos(df):
    #     indices = numpy.arange(len(df))
    #     numpy.random.shuffle(indices)
    #     num_values_to_typo = int(len(df) * fraction_to_typo)
    #     indices_to_typo = indices[:num_values_to_typo]
    #     df.loc[indices_to_typo, 'tweet'] = df.loc[indices_to_typo, 'tweet'].apply(lambda txt: typo_augmenter.augment(txt)[0])
    #     return df
    # warnings.filterwarnings('ignore')
    # typo_adder = FunctionTransformer(add_typos)

    # corrupted_tweet = BrokenCharacters(column='tweet', fraction=.2).transform(test)
    return typo_adder

def get_typo_fixer():
    # def fix_typos(df):
    #     df['tweet'] = df['tweet'].map(lambda txt: str(TextBlob(txt).correct()))
    #     # TODO: This is very slow. We might need to look into different libraries.
    #     return df
    # warnings.filterwarnings('ignore')
    # typo_fixer = FunctionTransformer(fix_typos)
    spell = Speller()

    def fix_typos(df):
        # df['tweet'] = df['tweet'].map(lambda txt: str(TextBlob(txt).correct()))
        df['tweet'] = df['tweet'].map(lambda txt: spell(txt))
        # TODO: This spellchecker is much faster. However, I am not entirely sure how good it is
        return df

    warnings.filterwarnings('ignore')
    typo_fixer = FunctionTransformer(fix_typos)
    return typo_fixer
