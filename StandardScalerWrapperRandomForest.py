import numpy
import numpy as np
import pandas as pd
import pickle
import argparse

import logging
logger = logging.getLogger()

def split_dataset(X, y, weights, split_ratio, should_stratify):
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)


def get_training_dataset(dataset_id):
    from azureml.core.dataset import Dataset
    from azureml.core.run import Run
    
    logger.info("Running get_training_dataset")
    ws = Run.get_context().experiment.workspace
    dataset = Dataset.get_by_id(workspace=ws, id=dataset_id)
    return dataset.to_pandas_dataframe()


def prepare_data(dataframe):
    from azureml.training.tabular.preprocessing import data_cleaning
    
    logger.info("Running prepare_data")
    label_column_name = 'Category'
    
    # extract the features, target and sample weight arrays
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
     is_timeseries=False, target_column=label_column_name)
    
    return X, y, sample_weights


def get_mapper_9133f9(column_names):
    from azureml.training.tabular.featurization.categorical.cat_imputer import CatImputer
    from azureml.training.tabular.featurization.categorical.labelencoder_transformer import LabelEncoderTransformer
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': CatImputer,
                'copy': True,
            },
            {
                'class': StringCastTransformer,
            },
            {
                'class': LabelEncoderTransformer,
                'hashing_seed_val': 314489979,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_49c852(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from azureml.training.tabular.featurization.utilities import wrap_in_list
    from numpy import uint8
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': CountVectorizer,
                'analyzer': 'word',
                'binary': True,
                'decode_error': 'strict',
                'dtype': numpy.uint8,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 1.0,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (1, 1),
                'preprocessor': None,
                'stop_words': None,
                'strip_accents': None,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': wrap_in_list,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_369e16(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from numpy import float32
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': TfidfVectorizer,
                'analyzer': 'char',
                'binary': False,
                'decode_error': 'strict',
                'dtype': numpy.float32,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 0.95,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (3, 3),
                'norm': 'l2',
                'preprocessor': None,
                'smooth_idf': True,
                'stop_words': None,
                'strip_accents': None,
                'sublinear_tf': False,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': None,
                'use_idf': False,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_2cf0a8(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from numpy import float32
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': TfidfVectorizer,
                'analyzer': 'word',
                'binary': False,
                'decode_error': 'strict',
                'dtype': numpy.float32,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 1.0,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (1, 2),
                'norm': 'l2',
                'preprocessor': None,
                'smooth_idf': True,
                'stop_words': None,
                'strip_accents': None,
                'sublinear_tf': False,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': None,
                'use_idf': False,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_ab1045(column_names):
    from sklearn.impute import SimpleImputer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': SimpleImputer,
                'add_indicator': False,
                'copy': True,
                'fill_value': None,
                'missing_values': numpy.nan,
                'strategy': 'mean',
                'verbose': 0,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_ed5fc4(column_names):
    from azureml.training.tabular.featurization.generic.imputation_marker import ImputationMarker
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': ImputationMarker,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_d05ded(column_names):
    from azureml.training.tabular.featurization.categorical.hashonehotvectorizer_transformer import HashOneHotVectorizerTransformer
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': HashOneHotVectorizerTransformer,
                'hashing_seed_val': 314489979,
                'num_cols': 4096,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_e30a49(column_names):
    from azureml.training.tabular.featurization.data.wordembeddings_provider import WordEmbeddingsProvider
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from azureml.training.tabular.featurization.text.wordembedding_transformer import WordEmbeddingTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': WordEmbeddingTransformer,
                'embeddings_provider': WordEmbeddingsProvider(embeddings_name='wiki_news_300d_1M_subword'),
                'token_pattern': '(?u)\\b\\w+\\b',
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_dbf06b(column_names):
    from azureml.training.tabular.featurization.text.bilstm_attention_transformer import BiLSTMAttentionTransformer
    from azureml.training.tabular.featurization.text.string_concat_transformer import StringConcatTransformer
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': StringConcatTransformer,
                'separator': '. ',
            },
            {
                'class': BiLSTMAttentionTransformer,
                'batch_size': 128,
                'device': 'cpu',
                'do_early_stopping': False,
                'embeddings_name': 'glove_6B_300d_word2vec',
                'epochs': 2,
                'iter_cnt': 20,
                'learning_rate': 0.005,
                'max_rows': 10000,
                'seed': None,
                'split_ratio': 0.8,
                'top_k': 1,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def generate_data_transformation_config():
    from sklearn.pipeline import FeatureUnion
    
    column_group_1 = ['default_location']
    
    column_group_2 = [['merchant_cat_code'], ['amt']]
    
    column_group_3 = ['payment_category']
    
    column_group_4 = ['trans_desc', 'default_brand', 'qrated_brand', 'coalesced_brand']
    
    column_group_5 = ['qrated_brand']
    
    column_group_6 = [['trans_desc'], ['default_brand'], ['qrated_brand'], ['coalesced_brand']]
    
    column_group_7 = ['sor', 'db_cr_cd', 'is_international']
    
    column_group_8 = [['merchant_cat_code']]
    
    feature_union = FeatureUnion([
        ('mapper_9133f9', get_mapper_9133f9(column_group_7)),
        ('mapper_49c852', get_mapper_49c852(column_group_3)),
        ('mapper_369e16', get_mapper_369e16(column_group_4)),
        ('mapper_2cf0a8', get_mapper_2cf0a8(column_group_4)),
        ('mapper_ab1045', get_mapper_ab1045(column_group_2)),
        ('mapper_ed5fc4', get_mapper_ed5fc4(column_group_8)),
        ('mapper_d05ded', get_mapper_d05ded(column_group_1)),
        ('mapper_e30a49', get_mapper_e30a49(column_group_5)),
        ('mapper_dbf06b', get_mapper_dbf06b(column_group_6)),
    ])
    return feature_union
    
    
def generate_preprocessor_config():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    
    return preproc
    
    
def generate_algorithm_config():
    from sklearn.ensemble import RandomForestClassifier
    
    algorithm = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight=None,
        criterion='gini',
        max_depth=None,
        max_features='auto',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_pipeline_with_ytransformer(pipeline):
    from azureml.training.tabular.models.pipeline_with_ytransformations import PipelineWithYTransformations
    from sklearn.preprocessing import LabelEncoder
    
    transformer = LabelEncoder()
    transformer_name = "LabelEncoder"
    return PipelineWithYTransformations(pipeline, transformer_name, transformer)
    
def build_model_pipeline():
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('featurization', generate_data_transformation_config()),
            ('preproc', generate_preprocessor_config()),
            ('model', generate_algorithm_config()),
        ]
    )
    
    return generate_pipeline_with_ytransformer(pipeline)


def train_model(X, y, sample_weights=None, transformer=None):
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    
    model = model_pipeline.fit(X, y)
    return model


def calculate_metrics(model, X, y, sample_weights, X_test, y_test, cv_splits=None):
    from azureml.training.tabular.score.scoring import score_classification
    
    y_pred_probs = model.predict_proba(X_test)
    if isinstance(y_pred_probs, pd.DataFrame):
        y_pred_probs = y_pred_probs.values
    class_labels = np.unique(y)
    train_labels = model.classes_
    metrics = score_classification(
        y_test, y_pred_probs, get_metrics_names(), class_labels, train_labels, use_binary=True)
    return metrics
def get_metrics_names():
    metrics_names = [
        'f1_score_macro',
        'matthews_correlation',
        'precision_score_weighted',
        'classification_report',
        'iou_weighted',
        'average_precision_score_classwise',
        'AUC_weighted',
        'norm_macro_recall',
        'iou_classwise',
        'precision_score_classwise',
        'average_precision_score_micro',
        'AUC_classwise',
        'confusion_matrix',
        'average_precision_score_macro',
        'accuracy_table',
        'accuracy',
        'recall_score_weighted',
        'weighted_accuracy',
        'iou',
        'average_precision_score_binary',
        'recall_score_classwise',
        'precision_score_macro',
        'iou_macro',
        'f1_score_weighted',
        'AUC_micro',
        'recall_score_macro',
        'balanced_accuracy',
        'f1_score_binary',
        'AUC_macro',
        'f1_score_micro',
        'AUC_binary',
        'recall_score_micro',
        'f1_score_classwise',
        'iou_micro',
        'precision_score_micro',
        'recall_score_binary',
        'average_precision_score_weighted',
        'log_loss',
        'precision_score_binary',
    ]
    return metrics_names


def main(training_dataset_id=None):
    from azureml.core.run import Run
    
    run = Run.get_context()
    
    df = get_training_dataset(training_dataset_id)
    X, y, sample_weights = prepare_data(df)
    split_ratio = 0.2
    try:
        (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y, sample_weights, split_ratio, should_stratify=True)
    except Exception:
        (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y, sample_weights, split_ratio, should_stratify=False)
    model = train_model(X_train, y_train, sample_weights_train)
    
    metrics = calculate_metrics(model, X, y, sample_weights, X_test=X_valid, y_test=y_valid)
    
    print(metrics)
    for metric in metrics:
        run.log(metric, metrics[metric])
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    run.upload_file('outputs/model.pkl', 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_id', type=str, default='bec36c76-bee0-4baf-8e24-599f7eb5af2a', help='Default training dataset id is populated from the parent run')
    args = parser.parse_args()
    
    main(args.training_dataset_id)