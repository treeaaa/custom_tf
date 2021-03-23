import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask
from flask import request
import json

app = Flask(__name__)


def load_csv(file_path, feature_select, label_select, missing):
    # file_path (str): 檔案位置 ex. r'C:\Users\carteryang\1_tf\titanic\train.csv'
    # feature_select(list(str)): Feature選擇的欄位名稱 ex ['Ticket','Fare','Cabin']
    # label_select(list(str)): label選擇的欄位名稱 ex ['Survived']
    # missing(str): 缺失資料的處理方式，支援
    #   "default":將缺失值也看成一種資料
    #   "abandon":直接捨去缺失資料
    #   "pad0": 填充0
    #   "padavg":填充平均數
    #   "padmode":填充眾數
    # reuturn (df,df): 回傳處理完的dataframe(feature部分跟label部分)
    df = pd.read_csv(file_path)
    if missing == 'padavg':
        df = df.fillna(df.mean())
    df_feature = df[feature_select]
    df_label = df[label_select].squeeze()
    return df_feature, df_label, df


def preprocess(df_feature, df_ori):
    # from https://www.tensorflow.org/tutorials/load_data/csv
    # return df_dict 以及 df_preprocessing_layer用於前處理
    # 可使用 df_preprocessing_layer(df_dict) 取的資料
    inputs = {}
    for name, column in df_feature.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    #  ############
    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}

    x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
    norm = preprocessing.Normalization()
    norm.adapt(np.array(df_ori[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    # ############
    preprocessed_inputs = [all_numeric_inputs]
    # ############
    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
        lookup = preprocessing.StringLookup(
            vocabulary=np.unique(df_feature[name]))
        one_hot = preprocessing.CategoryEncoding(
            max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)
    #   ###########
    #
    preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)
    df_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
    # tf.keras.utils.plot_model(
    #     model=df_preprocessing, rankdir="LR", dpi=72, show_shapes=True)
    # #######
    df_feature_dict = {name: np.array(value)
                       for name, value in df_feature.items()}
    return df_feature_dict, df_preprocessing, inputs


def model_choice(model_name, input_shape, output_shape):
    # 選擇model
    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16()
    elif model_name == 'ResNet50':
        model = tf.keras.applications.ResNet50()
    elif model_name == "MLP":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(output_shape)
        ])
    return model


def loss_choice(loss_name, logits=True):
    # 選擇loss
    if loss_name == 'MSE':
        loss = tf.keras.losses.MSE()
    elif loss_name == 'MAE':
        loss = tf.keras.losses.MAE()
    elif loss_name == 'SparseCategoricalCrossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=logits)
    elif loss_name == 'BinaryCrossentropy':
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=logits)
    return loss


def optimizer_choice(optimizer_name, lr_rate):
    # 選擇優化器
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_rate)
    elif optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_rate)
    return optimizer


def custom_tf(model_name, loss_name, optimizer_name, preprocessing_layer, inputs, input_shape, output_shape, lr_rate=1e-3):
    model = model_choice(model_name=model_name,
                         input_shape=input_shape,
                         output_shape=output_shape)
    preprocessing_inputs = preprocessing_layer(inputs)
    final = model(preprocessing_inputs)
    model_return = tf.keras.Model(inputs, final)

    optimizer = optimizer_choice(optimizer_name, lr_rate)
    loss = loss_choice(loss_name)
    model_return.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=['accuracy'])
    return model_return


@app.route('/', methods=['GET'])
def return_hi():
    # 用於測試
    return 'h'


@app.route('/custom_tf/arg', methods=['POST'])
def build():
    req = request.get_json()

    epochs = req['epochs']
    batch_size = req['batch_size']
    problem = req['problem']
    lr_rate = req['lr_rate']
    file_path = req['file_path']
    feature_select = req['feature_select']
    label_select = req['label_select']
    missing = req['missing']
    model_name = req['model_name']
    loss_name = req['loss_name']
    optimizer_name = req['optimizer_name']

    df_feature, df_label, df_ori = load_csv(file_path=file_path,
                                            feature_select=feature_select,
                                            label_select=label_select,
                                            missing=missing)
    df_feature_dict, df_preprocessing_layer, inputs = preprocess(df_feature=df_feature,
                                                                 df_ori=df_ori)
    if problem == 'regression':
        output_shape = 1
    elif problem == 'classification':
        output_shape = max(df_label) + 1
    model = custom_tf(model_name=model_name,
                      loss_name=loss_name,
                      optimizer_name=optimizer_name,
                      preprocessing_layer=df_preprocessing_layer,
                      inputs=inputs,
                      input_shape=df_preprocessing_layer.output_shape[-1],
                      # tf.keras.layer.output_shape = return (None,x)
                      # preprocess完成之後shape會改變(可能有onehot)
                      # 所以model intput以preprocess之後的為主
                      output_shape=output_shape,
                      lr_rate=lr_rate)
    df_ds = tf.data.Dataset.from_tensor_slices((df_feature_dict, df_label))

    df_batches = df_ds.shuffle(len(df_label)).batch(batch_size)

    train_history = model.fit(df_batches, epochs=epochs)
    loss = train_history.history['loss']

    return {'train_loss': loss}


if __name__ == '__main__':
    app.run(debug=True)
    # res = build(epochs=20,
    #             batch_size=32,
    #             problem='classification',
    #             lr_rate=1e-3,
    #             file_path=r'C:\Users\carteryang\1_tf\titanic\train.csv',
    #             feature_select=['Sex', 'Age', 'Fare'],
    #             label_select=['Survived'],
    #             missing='padavg',
    #             model_name='MLP',
    #             loss_name='SparseCategoricalCrossentropy',
    #             optimizer_name='Adam',
    #             )
