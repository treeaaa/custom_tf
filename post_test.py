import requests
import json
headers = {"Content-Type": "application/json"}
data = {"epochs": 20,
        "batch_size": 32,
        "problem": "classification",
        "lr_rate": 1e-3,
        "file_path": "C:\\Users\\carteryang\\Desktop\\III_project\\custom_tf2\\titanic\\train.csv",
        "feature_select": ["Sex", "Age", "Fare"],
        "label_select": ["Survived"],
        "missing": "padavg",
        "model_name": "MLP",
        "loss_name": "SparseCategoricalCrossentropy",
        "optimizer_name": "Adam",
        "activation_name":"relu"}
# r = requests.get('http://127.0.0.1:5000/custom_tf/arg')
# print(r.text)
data = json.dumps(data)
# print(data)
a = requests.post("http://127.0.0.1:5000/custom_tf/arg",
                  data=data, headers=headers)
print(a.text)