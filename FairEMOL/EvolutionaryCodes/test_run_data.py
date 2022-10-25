"""
测试数据导入
https://github.com/JSGoedhart/fairness-comparison/tree/df503855bfc2eeb9c89c1075b661bf8de5e6d18c
"""

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData


dataset = get_dataset_names()
print(dataset)
supported_tags = ["original", "numerical", "numerical-binsensitive",
                  "categorical-binsensitive", "numerical-for-NN"]

for dataset_obj in DATASETS:
    print("\nIn dataset of ", dataset_obj.dataset_name)
    processed_dataset = ProcessedData(dataset_obj)
    train_test_splits = processed_dataset.create_train_test_splits()
    all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
    for sensitive in all_sensitive_attributes:
        print(" Sensitive attribute:" + sensitive)
        for supported_tag in supported_tags:
            if supported_tag in train_test_splits.keys():
                data = train_test_splits[supported_tag]
                print("  type: ", supported_tag, ' --- is OK')
