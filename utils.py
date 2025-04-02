from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def get_train_test(real, fake=None, test_size=0.2, random_state=42):
    def _gather_paths(root_dir, label):
        paths = []
        for subdir, dirs, files in os.walk(root_dir):
            dirs.sort()
            files.sort()
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append({
                        'path': os.path.join(subdir, file),
                        'label': label
                    })
        return paths

    real_data = _gather_paths(real, 1)
    if fake is None:
       return pd.DataFrame(real_data), None
    
    fake_data = _gather_paths(fake, 0)
    df = pd.DataFrame(real_data + fake_data)
    df['label'] = df['label'].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )

    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, test_df




def get_classification_report(pred_df):
    # get the true labels
    y_true = pred_df['label'].values
    # get the predicted labels
    y_pred = pred_df['predictions'].values

    # get the classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    return pd.DataFrame(report).transpose()



def plot_confusion_matrix(pred_df):
    # get the true labels
    y_true = pred_df['label'].values
    # get the predicted labels
    y_pred = pred_df['predictions'].values

    # get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()