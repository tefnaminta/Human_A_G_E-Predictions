import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./age_gender.csv/age_gender.csv')

# collecting basic Information
def data_info():
    print(df.info())
    print(df.describe())
    print(df['age'].value_counts())
    print(df['ethnicity'].value_counts())
    print(df['gender'].value_counts())

# data preprocessing
def preprocessing_data(df):
    num_pixels = len(df['pixels'][0].split(" "))
    img_height = int(np.sqrt(len(df['pixels'][0].split(" "))))
    img_width = int(np.sqrt(len(df['pixels'][0].split(" "))))
    # print(num_pixels, img_height, img_width)
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))
    for i in range(len(df['pixels'])):
        df['pixels'][i] = df['pixels'][i].reshape(48,48)
    return df

# data visualization
def data_visualization():
    plt.figure(figsize=(25, 25))

    for i in range(9):
        index = np.random.randint(0, len(df))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(3,3, i+1)
        plt.imshow(df['pixels'].iloc[index].reshape(48, 48),"gray")
        plt.title(' Age: {}\n Ethnicity: {}\n gender: {}'.format(df['age'].iloc[index], {0:"White", 1:"Black", 2:"Asian", 3:"Indian", 4:"Hispanic"}[df['ethnicity'].iloc[index]], {0:"Male", 1:"Female"}[df['gender'].iloc[index]]),loc="left",color='red',fontsize = 8)
    plt.show()

# data_info()
data = preprocessing_data(df)
# data_visualization()
age_y = np.array(df['age'])
gender_y = np.array(df['gender'])
eth_y = np.array(df['ethnicity'])