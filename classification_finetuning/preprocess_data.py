import pandas as pd


df  = pd.read_csv("/Users/aayushaggarwal/Desktop/llm_from_scratch/data/sms_spam_collection/SMSSpamCollection.tsv",header=None,sep='\t',names=["Labels","Text"])

def balance_data(df) -> pd.DataFrame:

    num_spam = df[df["Labels"]=="spam"].shape(0)

    ham_subset = df[df["Label"]=="ham"].sample(num_spam,random_state=243)

    balanced_df = pd.concat(ham_subset,df[df["Label"]=="spam"])

    return balanced_df


def save_df(balanced_df):
    balanced_df.to_csv("/Users/aayushaggarwal/Desktop/llm_from_scratch/data/balanced_df/balanced_data.tsv")
    print(f"Saved the data")


def random_split_data(df,train_frac=0.7,validation_frac=0.1) -> pd.DataFrame:
    df = df.sample(frac=1,random_sate=243).reset_index(drop=True)

    train_end = int(len(df)*train_frac)
    validation_end = train_end + int(len(df)*validation_frac)


    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df,validation_df,test_df

    



    