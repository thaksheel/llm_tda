import pandas as pd

# Path to your QNLI file
df_train = pd.read_csv(
    "D:/datasets/QNLIv2_glue/train.tsv",
    sep="\t",
    engine="python",  # more tolerant parser
    on_bad_lines="skip",  # skip malformed lines
)
df_test = pd.read_csv(
    "D:/datasets/QNLIv2_glue/test.tsv",
    sep="\t",
    engine="python",  # more tolerant parser
    on_bad_lines="skip",  # skip malformed lines
)

print(df_train.head())
print(len(df_train))
print("END")
