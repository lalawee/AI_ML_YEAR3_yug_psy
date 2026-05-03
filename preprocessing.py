import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(data):
    """
    Unified preprocessing function for the Titanic survival prediction pipeline.

    Performs the following steps:
      1. Cleans the 'Passenger Fare' column (strips '$' and ',' then casts to float).
      2. Encodes 'Survived' from 'Yes'/'No' to 1/0 (if column is present).
      3. Label-encodes categorical columns: 'Ticket Class', 'Embarkation Country', 'Gender'.
      4. Drops irrelevant columns: 'Passenger ID', 'Ticket Number', 'Cabin', 'Name'.
      5. Fills remaining missing values with the column median (numeric columns only).

    Parameters
    ----------
    data : pd.DataFrame
        Raw dataset loaded from CSV.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except 'Survived').
    y : pd.Series or None
        Target vector (1 = survived, 0 = did not survive).
        Returns None if 'Survived' was not present in the input data.
    """
    data = data.copy()

    # 1. Clean Passenger Fare
    data['Passenger Fare'] = (
        data['Passenger Fare']
        .replace({'\\$': '', ',': ''}, regex=True)
        .astype(float)
    )

    # 2. Encode target column using explicit Yes/No -> 1/0 mapping
    has_target = 'Survived' in data.columns
    if has_target:
        data['Survived'] = data['Survived'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 3. Label-encode categorical feature columns
    le = LabelEncoder()
    for col in ['Ticket Class', 'Embarkation Country', 'Gender']:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))

    # 4. Drop irrelevant columns
    cols_to_drop = [c for c in ['Passenger ID', 'Ticket Number', 'Cabin', 'Name'] if c in data.columns]
    data = data.drop(cols_to_drop, axis=1)

    # 5. Fill missing values with column median (numeric columns only)
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Split into features and target
    if has_target:
        X = data.drop('Survived', axis=1)
        y = data['Survived']
    else:
        X = data
        y = None

    return X, y
