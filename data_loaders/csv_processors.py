import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import numpy as np

def process_rsna_csv(image_root, csv_data):
    """
    Insert full pathes to images in new Path column
    Args:
        image_root: Path to dir with images
        csv_data: Path to file of pundas.Dataframe

    Returns:
        updated pundas.Datasrame with correct image pathes in Path column
    """
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    # Define proper path for each image
    csv_data["Path"] = str(image_root)+ "/" + csv_data["patientId"] + ".dcm"

    return csv_data


def process_chest14_csv(image_root, csv_data):
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    csv_data["Path"] = str(image_root) + "/" + csv_data["Image Index"]
    csv_data["Target"] = (csv_data["Finding Labels"] != "No Finding").astype(np.uint8())

    return csv_data


def process_chexpert_csv(image_root, csv_data):
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    csv_data = csv_data[csv_data["Frontal/Lateral"] == "Frontal"]
    csv_data["Path"] = str(image_root) + "/" + csv_data["Path"]
    csv_data["Target"] = (csv_data['No Finding'] != 1.0).astype(np.uint8())

    return csv_data

def process_chexpert_5_csv(image_root, csv_data):
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    csv_data = csv_data[csv_data["Frontal/Lateral"] == "Frontal"]
    csv_data["Path"] = str(image_root) + "/" + csv_data["Path"]
    csv_data = csv_data[(csv_data["Atelectasis"].notnull()) | (csv_data["Cardiomegaly"].notnull()) | (csv_data["Consolidation"].notnull()) | (csv_data["Edema"].notnull()) | (csv_data["Pleural Effusion"].notnull())]

    csv_data["Atelectasis"] = (csv_data['Atelectasis'].notnull() & (csv_data['Atelectasis'] != 0.0)).astype(np.uint8())
    csv_data["Cardiomegaly"] = (csv_data['Cardiomegaly'].notnull() & (csv_data['Cardiomegaly'] != 0.0)).astype(np.uint8())
    csv_data["Consolidation"] = (csv_data['Consolidation'].notnull() & (csv_data['Consolidation'] != 0.0)).astype(np.uint8())
    csv_data["Edema"] = (csv_data['Edema'].notnull() & (csv_data['Edema'] != 0.0)).astype(np.uint8())
    csv_data["Pleural Effusion"] = (csv_data['Pleural Effusion'].notnull() & (csv_data['Pleural Effusion'] != 0.0)).astype(np.uint8())
    
    return csv_data

def process_chexpert_rare_6_csv(image_root, csv_data):
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    df = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    df = df[df["Frontal/Lateral"] == "Frontal"]
    df["Path"] = str(image_root) + "/" + csv_data["Path"]

    rare_findings = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Lung Lesion",
        "Pneumonia",
        "Pneumothorax",
        "Fracture"]
    
    rare_idx = df["No Finding"] == 10
    for rare_f in rare_findings:
        rare_idx = rare_idx | ((df[rare_f] == 1.0) | (df[rare_f] == -1.0))
        
    rare_df = df[rare_idx]
    rare_df['Other'] = 0

    other_fundings = ['Atelectasis','Cardiomegaly','Consolidation','Edema', 'Pleural Effusion', 'Lung Opacity', 'Pleural Other','Support Devices']
    has_other = rare_df['Other'] == 1

    for f in other_fundings:
        has_other = has_other | ((rare_df[f]==1.0)|(rare_df[f] == -1.0))
        
    rare_df['Other'] = has_other.astype(np.uint8())
    

    total_df = rare_df
    all_findings = ['Atelectasis','Cardiomegaly','Consolidation','Edema', 'Pleural Effusion', 
                            'No Finding', 'Enlarged Cardiomediastinum','Lung Opacity', 'Lung Lesion',
                            'Pneumonia','Pneumothorax', 'Pleural Other','Fracture', 'Support Devices', "Other"]

    for f in all_findings:
        total_df[f] = (total_df[f].notnull() & (total_df[f] != 0.0)).astype(np.uint8())

    return total_df


def process_chexpert_14_csv(image_root, csv_data):
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    df = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    df = df[df["Frontal/Lateral"] == "Frontal"]
    df["Path"] = str(image_root) + "/" + csv_data["Path"]

    all_findings = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]


    for f in all_findings:
        df[f] = (df[f].notnull() & (df[f] != 0.0)).astype(np.uint8())

    return df


def process_tbx11k_csv(image_root, csv_data):
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    csv_data["Path"] = str(image_root) + "/" + csv_data["Path"]
    csv_data["Target"] = (csv_data["Path"].str.split("/").str.get(-2) != "health").astype(np.uint8())
    return csv_data


def process_vinbigdata_csv(image_root, csv_data):
    """
    Insert full pathes to images in new Path column and assign proper targets
    Args:
        image_root: Path to dir with images
        csv_data: Path to file of pundas.Dataframe

    Returns:
        updated pundas.Datasrame with correct image pathes in Path column
    """
    image_root = pathlib.Path(image_root) if not isinstance(image_root, pathlib.PosixPath) else image_root
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    # Define proper path for each image
    csv_data["Path"] = str(image_root)+ "/" + csv_data["image_id"] + ".jpg"

    # Define proper target fpr each image
    csv_data["Target"] = (csv_data["class_name"] != "No finding").astype(np.uint8())

    clean_dfs = []
    for group_name, group_df in csv_data.groupby(by=["Path"]):
        
        label_sum = group_df['Target'].sum()
        if label_sum > len(group_df)/2:
            label = 1
        else:
            label = 0
        
        clean_df = pd.DataFrame([{"Path":group_name,"Target": label}])
        clean_dfs.append(clean_df)

    csv_data = pd.concat(clean_dfs)

    return csv_data


def create_pmeumonia_full_csv(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    train_dir =  dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    phases = {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    }
    targets = {
        "NORMAL": 0,
        "PNEUMONIA": 1
    }

    dataframes = []

    for phase, phase_dir in phases.items():
        for target_folder, target in targets.items():
            img_dir = phase_dir / target_folder
            filenames = [str(img_path) for img_path in img_dir.glob("*.jpeg")]
            df = pd.DataFrame(filenames,columns=["Path"])
            df["Target"] = target
            df["Phase"] = phase

            dataframes.append(df)

    return pd.concat(dataframes) 


def create_chest14_full_csv(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    
    # Read full csv 
    csv_data =  pd.read_csv(dataset_dir / "Data_Entry_2017.csv")

    # Preprocess csv to have full paths to images and int labels pathology(1)/normal(0)
    csv_data = process_chest14_csv(dataset_dir/"images", csv_data)

    # Create list of train_val images
    train_val_images = pd.read_csv(dataset_dir / "train_val_list.txt", header=None)
    train_val_images.columns = ["Image Index"]

    # Filter out train_val images
    train_val_csv = csv_data[csv_data["Image Index"].isin(train_val_images["Image Index"].values)]
    
    # Split into separate train and val partitions (80%/20%)
    train_csv, val_csv = train_test_split(train_val_csv, test_size=0.2, stratify=train_val_csv["Target"].values)

    # Create list of test images
    test_images = pd.read_csv(dataset_dir / "test_list.txt", header=None)
    test_images.columns = ["Image Index"]

    # Filter out test images
    test_csv = csv_data[csv_data["Image Index"].isin(test_images["Image Index"].values)]

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"
    test_csv["Phase"] = "test"

    return pd.concat([train_csv, val_csv, test_csv]) 


def create_rsna_full_csv(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    # Read full csv 
    train_val_csv =  pd.read_csv(dataset_dir / "stage_2_train_labels.csv")

    # Preprocess csv to have full paths to images and int labels pathology(1)/normal(0)
    train_val_csv = process_rsna_csv(dataset_dir / "stage_2_train_images", train_val_csv)

    # Split into separate train and val partitions (80%/20%)
    train_csv, val_csv = train_test_split(train_val_csv, test_size=0.2, stratify=train_val_csv["Target"].values)

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"
    
    return pd.concat([train_csv, val_csv])


def create_gb7_flg_full_csv(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    targets = {
        "MD__20190111-1231_pnevmoniya/Undef": 1,
        "MD__20200424__norma/Undef": 0
    }

    # Collect pneumonia and normal images into one dataframe
    dataframes = []
    for target_folder, target in targets.items():
        img_dir = dataset_dir / target_folder
        filenames = [str(img_path) for img_path in img_dir.glob("*")]
        df = pd.DataFrame(filenames,columns=["Path"])
        df["Target"] = target

        dataframes.append(df)

    csv_data = pd.concat(dataframes)

    # Split into separate train and val partitions (80%/20%)
    train_csv, val_csv = train_test_split(csv_data, test_size=0.2, stratify=csv_data["Target"].values)

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"

    return pd.concat([train_csv, val_csv])

def create_chexpert_full_csv(dataset_dir, csv_data):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    # Read full csv 
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    # Group by patient id
    csv_data["patient"] = csv_data["Path"].str.split('/').str[2]

    # Preprocess csv to have full paths to images and int labels pathology(1)/normal(0)
    csv_data = process_chexpert_csv(dataset_dir, csv_data)
    
    # Split to have patient images only in 1 partition
    train_data, val_data = train_test_split(
                np.array(list(csv_data.groupby(by=["patient"]).indices.items()),dtype=object), 
                test_size=0.2)
    
    train_csv = csv_data.iloc[np.concatenate(train_data[:,1])]
    val_csv = csv_data.iloc[np.concatenate(val_data[:,1])]

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"

    return pd.concat([train_csv, val_csv])

def create_chexpert_5_full_csv(dataset_dir, csv_data):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    # Read full csv 
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    # Group by patient id
    csv_data["patient"] = csv_data["Path"].str.split('/').str[2]

    # Preprocess csv to have full paths to images and int labels pathology(1)/normal(0)
    csv_data = process_chexpert_5_csv(dataset_dir, csv_data)
    
    # Split to have patient images only in 1 partition
    train_data, val_data = train_test_split(
                np.array(list(csv_data.groupby(by=["patient"]).indices.items()),dtype=object), 
                test_size=0.2)
    
    train_csv = csv_data.iloc[np.concatenate(train_data[:,1])]
    val_csv = csv_data.iloc[np.concatenate(val_data[:,1])]

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"

    return pd.concat([train_csv, val_csv])

def create_chexpert_14_full_csv(dataset_dir, csv_data):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    # Read full csv 
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    # Group by patient id
    csv_data["patient"] = csv_data["Path"].str.split('/').str[2]

    # Preprocess csv to have full paths to images and int labels 1/0 for each finding, None -> 0, -1 -> 1
    csv_data = process_chexpert_14_csv(dataset_dir, csv_data)
    
    # Split to have patient images only in 1 partition
    train_data, val_data = train_test_split(
                np.array(list(csv_data.groupby(by=["patient"]).indices.items()),dtype=object), 
                test_size=0.2)
    
    train_csv = csv_data.iloc[np.concatenate(train_data[:,1])]
    val_csv = csv_data.iloc[np.concatenate(val_data[:,1])]

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"

    return pd.concat([train_csv, val_csv])

def create_chexpert_rare_6_full_csv(dataset_dir, csv_data):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    # Read full csv 
    csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data

    # Group by patient id
    csv_data["patient"] = csv_data["Path"].str.split('/').str[2]

    # Preprocess csv to have full paths to images and int labels pathology(1)/normal(0)
    csv_data = process_chexpert_rare_6_csv(dataset_dir, csv_data)
    
    # Split to have patient images only in 1 partition
    train_data, val_data = train_test_split(
                np.array(list(csv_data.groupby(by=["patient"]).indices.items()),dtype=object), 
                test_size=0.2)
    
    train_csv = csv_data.iloc[np.concatenate(train_data[:,1])]
    val_csv = csv_data.iloc[np.concatenate(val_data[:,1])]

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"

    return pd.concat([train_csv, val_csv])


def create_tbx11k_full_csv(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir
    
    files = [
        {
            "list": "lists/TBX11K_train.txt",
            "phase": "train"
        },
        {
            "list": "lists/TBX11K_val.txt",
            "phase": "val"
        }
    ]
    dataframes = []

    for f in files:
        csv_data = pd.read_csv(dataset_dir / f["list"], header=None)
        csv_data.columns = ["Path"]

        csv_data = process_tbx11k_csv(dataset_dir / "imgs", csv_data)
        csv_data["Phase"] = f["phase"]

        dataframes.append(csv_data)

    return pd.concat(dataframes) 


def create_vinbigdata_full_csv(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir) if not isinstance(dataset_dir, pathlib.PosixPath) else dataset_dir

    train_dir =  dataset_dir / "train"
    test_dir = dataset_dir / "test"

    # Process train_val data
    train_val_csv = process_vinbigdata_csv(train_dir, dataset_dir/"train.csv")
    
    # Split train_val data on train and val sets
    train_csv, val_csv = train_test_split(train_val_csv, test_size=0.2, stratify=train_val_csv["Target"].values)
    
    # Process test data
    #test_csv = process_vinbigdata_csv(test_dir, dataset_dir/"test.csv")

    # Assign relevant phase
    train_csv["Phase"] = "train"
    val_csv["Phase"] = "val"
    #test_csv["Phase"] = "test"

    return pd.concat([train_csv, val_csv]) 







