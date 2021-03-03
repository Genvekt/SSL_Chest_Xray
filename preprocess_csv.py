from data_loaders.csv_processors import create_chest14_full_csv, create_gb7_flg_full_csv, create_tbx11k_full_csv
from data_loaders.csv_processors import create_chexpert_full_csv, create_pmeumonia_full_csv, create_rsna_full_csv
from data_loaders.csv_processors import create_vinbigdata_full_csv, create_chexpert_5_full_csv

import pandas as pd
pd.options.mode.chained_assignment = None  
                                        
if __name__ == "__main__":
    # df = create_chest14_full_csv("/datasets/chest-14")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/chest-14.csv")

    # df = create_rsna_full_csv("/datasets/rsna")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/rsna.csv")

    # df = create_pmeumonia_full_csv("/datasets/chest-xray-pneumonia")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/chest-xray-pneumonia.csv")

    # df = create_gb7_flg_full_csv("/datasets/GB7/FLG")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/gb7-flg.csv")

    # df = create_chexpert_full_csv("/new_data/CheXpert", "/home/intern/SSL_Chest_Xray/datasets/chexpert_train_fixed.csv")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/chexpert.csv")

    # df = create_tbx11k_full_csv("/new_data/TBX11K")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/tbx11k.csv")

    # df = create_vinbigdata_full_csv("/datasets/vinbigdata")
    # df.to_csv("/home/intern/SSL_Chest_Xray/datasets/vinbigdata.csv")

    df = create_chexpert_5_full_csv("/new_data/CheXpert", "/home/intern/SSL_Chest_Xray/datasets/chexpert_train_fixed.csv")
    df.to_csv("/home/intern/SSL_Chest_Xray/datasets/chexpert_5.csv")


