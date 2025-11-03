import os
import numpy as np

def generate_multiple_splits(all_txt_path, output_root, n_splits=20, split_ratio=0.5):

    with open(all_txt_path, 'r', encoding='utf-8') as f:
        all_data = [line.strip() for line in f if line.strip()]
    print(f"Successfully read raw data, total {len(all_data)} records")     


    os.makedirs(output_root, exist_ok=True)


    for split_idx in range(1, n_splits + 1):
        np.random.seed(split_idx)  
        shuffled_data = np.random.permutation(all_data)
        split_pos = int(len(shuffled_data) * split_ratio)
        train_data = shuffled_data[:split_pos]
        test_data = shuffled_data[split_pos:]

        split_dir = os.path.join(output_root, f"split_{split_idx:02d}")
        os.makedirs(split_dir, exist_ok=True)

        train_path = os.path.join(split_dir, "train.txt")
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_data))
        
        test_path = os.path.join(split_dir, "test.txt")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_data))




if __name__ == "__main__":
    ALL_TXT_PATH = "D:\pythoncollection\msr\split\\all.txt"  
    OUTPUT_ROOT = "multiple_splits"  


    generate_multiple_splits(
        all_txt_path=ALL_TXT_PATH,
        output_root=OUTPUT_ROOT,
        n_splits=20,  #Number of divisions
        split_ratio=0.5  
    )

