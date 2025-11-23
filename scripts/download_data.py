# """Download dataset"""

# def download_dataset():
#     """Download and extract the dataset"""
#     print("Downloading dataset...")
#     # TODO: Implement download logic
#     print("Dataset downloaded to data/raw/")

# if __name__ == "__main__":
#     download_dataset()


"""Download dataset from Kaggle"""
import os
import shutil
import pandas as pd

def download_dataset():
    """Download news articles dataset from Kaggle"""
    
    print("=" * 60)
    print("Downloading News Articles Dataset from Kaggle...")
    print("=" * 60)
    
    try:
        import kagglehub
        
        # Download latest version
        print("\n📥 Downloading dataset...")
        path = kagglehub.dataset_download("asad1m9a9h6mood/news-articles")
        print(f"✓ Dataset downloaded to: {path}")
        
        # Create data/raw directory if it doesn't exist
        raw_data_dir = "data/raw"
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Copy files to data/raw/
        print(f"\n📁 Moving files to {raw_data_dir}/...")
        for file in os.listdir(path):
            src = os.path.join(path, file)
            dst = os.path.join(raw_data_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"✓ Copied: {file}")
        
        # Display dataset information
        print("\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        
        # Try to load and show basic info
        csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        
        if csv_files:
            print(f"\n✓ Found {len(csv_files)} CSV file(s)")
            
            for csv_file in csv_files:
                csv_path = os.path.join(raw_data_dir, csv_file)
                df = pd.read_csv(csv_path)
                
                print(f"\n📄 File: {csv_file}")
                print(f"   • Total articles: {len(df)}")
                print(f"   • Columns: {list(df.columns)}")
                print(f"   • File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")
                
                # Show sample
                print(f"\n   Sample data (first 3 rows):")
                print(df.head(3))
                
                # Check for missing values
                print(f"\n   Missing values:")
                print(df.isnull().sum())
        
        print("\n" + "=" * 60)
        print("✅ Dataset download complete!")
        print("=" * 60)
        print(f"\nDataset location: {os.path.abspath(raw_data_dir)}")
        print("\nNext steps:")
        print("1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
        print("2. Preprocess data: Implement src/preprocessing modules")
        
    except ImportError:
        print("❌ Error: kagglehub not installed")
        print("Install it with: pip install kagglehub")
        return False
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_dataset()