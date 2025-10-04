# main.py
import subprocess
import sys

def run_script(script_name):
    """
    Runs a Python script using subprocess and checks for errors.
    """
    print(f"--- Running {script_name} ---")
    try:
        # Using sys.executable ensures the correct Python interpreter is used
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"--- Successfully completed {script_name} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- An error occurred while running {script_name} ---")
        print("Stderr:")
        print(e.stderr)
        print("Stdout:")
        print(e.stdout)
        sys.exit(1)
    except FileNotFoundError:
        print(f"--- Error: {script_name} not found. Please ensure it's in the same directory. ---")
        sys.exit(1)

def main():
    """
    Orchestrates the entire chat intent classification pipeline.
    """
    # Step 1: Ingest and clean the raw data
    run_script("1) ingest_clean.py")

    # Step 2: Load the cleaned data into the MySQL database
    run_script("2) load_to_mysql.py")

    # Step 3: (Optional) Insert synthetic data for augmentation
    # Note: This step assumes 'augmented_samples_full.csv' is available.
    # It can be skipped if you are not using synthetic data.

    #run_script("(opt) insert_synthetic_to_db.py")

    # Step 4: Prepare data for training (splits and tokenizes)
    run_script("3) prepare_data.py")

    # Step 5: Fine-tune the BERT model
    run_script("4) finetune_bert.py")

    # Step 6: Evaluate the model and export metrics
    # Note: I'm excluding evaluator.py as per your request, so we
    # rely on finetune_bert.py for basic validation metrics
    # and would need a separate script for detailed test set metrics.
    # The push_model.py script will upload the `test_metrics_bert.csv` if it exists.


    run_script("5) validation_dashboard.py")

    run_script("6.1) error_analysis.py")

    run_script("6.2) analysis_export_top_confusions.py")
    
    run_script("7) export_bert_metrics.py")

    # Step 7: Push the model and artifacts to Hugging Face Hub
    run_script("8) push_model.py")

    print("\n--- Pipeline execution complete! and finally run streamlit run app.py if you would like to ---")

if __name__ == "__main__":
    main()