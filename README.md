Prerequisites: Python, Git
Python download: https://www.python.org/downloads/
Git download: https://git-scm.com/downloads

REPOSITORY INSTALLATION INSTRUCTIONS:
0. Open terminal and run the command "git clone https://github.com/SamBalakhanei/nlp-proj-code.git"
1. Run the command "cd nlp-proj-code"
2. Create a python virtual environment by running the command "python -m venv venv"
3. Activate the virtual environment by running the command "source venv/bin/activate" (UNIX) or "call venv/Scripts/activate" (Windows CMD)
4. Run the command "pip install -r requirements.txt"
5. Go to https://huggingface.co/settings/tokens and click "Create New Token" at the top right
6. Select "Read", give your token a name, and click "Create Token"
7. Copy the token
8. Create a file in nlp-proj-code called ".env" (/nlp-proj-code/.env)
9. In this file, write "HF_TOKEN=<your_token_here>" and replace "<your_token_here>" with the token copied in step 7
10. Save the .env file


RUN INSTRUCTIONS:

python3 benchmark.py \       
  --db ./nport.db \
  --questions "<PATH_TO_QUESTIONS_SHEET_FILE>" \
  --sheet <SHEET_NAME> \
  --question_col "<COLUMN_QUESTIONS_NAME>" \
  --row_start <ROW_NUMBER1> --row_end <ROW_NUMBER2> \
  --out <ROW_NUMBER1>_<ROW_NUMBER2>.jsonl \
  --include_sample \

(replace ROW_NUMBER with starting row and ending row from the google sheet (inclusive))

5. That should create a file called "prompts_<ROW_NUMBER1>_<ROW_NUMBER2>.jsonl" so go in there and make sure its the right questions and system prompt in there
6. Run this command to actually run the benchmark:

python3 benchmark.py \       
  --db ./nport.db \
  --questions "<PATH_TO_QUESTIONS_SHEET_FILE>" \
  --sheet <SHEET_NAME> \
  --question_col "<COLUMN_QUESTIONS_NAME>" \
  --row_start <ROW_NUMBER1> --row_end <ROW_NUMBER2> \
  --out <ROW_NUMBER1>_<ROW_NUMBER2>.jsonl \
  --include_samples \
  --run \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --max_tokens 256 \
  --temperature 0.0 \
  
  7. That should create the file completions_<ROW_NUMBER1>_<ROW_NUMBER2>.jsonl with the generated sql. Go to chatgpt and give it and tell it to get rid of the \n so u can run the query
