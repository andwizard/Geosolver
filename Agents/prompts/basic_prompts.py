BASIC_PROMPT = """
You are a GIS analyst. Your job is to generate a complete, runnable Python script 
for some spatial analysis task.

TASK:
{task_text}

REQUIRED DATA PATHS:
{file_list}

EXEPTED OUTPUT:
{expected_output}

OUTPUT_DIR:
{output_dir}

RULES:
1. Write a complete Python script in "script" that:
   - Loads data following the provided paths
   - Saves any output figures to {output_dir}/ with descriptive filenames
2. To fullfill the EXPECTED OUTPUT, your script should:
   - Print the final answer to the console (e.g. optimal locations, statistics, etc.)
   - Save any required figures to {output_dir}/ with descriptive filenames.
3. Please strictly use the given path to load and save data, do not make any modifications to the path.
"""