with open(r'D:\Nexsure\backend\app\core\explain.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''def explain_local_prediction(
    model: Any,
    pipeline: Any = None,
    input_df: pd.DataFrame = None,
    explainer: Any = None,
    transformed_array: Any = None,
    feature_names: List[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:'''

replacement = '''def explain_local_prediction(
    model: Any,
    pipeline: Any = None,
    input_df: pd.DataFrame = None,
    explainer: Any = None,
    transformed_input: Any = None,
    transformed_array: Any = None,
    feature_names: List[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    if transformed_input is not None:
        transformed = transformed_input
    elif transformed_array is not None:
        transformed = transformed_array'''

if target in code:
    code = code.replace(target, replacement)
    with open(r'D:\Nexsure\backend\app\core\explain.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Successfully updated explain.py with transformed_input')
else:
    print('target not found')
