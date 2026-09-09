with open(r'D:\Nexsure\backend\app\core\explain.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_func = '''def explain_local_prediction(
    model: Any,
    pipeline: Any,
    input_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate local SHAP explanations and human-readable text for a prediction."""
    transformed = pipeline.transform(input_df)
    feature_names = _get_feature_names(pipeline)
    explainer = _get_shap_explainer(model, transformed)
    shap_values = explainer(transformed)'''

new_func = '''def explain_local_prediction(
    model: Any,
    pipeline: Any = None,
    input_df: pd.DataFrame = None,
    explainer: Any = None,
    transformed_array: Any = None,
    feature_names: List[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate local SHAP explanations and human-readable text for a prediction."""
    if transformed_array is not None:
        transformed = transformed_array
    elif pipeline is not None and input_df is not None:
        transformed = pipeline.transform(input_df)
    else:
        raise ValueError("Must provide either transformed_array or both pipeline and input_df")

    if feature_names is None:
        if pipeline is not None:
            feature_names = _get_feature_names(pipeline)
        else:
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    if explainer is None:
        explainer = _get_shap_explainer(model, transformed)

    shap_values = explainer(transformed)'''

if old_func in code:
    code = code.replace(old_func, new_func)
    with open(r'D:\Nexsure\backend\app\core\explain.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Updated explain.py successfully')
else:
    print('Could not find old_func')
