import ast
import re

def get_error_pass(raw):
    result = []
    for span in re.findall(r'\{[^{}]*\}', raw):
        try:
            result.append(ast.literal_eval(span))
        except:
            pass
    return result

def checker(df, columns=None, get_response_from_different_seed=True):
    flag = False
    correctable_baselines = []
    incorrectable_baselines = []
    baselines = [(col.split("_")[0], col.split("_")[1], col.split("_")[2]) for col in df.columns if "generated" in col]
    for baseline in baselines:
        model_wise_error_cnt = 0
        model_name, method, seed = baseline
        
        if not f'{model_name}_{method}_{seed}_property' in df:
            for i, row in df.iterrows():
                try:
                    if "multi" in method:
                        dictionaries = get_error_pass(row[f'{model_name}_{method}_{seed}_generated_'].replace("\\n", "").replace("']", "}']").lower())
                        if len(dictionaries) == 0:
                            raise Exception("")
                        for d in dictionaries:
                            if columns:
                                for col in columns:
                                    d[col]
                    else:
                        dictionary = get_error_pass(row[f'{model_name}_{method}_{seed}_generated_'].replace("\\n", "").replace("']", "}']").lower())[-1]
                        if columns:
                            for col in columns:
                                dictionary[col]
                except:
                    # try to get response from different seed
                    success_to_get_response = False
                    if get_response_from_different_seed:
                        for different_seed in range(3):
                            try:
                                get_error_pass(row[f'{model_name}_{method}_{different_seed}_generated_'].replace("\\n", "").replace("']", "}']").lower())[-1]
                                df.at[i, f'{model_name}_{method}_{seed}_generated_'] = row[f'{model_name}_{method}_{different_seed}_generated_'].replace("\\n", "").replace("']", "}']").lower()
                                
                                success_to_get_response = True
                                break
                            except:
                                pass
                    if not success_to_get_response:
                        print(baseline)
                        model_wise_error_cnt += 1
                        print(row[f'{model_name}_{method}_{seed}_generated_'].replace("\"", "\"\""))
                        print()
            if model_wise_error_cnt:
                print(model_wise_error_cnt)
                if model_wise_error_cnt < 20:
                    correctable_baselines.append(model_name)
                else:
                    incorrectable_baselines.append(model_name)
    if len(correctable_baselines) != 0:
        print(correctable_baselines)
        raise Exception("")

    return df, [baseline for baseline in baselines if baseline[0] not in incorrectable_baselines]
        
            
    