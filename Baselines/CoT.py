import json
from experiment import build_agent, load_json_if_exists, parse_response
import os

def collect_step_dimension_for_Direct_Score(step_num):
    step_config = json.load(open(r"jsons\steps_config_for_Direct_Score.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    Category_lst = "\n".join(json.load(open(r"jsons\category_lst.json", 'r', encoding='utf-8')))
    dimensions = ''
    for id, dim in enumerate(step_config['Step-rubrics'], start=1):
        dimensions += f"# Dimension-{id}: {dim['dimension_name']}:\n{dim['dimension_rubrics'].replace('{Category_lst}', Category_lst)}\n\n"

    prompt = "## Step: {step_name}\nStep requirement: {step_description}\n\n{dimensions}".format(
        step_name = step_config['Step-Name'],
        step_description = step_config['Step-Description'],
        dimensions = dimensions
    )
    return prompt


def Judge_Scoring(model_name, FS_num, step_num, response_content, Judge_LLM):
    prompts = json.load(open(r"jsons\CoT_prompt.json", 'r', encoding='utf-8'))
    fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
    score_dimensions = collect_step_dimension_for_Direct_Score(step_num)
    output_template = json.load(open(r"jsons\Score_Output_Template_ch.json", 'r', encoding='utf-8'))['Step-' + str(step_num)]

    user_prompt = prompts[1].format(future_scenario = fs, \
                                    response_content = "Step-" + str(step_num) + ' Response Content: \n' + response_content['Step-' + str(step_num)], \
                                    score_dimensions = score_dimensions,
                                    output_template = output_template,\
                                    step_num = step_num)
    
    if step_num == 1:
        Judge_LLM.add_system(prompts[0])

    Judge_LLM.add_user(user_prompt)
    Response, Time = Judge_LLM.ask()

    response, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = parse_response(model_name, Response, Time)

    Judge_LLM.add_assistant(response)

    return response, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time



def add_step_head(lst, ans_id):
    new_lst = lst
    for step in STEP_num:
        new_lst[ans_id]["Step-" + str(step)] = {}
    return new_lst



def CoT_Score(Judge_model_name, ans_id, single_response, file_core_name, temperature):
    
    file_name = os.path.join("Results", "Raw_Results", file_core_name + "_" + Judge_model_name + ".json")
    token_file_name = os.path.join("Results", 'Latency_Results', "[TOKEN]" + file_core_name + "_" + Judge_model_name + ".json")
    time_file_name = os.path.join("Results", 'Latency_Results', "[TIME]" + file_core_name + "_" + Judge_model_name + ".json")
    
    final_score = load_json_if_exists(file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)

    Judge_LLM = build_agent(model_name = model_name, agent_name = 'Judge_LLM', temperature = temperature)
    final_result = {}

    token_consumption_lst = add_step_head(token_consumption_lst, ans_id)
    time_consumption_lst = add_step_head(time_consumption_lst, ans_id)


    for step_num in STEP_num:

        Judge_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
            Judge_Scoring(Judge_model_name, int(ans_id.split("FS")[1]), step_num, single_response, Judge_LLM)
        
        if Judge_result.startswith("```json") and Judge_result.endswith("```"):
            Judge_result = Judge_result.strip("```json").strip("```")
        
        
        max_retry = 5
        retry_count = 0

        while True:
            try:
                Judge_result = json.loads(Judge_result)
                break

            except json.JSONDecodeError:
                retry_count += 1
                print(f"[Warning] JSON decode failed for {ans_id}, retry {retry_count}")

                if retry_count >= max_retry:
                    raise ValueError(f"SoT JSON decode failed after {max_retry} retries for {ans_id}")

                Judge_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
                    Judge_Scoring(Judge_model_name, int(ans_id.split("FS")[1]), single_response, temperature)

        token_consumption_lst[ans_id]["Step-" + str(step_num)] = {
                'input_tokens': input_tokens, 
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'reasoning_tokens': reasoning_tokens
            }
        time_consumption_lst[ans_id]["Step-" + str(step_num)] = Time
        final_result.update(Judge_result)

    final_score[ans_id] = final_result


    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=4)
    with open(token_file_name, "w", encoding="utf-8") as f:
        json.dump(token_consumption_lst, f, ensure_ascii=False, indent=4)
    with open(time_file_name, "w", encoding="utf-8") as f:
        json.dump(time_consumption_lst, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    '''
    Pure_responses = {
            "A01_FS1": {'Step-1': ***, 'Step-2': ***, ...},
            ...
        }
    '''
    STEP_num = [1, 2, 3, 4, 5, 6]
    
    # CoT
    Pure_responses = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))
    # Pure_responses = dict(list(Pure_responses.items())[138:])

    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4-mini']
    model_name = 'gemini-3.1-pro-preview'
    temperature = 0.2
            
    for ans_id in Pure_responses:
        CoT_Score(model_name, ans_id, Pure_responses[ans_id], 'CoT', temperature)