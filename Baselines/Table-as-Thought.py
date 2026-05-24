import json
from experiment import build_agent, load_json_if_exists, collect_score_template, collect_response, parse_response
import os

def collect_step_dimension_for_Table_as_Thought(step_num):
    step_config = json.load(open(r"jsons\steps_config_for_Direct_Score.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    Category_lst = "\n".join(json.load(open(r"jsons\category_lst.json", 'r', encoding='utf-8')))
    dimensions = ''
    for id, dim in enumerate(step_config['Step-rubrics'], start=1):
        dimensions += f"# Dimension-{id}: {dim['dimension_name']}:\n{dim['dimension_rubrics'].replace('{Category_lst}', Category_lst)}\n\n"

    prompt = "## Step: {step_name}\nStep requirements: {step_description}\n\n{dimensions}".format(
        step_name = step_config['Step-Name'],
        step_description = step_config['Step-Description'],
        dimensions = dimensions
    )
    return prompt


def Judge_Scoring(model_name, FS_num, response_content, temperature):
    Judge_LLM = build_agent(model_name = model_name, agent_name = 'Judge_LLM', temperature = temperature)
    
    prompts = json.load(open(r"jsons\Direct_Score_prompt.json", 'r', encoding='utf-8'))
    fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
    score_dimensions = '\n'.join([collect_step_dimension_for_Table_as_Thought(step) for step in step_num])
    output_template = collect_score_template(r"jsons\Table_Score_Output_Template_ch.json")

    user_prompt = prompts[1].format(future_scenario = fs, \
                                    response_content = collect_response(response_content), \
                                    score_dimensions = score_dimensions,
                                    output_template = output_template)

    Judge_LLM.add_system(prompts[0])
    Judge_LLM.add_user(user_prompt)
    response, Time = Judge_LLM.ask()

    return parse_response(model_name, response, Time)


def Table_as_Thought(Judge_model_name, ans_id, single_response, file_core_name, temperature):
    
    file_name = os.path.join("Results", "Raw_Results", file_core_name + "_" + Judge_model_name + ".json")
    token_file_name = os.path.join("Results", 'Latency_Results', "[TOKEN]" + file_core_name + "_" + Judge_model_name + ".json")
    time_file_name = os.path.join("Results", 'Latency_Results', "[TIME]" + file_core_name + "_" + Judge_model_name + ".json")
    
    final_score = load_json_if_exists(file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)

    Judge_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
        Judge_Scoring(Judge_model_name, int(ans_id.split("FS")[1]), single_response, temperature)


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


    token_consumption_lst[ans_id] = {
            'input_tokens': input_tokens, 
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }
    time_consumption_lst[ans_id] = Time

    final_score[ans_id] = Judge_result


    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=4)
    with open(token_file_name, "w", encoding="utf-8") as f:
        json.dump(token_consumption_lst, f, ensure_ascii=False, indent=4)
    with open(time_file_name, "w", encoding="utf-8") as f:
        json.dump(time_consumption_lst, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    # TaT
    step_num = [1, 2, 3, 4, 5, 6]

    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4-mini']
    model_name = 'gpt-5.4-mini'
    temperature = 0.2

    Pure_responses = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))
    for ans_id in Pure_responses:
        Table_as_Thought(model_name, ans_id, Pure_responses[ans_id], 'Table_as_Thought', temperature)