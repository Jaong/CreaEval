"""
This file implements the step-by-step evidence extraction ablation experiment in Phase 1 (Evidence Extraction), referred to as AblationB.
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment import build_agent, collect_step_dimension_for_sot, parse_response, load_json_if_exists, \
                        collect_step_dimension_for_Judge, collect_score_template




def collect_step_responses_for_sot(raw_response):
    prompt = ''

    for step, content in raw_response.items():
        step_title = '### ' + step + ' Response Content\n'
        prompt += step_title
        prompt += content
        prompt += '\n\n'
    return prompt.strip()





def update_dim_name_for_AblationB(sot_file_name, ans_id):
    final_sot = json.load(open(sot_file_name, 'r', encoding='utf-8'))
    save_file = sot_file_name.replace('.json', '_with_dim_name.json')
    new_data = {}

    dimension_map = {
        "Step-1": ["Fluency", "Flexibility", "Elaboration", "Originality"],
        "Step-2": ["Condition Phrase", "Stem & KVP", "Purpose", "FS Parameters", "Focus", "Adequacy"],
        "Step-3": ["Fluency", "Flexibility", "Elaboration", "Originality"],
        "Step-4": ["Correctly Written", "Relevance"],
        "Step-5": ["Correctly Used"],
        "Step-6": ["Relevance", "Effectiveness", "Criteria", "Impact", "Humaness", "Development"]
    }
    ans_content = final_sot[ans_id]

    for step, step_content in ans_content.items():
        new_data[step] = {}

        # sot = step_content["sot_evidence"]
        new_sot = {}

        dim_names = dimension_map[step]

        # ===== Step-1 / 3 / 4 =====
        if step in ['Step-1', 'Step-3', 'Step-4']:
            for item_id, item_content in step_content.items():
                new_item = {}

                for i, dim_name in enumerate(dim_names, start=1):
                    key = f"dimension_{i}"
                    if key in item_content:
                        new_item[dim_name] = item_content[key]

                new_sot[item_id] = new_item

        # ===== Step-2 / 5 / 6 =====
        else:
            for i, dim_name in enumerate(dim_names, start=1):
                key = f"dimension_{i}"
                if key in step_content:
                    new_sot[dim_name] = step_content[key]

        new_data[step] = new_sot

    final_sot[ans_id] = new_data

    # return new_data
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(final_sot, f, ensure_ascii=False, indent=4)
    return save_file




def experiment_SoT(SoT_model_name, ans_id, Pure_responses, final_sot_file_name, temperature):
    
    def SoT_Extraction_one_pass(model_name, FS_num, raw_text):

        SoT_LLM = build_agent(model_name = model_name, agent_name = 'SoT_LLM', temperature = temperature)

        prompts = json.load(open(r"jsons\SoT_AblationB_prompt.json", 'r', encoding='utf-8'))
        fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
        score_dimensions_wo_rubrics = '\n\n\n'.join([collect_step_dimension_for_sot(step) for step in step_num])
        step_schema = json.load(open(r"jsons\step_schema_Ablation.json", 'r', encoding='utf-8'))

        user_prompt = prompts[1].format(future_scenario = fs, \
                                        step_schema = step_schema, \
                                        score_dimensions_wo_rubrics = score_dimensions_wo_rubrics,
                                        raw_text = collect_step_responses_for_sot(raw_text))

        
        SoT_LLM.add_system(prompts[0])
        SoT_LLM.add_user(user_prompt)
        response, Time = SoT_LLM.ask()
        return parse_response(model_name, response, Time)

            
    token_file_name = r"Results\Latency_Results\[TOKEN]AblationB_sot_results_" + SoT_model_name + ".json"
    time_file_name = r"Results\Latency_Results\[TIME]AblationB_sot_results_" + SoT_model_name + ".json"

    final_sot = load_json_if_exists(final_sot_file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)

            
    SoT_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
        SoT_Extraction_one_pass(SoT_model_name, int(ans_id.split("FS")[1]), Pure_responses)
        
    if SoT_result.startswith("```json") and SoT_result.endswith("```"):
        SoT_result = SoT_result.strip("```json").strip("```")


    max_retry = 5
    retry_count = 0

    while True:
        try:
            SoT_result = json.loads(SoT_result)
            break

        except json.JSONDecodeError:
            retry_count += 1
            print(f"[Warning] JSON decode failed for {ans_id}, retry {retry_count}")

            if retry_count >= max_retry:
                raise ValueError(f"SoT JSON decode failed after {max_retry} retries for {ans_id}")

            SoT_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
                SoT_Extraction_one_pass(SoT_model_name, int(ans_id.split("FS")[1]), Pure_responses)

    token_consumption_lst[ans_id] = {
            'input_tokens': input_tokens, 
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }
    time_consumption_lst[ans_id] = Time

    final_sot[ans_id] = SoT_result

    with open(final_sot_file_name, "w", encoding="utf-8") as f:
        json.dump(final_sot, f, ensure_ascii=False, indent=4)
    with open(token_file_name, "w", encoding="utf-8") as f:
        json.dump(token_consumption_lst, f, ensure_ascii=False, indent=4)
    with open(time_file_name, "w", encoding="utf-8") as f:
        json.dump(time_consumption_lst, f, ensure_ascii=False, indent=4)





def experiment_Judge(SoT_LLM_model_name, Judge_model_name, ans_id, temperature, sot_file_name):
    '''
    Step-2: Score each answer based on the extracted SoT information and save to final_score
    '''

    def Judge_Scoring(model_name, FS_num, sot_output, temperature):
        Judge_LLM = build_agent(model_name = model_name, agent_name = 'Judge_LLM', temperature = temperature)
        
        prompts = json.load(open(r"jsons\Judge_LLM_prompt.json", 'r', encoding='utf-8'))
        fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
        score_dimensions = '\n\n\n'.join([collect_step_dimension_for_Judge(step) for step in range(1, 7)])
        output_template = collect_score_template(r"jsons\Score_Output_Template_ch.json")

        user_prompt = prompts[1].format(future_scenario = fs, \
                                        sot_output = sot_output, \
                                        score_dimensions = score_dimensions,
                                        output_template = output_template)

        Judge_LLM.add_system(prompts[0])
        Judge_LLM.add_user(user_prompt)
        response, Time = Judge_LLM.ask()

        return parse_response(model_name, response, Time)
        

    file_name = os.path.join("Results", "Raw_Results", "AblationB_" + SoT_LLM_model_name + '_&_' + Judge_model_name + ".json")
    token_file_name = os.path.join("Results", 'Latency_Results', "[TOKEN]AblationB_" + SoT_LLM_model_name + '_&_' + Judge_model_name + ".json")
    time_file_name = os.path.join("Results", 'Latency_Results', "[TIME]AblationB_" + SoT_LLM_model_name + '_&_' + Judge_model_name + ".json")
    
    final_score = load_json_if_exists(file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)
    
    final_sot = json.load(open(sot_file_name, 'r', encoding='utf-8'))
    
    sot_output = json.dumps(final_sot, ensure_ascii=False, indent=4)

    Judge_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
        Judge_Scoring(Judge_model_name, int(ans_id.split("FS")[1]), sot_output, temperature)
    

    token_consumption_lst[ans_id] = {
            'input_tokens': input_tokens, 
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }
    time_consumption_lst[ans_id] = Time

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
                Judge_Scoring(Judge_model_name, int(ans_id.split("FS")[1]), sot_output, temperature)

    final_score[ans_id] = Judge_result

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=4)
    with open(token_file_name, "w", encoding="utf-8") as f:
        json.dump(token_consumption_lst, f, ensure_ascii=False, indent=4)
    with open(time_file_name, "w", encoding="utf-8") as f:
        json.dump(time_consumption_lst, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    Pure_responses = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))
    # Pure_responses = dict(list(Pure_responses.items())[190:])
    step_num = [1, 2, 3, 4, 5, 6]

    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4']
    SoT_LLM_model_name = 'qwen3.6-plus-2026-04-02'
    SoT_LLM_temperature = 0.2
    sot_save_file_name = r'Results\AblationB_sot_results_' + SoT_LLM_model_name + '.json'


    for ans_id in Pure_responses:   

        print('=====================', ans_id, '=====================')        
        
        experiment_SoT(SoT_LLM_model_name, ans_id, Pure_responses[ans_id], sot_save_file_name, SoT_LLM_temperature)
        sot_with_dim_name_file_name = update_dim_name_for_AblationB(sot_save_file_name, ans_id)


        Judge_LLM_temperature = 0.2

        for Judge_LLM_model_name in ['qwen3.6-plus-2026-04-02']: # , 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4']:
            experiment_Judge(SoT_LLM_model_name, Judge_LLM_model_name, ans_id, Judge_LLM_temperature, sot_with_dim_name_file_name)