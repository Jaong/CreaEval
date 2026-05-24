"""
This file implements the ablation study of the memory component in Phase 1 (Evidence Extraction), referred to as AblationA.
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment import build_agent, collect_step_dimension_for_sot, parse_response, load_json_if_exists, \
                        collect_step_dimension_for_Judge, collect_score_template



def update_dim_name(sot_file_name):

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

    for ans_id, ans_content in final_sot.items():
        new_data[ans_id] = {}

        for step, step_content in ans_content.items():
            new_data[ans_id][step] = {}

            sot = step_content
            new_sot = {}

            dim_names = dimension_map[step]

            # ===== Step-1 / 3 / 4 =====
            if step in ['Step-1', 'Step-3', 'Step-4']:
                for item_id, item_content in sot.items():
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
                    if key in sot:
                        new_sot[dim_name] = sot[key]

            new_data[ans_id][step]["sot_evidence"] = new_sot
            new_data[ans_id][step]["task_status_memory"] = step_content["task_status_memory"]

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    return save_file





def check_sot(ans_id, sot_results):
    temp = 1
    for step in [1, 2, 3, 4, 5, 6]: # step_num:
        step_sot = sot_results['Step-' + str(step)]
        
        if list(step_sot.keys()) != ['sot_evidence', 'task_status_memory']:
            temp = 0
        
        if step in [1, 3, 4]:
            sot_evidence = step_sot['sot_evidence']
            if 'item_1' not in sot_evidence:
                temp = 0
            if 'dimension_1' not in sot_evidence['item_1']:
                temp = 0
        
        else:
            sot_evidence = step_sot['sot_evidence']
            if 'dimension_1' not in sot_evidence:
                temp = 0
    
    if temp:
        print(ans_id, ' OK!')
    else:
        print(ans_id, ' Error!')
    # return temp


def experiment_SoT(SoT_model_name, ans_id, Pure_responses, final_sot_file_name, temperature):
    
    def SoT_Extraction(model_name, step, FS_num, raw_text, pre_sot):


        SoT_LLM = build_agent(model_name = model_name, agent_name = 'SoT_LLM', temperature = temperature)
        prompts = json.load(open(r"jsons\SoT_LLM_prompt.json", 'r', encoding='utf-8'))
        fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
        score_dimensions_wo_rubrics = collect_step_dimension_for_sot(step)
        step_schema = json.load(open(r"jsons\step_schema_Ablation.json", 'r', encoding='utf-8'))['Step-' + str(step)]

        user_prompt = prompts[1].format(step_num = 'Step-' + str(step), \
                                        future_scenario = fs, \
                                        step_schema = step_schema, \
                                        score_dimensions_wo_rubrics = score_dimensions_wo_rubrics,
                                        raw_text = raw_text)

        
        SoT_LLM.add_system(prompts[0])
        SoT_LLM.add_user(user_prompt)
        response, Time = SoT_LLM.ask()
        return parse_response(model_name, response, Time)
        
        

    token_file_name = r"Results\Latency_Results\[TOKEN]AblationA_sot_results_" + SoT_model_name + ".json"
    time_file_name = r"Results\Latency_Results\[TIME]AblationA_sot_results_" + SoT_model_name + ".json"

    final_sot = load_json_if_exists(final_sot_file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)

    for step in step_num:
            
        step_answer = Pure_responses['Step-' + str(step)]

        SoT_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
            SoT_Extraction(SoT_model_name, step, int(ans_id.split("FS")[1]), step_answer, final_sot[ans_id])
        
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
                print(f"[Warning] JSON decode failed for {ans_id} Step-{step}, retry {retry_count}")

                if retry_count >= max_retry:
                    raise ValueError(f"SoT JSON decode failed after {max_retry} retries for {ans_id} Step-{step}")

                SoT_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
                    SoT_Extraction(SoT_model_name, step, int(ans_id.split("FS")[1]), step_answer, final_sot[ans_id])

        token_consumption_lst[ans_id]['Step-' + str(step)] = {
                'input_tokens': input_tokens, 
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'reasoning_tokens': reasoning_tokens
            }
        time_consumption_lst[ans_id]['Step-' + str(step)] = Time

        final_sot[ans_id]['Step-' + str(step)] = SoT_result

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
        

    file_name = os.path.join("Results", "Raw_Results", "SoT_Proposed_" + SoT_LLM_model_name + '_&_' + Judge_model_name + ".json")
    token_file_name = os.path.join("Results", 'Latency_Results', "[TOKEN]SoT_Proposed_" + SoT_LLM_model_name + '_&_' + Judge_model_name + ".json")
    time_file_name = os.path.join("Results", 'Latency_Results', "[TIME]SoT_Proposed_" + SoT_LLM_model_name + '_&_' + Judge_model_name + ".json")
    
    final_score = load_json_if_exists(file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)
    
    final_sot = json.load(open(sot_file_name, 'r', encoding='utf-8'))

    sot_output = final_sot[ans_id]
    sot_output = json.dumps(sot_output, ensure_ascii=False, indent=4)

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
    # Pure_responses = dict(list(Pure_responses.items())[4:5])
    step_num = [1, 2, 3, 4, 5, 6]

    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4-mini']
    SoT_LLM_model_name = 'qwen3.6-plus-2026-04-02'
    SoT_LLM_temperature = 0.2
    sot_save_file_name = r'Results\AblationA_sot_results_' + SoT_LLM_model_name + '.json'


    for ans_id in Pure_responses:
        experiment_SoT(SoT_LLM_model_name, ans_id, Pure_responses[ans_id], sot_save_file_name, SoT_LLM_temperature)

    sot_with_dim_name_file_name = update_dim_name(sot_save_file_name)

    # Step-2: Score
    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4']
    Judge_LLM_model_name = 'qwen3.6-plus-2026-04-02'
    Judge_LLM_temperature = 0.2


    for ans_id in Pure_responses:
        experiment_Judge(SoT_LLM_model_name, Judge_LLM_model_name, ans_id, Judge_LLM_temperature, sot_with_dim_name_file_name)
    