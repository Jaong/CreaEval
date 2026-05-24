"""
This script is used to run the automated scoring process of CreaEval, which consists of two stages:
SoT Evidence Extraction and Evidence-based Judging.
"""

import json
from agents import Gemini_Agent, GPT_Agent, Qwen_Agent, Deepseek_Agent
import os

responses_lst_A = [f"A{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
responses_lst_B = [f"B{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
total_responses_lst = responses_lst_A + responses_lst_B

def parse_response(model_name, response, Time):
    if 'gpt' in model_name or 'qwen' in model_name:
        return (
            response.output_text,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens,
            response.usage.output_tokens_details.reasoning_tokens,
            Time
        )

    elif 'gemini' in model_name:
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            response.usage.completion_tokens_details.reasoning_tokens,
            Time
        )
    
    elif 'deepseek' in model_name:
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            0,
            Time
        )


def build_agent(model_name, agent_name, temperature):
    if 'gpt' in model_name:
        return GPT_Agent(
            model_name=model_name,
            agent_name=agent_name,
            temperature=temperature
        )
    
    elif 'gemini' in model_name:
        return Gemini_Agent(
            model_name=model_name,
            agent_name=agent_name,
            temperature=temperature
        )
    
    elif 'qwen' in model_name:
        return Qwen_Agent(
            model_name=model_name,
            agent_name=agent_name,
            temperature=temperature
        )
    
    elif 'deepseek' in model_name:
        return Deepseek_Agent(
            model_name=model_name,
            agent_name=agent_name,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {ans_id: {} for ans_id in total_responses_lst}


def collect_response(Pure_response):
    '''
    Convert each response (Pure_response format) into prompt format.
    '''
    response = '## {step} Response Content\n{content}\n'
    return '\n'.join([response.format(step = 'Step-' + str(step), content = Pure_response['Step-' + str(step)]) for step in range(1, 7)])



def collect_score_template(file_name):
    '''
    Return the scoring outputs for all steps.
    '''
    output_template = json.load(open(file_name, 'r', encoding='utf-8'))
    
    result = []
    for i, step in enumerate(range(1, 7)):
        content = output_template['Step-' + str(step)][1:-1].strip()
        
        if i != len(range(1, 7)) - 1:
            content += ','
        
        result.append(content)
    
    return '{\n' + '\n'.join(result) + '\n}'




def collect_step_dimension_for_sot(step_num):
    '''
    Construct prompts that describe each evaluation dimension for SoT_LLM.
    '''
    step_config = json.load(open(r"jsons\steps_config_for_SoT_LLM.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    Category_lst = "\n".join(json.load(open(r"jsons\category_lst.json", 'r', encoding='utf-8')))
    dimensions = ''
    for id, dim in enumerate(step_config['Step-rubrics'], start=1):
        dimensions += f"# Dimension-{id}: {dim['dimension_name']} \n{dim['dimension_rubrics'].replace('{Category_lst}', Category_lst)}\n\n"

    prompt = "## {step_name}\nStep Requirements: {step_description}\n\n{dimensions}".format(
        step_name = step_config['Step-Name'],
        step_description = step_config['Step-Description'],
        dimensions = dimensions
    )
    return prompt.strip()


def collect_step_dimension_for_Judge(step_num):
    '''
    Construct dimension-description prompts for Judge-LLM, excluding the Category_list.
    '''
    step_config = json.load(open(r"jsons\steps_config_for_Judge_LLM.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    dimensions = ''
    for id, dim in enumerate(step_config['Step-rubrics'], start=1):
        dimensions += f"# Dimension-{id}: {dim['dimension_name']} \n{dim['dimension_rubrics']}\n\n"

    prompt = "## {step_name}\nStep Requirements: {step_description}\n\n{dimensions}\n\n".format(
        step_name = step_config['Step-Name'],
        step_description = step_config['Step-Description'],
        dimensions = dimensions
    )
    return prompt.strip()



def experiment_SoT(SoT_model_name, ans_id, Pure_responses, final_sot_file_name, temperature):
    '''
    Step-1: Extract SoT information for each step and save to final_sot
    '''

    def SoT_Extraction(model_name, step, FS_num, raw_text, pre_sot):
        dependency_step = {2: [1], 3:[1, 2], 4:[1, 2, 3], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4, 5]}
        SoT_LLM = build_agent(model_name = model_name, agent_name = 'SoT_LLM', temperature = temperature)

        prompts = json.load(open(r"jsons\SoT_LLM_prompt.json", 'r', encoding='utf-8'))
        fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
        score_dimensions_wo_rubrics = collect_step_dimension_for_sot(step)
        step_schema = json.load(open(r"jsons\step_schema.json", 'r', encoding='utf-8'))['Step-' + str(step)]


        user_prompt = prompts[1].format(step_num = 'Step-' + str(step), \
                                        future_scenario = fs, \
                                        step_schema = step_schema, \
                                        score_dimensions_wo_rubrics = score_dimensions_wo_rubrics,
                                        raw_text = raw_text)


        # SoT_Memory_Retrieval
        if step in dependency_step:
            
            user_prompt += '\n\n'
            user_prompt += '\n\n'.join([
                f"[Memory from Step-{step_p}]\n{pre_sot['Step-' + str(step_p)]['task_status_memory']}"
                for step_p in dependency_step[step]
            ])

        SoT_LLM.add_system(prompts[0])
        SoT_LLM.add_user(user_prompt)
        response, Time = SoT_LLM.ask()

        return parse_response(model_name, response, Time)
        

    token_file_name = r"Results\Latency_Results\[TOKEN]sot_results_" + SoT_model_name + ".json"
    time_file_name = r"Results\Latency_Results\[TIME]sot_results_" + SoT_model_name + ".json"

    final_sot = load_json_if_exists(final_sot_file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)

    for step in step_num:
            
        step_answer = Pure_responses['Step-' + str(step)]

        SoT_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
            SoT_Extraction(SoT_model_name, step, int(ans_id.split("FS")[1]), step_answer, final_sot[ans_id])
        
        if SoT_result.startswith("```json") and SoT_result.endswith("```"):
            SoT_result = SoT_result.strip("```json").strip("```")


        try:
            SoT_result = json.loads(SoT_result)
        except json.JSONDecodeError:
            print(f"[Warning] JSON decode failed for {ans_id} Step-{step}")
            SoT_result, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = \
                SoT_Extraction(SoT_model_name, step, int(ans_id.split("FS")[1]), step_answer, final_sot[ans_id])
            if SoT_result.startswith("```json") and SoT_result.endswith("```"):
                SoT_result = SoT_result.strip("```json").strip("```")
            SoT_result = json.loads(SoT_result)

        # print(SoT_result['sot_evidence'])
        token_consumption_lst[ans_id]['Step-' + str(step)] = {
                'input_tokens': input_tokens, 
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'reasoning_tokens': reasoning_tokens
            }
        time_consumption_lst[ans_id]['Step-' + str(step)] = Time

        final_sot[ans_id]['Step-' + str(step)] = SoT_result
        
        print(f"{ans_id} Step-{step} SoT results have been generated.")

    with open(final_sot_file_name, "w", encoding="utf-8") as f:
        json.dump(final_sot, f, ensure_ascii=False, indent=4)
    with open(token_file_name, "w", encoding="utf-8") as f:
        json.dump(token_consumption_lst, f, ensure_ascii=False, indent=4)
    with open(time_file_name, "w", encoding="utf-8") as f:
        json.dump(time_consumption_lst, f, ensure_ascii=False, indent=4)

    print('=========================', ans_id, 'SoT Completed! =========================\n')


def experiment_Judge(SoT_LLM_model_name, Judge_model_name, ans_id, temperature, sot_file_name):
    '''
    Step-2: Score each answer based on the extracted SoT information and save to final_score
    '''

    def collect_sot_for_judge(sot_result):
        '''
        Concatenate the SoT results used for scoring for each response.
        Input is the SoT of a single response, i.e., sot_results[ans_id].
        '''
        result = {}

        for step_name, step_content in sot_result.items():
            assert 'sot_evidence' in step_content, f"{ans_id} Step {step_name} is missing the 'sot_evidence' field!"
            result[step_name] = step_content["sot_evidence"]

        return result


    def Judge_Scoring(model_name, FS_num, sot_output, temperature):
        '''
        Use the Judge_LLM to score each response and output the final scoring results.
        '''
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
  
    sot_output = collect_sot_for_judge(final_sot[ans_id])
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

    print('=========================', Judge_model_name, '  ', ans_id, ' Completed! =========================')



def update_dim_name(sot_file_name):
    '''
    Replace "dimension_i" in the SoT results with the corresponding <dimension_name>.
    '''
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

            sot = step_content["sot_evidence"]
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

    # return new_data
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print('Saved to ', save_file)
    return save_file


def check_sot(sot_results):
    '''
    Check whether the SoT output for the specified response conforms to the required template.
    Return True or False.
    '''
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
        print("No issues found during the check!")
    return temp



def check_wo_evidence(sot_file):
    """
    Check whether each step of every response contains the "sot_evidence" field.
    """
    temp = 1
    step_num = [1, 2, 3, 4, 5, 6]
    final_sot = json.load(open(sot_file, 'r', encoding='utf-8'))
    for ans_id, ans_content in final_sot.items():

        for step in step_num:

            step_sot = ans_content['Step-' + str(step)]
            if 'sot_evidence' not in step_sot:
                temp = 0
                print(ans_id, f"Step-{step} is missing the 'sot_evidence' field")
    
    if temp: print("All SoT fields contain the 'sot_evidence' field.")



if __name__ == "__main__":
    '''
    Pure_responses = {
            "A01_FS1": {'Step-1': ***, 'Step-2': ***, ...},
            ...
        }
    '''
    
    # Step-1: Extract SoT information for each step and save to final_sot
    Pure_responses = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))
    # Pure_responses = dict(list(Pure_responses.items())[155:])

    step_num = [1, 2, 3, 4, 5, 6]
    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4']
    SoT_LLM_model_name = 'gpt-5.4'
    SoT_LLM_temperature = 0.2
    sot_save_file_name = r'Results\sot_results_' + SoT_LLM_model_name + '.json'


    # Step-1: SoT Extraction
    for ans_id in Pure_responses:
    
        while True:            
            experiment_SoT(SoT_LLM_model_name, ans_id, Pure_responses[ans_id], sot_save_file_name, SoT_LLM_temperature)
            sot_results = json.load(open(sot_save_file_name, 'r', encoding='utf-8'))[ans_id]
            if check_sot(sot_results):
                break
            else: print(ans_id, "Format error!")

    sot_with_dim_name_file_name = update_dim_name(sot_save_file_name)
    check_wo_evidence(sot_with_dim_name_file_name)


    # Step-2: Score
    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4']
    Judge_LLM_model_name = 'gpt-5.4'
    Judge_LLM_temperature = 0.2

    for ans_id in Pure_responses:
        experiment_Judge(SoT_LLM_model_name, Judge_LLM_model_name, ans_id, Judge_LLM_temperature, sot_with_dim_name_file_name)
    