import json
from experiment import build_agent, load_json_if_exists, parse_response
import os

Score_LLM_refine_prompt = '请严格基于维度说明和作答内容，重新检查你之前的评分结果，如发现不一致或无依据的评分，请进行修正，并输出最终修订后的 JSON 结果。'
Judge_LLM_refine_prompt = '请重新审查你刚刚的裁决结果，判断所选择的评分是否最符合维度说明与作答内容的证据支持。如存在更合理的候选评分或明显不一致之处，请进行修正，并输出最终的裁决结果，只需输出你选择的评分序号 [1, 2, 3]。'


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



def ToT_Step_Score(model_name, FS_num, step_num, response_content, Model_LST):
    Score_prompts = json.load(open(r"jsons\CoT_prompt.json", 'r', encoding='utf-8'))
    Judge_prompts = json.load(open(r"jsons\ToT_Judge_prompt.json", 'r', encoding='utf-8'))
    fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
    score_dimensions = collect_step_dimension_for_Direct_Score(step_num)
    output_template = json.load(open(r"jsons\Score_Output_Template_ch.json", 'r', encoding='utf-8'))['Step-' + str(step_num)]

    time_lst, token_lst = [], []
    Score_lst = []
    
    Score_user_prompt = Score_prompts[1].format(future_scenario = fs, \
                                    response_content = "Step-" + str(step_num) + ' Response Content: \n' + response_content['Step-' + str(step_num)], \
                                    score_dimensions = score_dimensions, \
                                    output_template = output_template,\
                                    step_num = step_num)
    Judge_LLM = Model_LST[-1]
    if step_num == 1:
        # Score_LLM
        for model in Model_LST[:-1]:
            model.add_system(Score_prompts[0])
        
        # Judge_LLM
        Judge_LLM.add_system(Judge_prompts[0])

    for Score_LLM in Model_LST[:-1]:
        Score_LLM.add_user(Score_user_prompt)
        
        Response, Time = Score_LLM.ask()

        response, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = parse_response(model_name, Response, Time)

        while True:
            try:
                if response.startswith("```json") and response.endswith("```"):
                    response = response.strip("```json").strip("```")
                response = response.replace("'", '"')
                response = json.loads(response)
                break
            except json.JSONDecodeError:
                print(Score_LLM.Agent_name, 'json ERROR, RETRY!')
                Response, Time = Score_LLM.ask()
                response, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = parse_response(model_name, Response, Time)

        Score_LLM.add_assistant(response)

        time_lst.append(Time)
        token_lst.append({
                'input_tokens': input_tokens, 
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'reasoning_tokens': reasoning_tokens
            })
        
        # self-refine
        Score_LLM.add_user(Score_LLM_refine_prompt)

        refine_Response, refine_Time = Score_LLM.ask()

        refine_response, refine_input_tokens, refine_output_tokens, refine_total_tokens, \
            refine_reasoning_tokens, refine_Time = parse_response(model_name, refine_Response, refine_Time)

        while True:
            try:
                if refine_response.startswith("```json") and refine_response.endswith("```"):
                    refine_response = refine_response.strip("```json").strip("```")
                refine_response = refine_response.replace("'", '"')
                refine_response = json.loads(refine_response)
                break
            except json.JSONDecodeError:
                print(Score_LLM.Agent_name, 'json ERROR, RETRY!')
                refine_Response, refine_Time = Score_LLM.ask()
                refine_response, refine_input_tokens, refine_output_tokens, refine_total_tokens, \
                    refine_reasoning_tokens, refine_Time = parse_response(model_name, refine_Response, refine_Time)
        
        Score_LLM.add_assistant(refine_response)

        time_lst.append(refine_Time)
        token_lst.append({
                'input_tokens': refine_input_tokens, 
                'output_tokens': refine_output_tokens,
                'total_tokens': refine_total_tokens,
                'reasoning_tokens': refine_reasoning_tokens
            })

        Score_lst.append(refine_response)

    Judge_user_prompt = Judge_prompts[1].format(future_scenario = fs, \
                                    response_content = "Step-" + str(step_num) + ' Response Content: \n' + response_content['Step-' + str(step_num)], \
                                    score_dimensions = score_dimensions, \
                                    score_result_1 = json.dumps(Score_lst[0], ensure_ascii=False, indent=4),\
                                    score_result_2 = json.dumps(Score_lst[1], ensure_ascii=False, indent=4),\
                                    score_result_3 = json.dumps(Score_lst[2], ensure_ascii=False, indent=4),\
                                    step_num = step_num)

    # Judge_LLM
    Judge_LLM.add_user(Judge_user_prompt)

    Response, Time = Judge_LLM.ask()

    response, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = parse_response(model_name, Response, Time)

    while True:
        if response in ['1', '2', '3']:
            true_response = Score_lst[int(response) - 1]
            break
        else:
            print(Judge_LLM.Agent_name, "output is not [1, 2, 3], retrying...")
            Response, Time = Judge_LLM.ask()
            response, input_tokens, output_tokens, total_tokens, reasoning_tokens, Time = parse_response(model_name, Response, Time)

    Judge_LLM.add_assistant(response)

    time_lst.append(Time)
    token_lst.append({
            'input_tokens': input_tokens, 
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        })
    
    # Judge_LLM self-refine
    Judge_LLM.add_user(Judge_LLM_refine_prompt)

    refine_Response, refine_Time = Judge_LLM.ask()

    refine_response, refine_input_tokens, refine_output_tokens, refine_total_tokens, \
        refine_reasoning_tokens, refine_Time = parse_response(model_name, refine_Response, refine_Time)

    while True:
        if refine_response in ['1', '2', '3']:
            refine_true_response = Score_lst[int(refine_response) - 1]
            break
        else:
            print(Judge_LLM.Agent_name, 'output is not [1, 2, 3], retrying...')
            refine_Response, refine_Time = Judge_LLM.ask()
            refine_response, refine_input_tokens, refine_output_tokens, refine_total_tokens, \
                refine_reasoning_tokens, refine_Time = parse_response(model_name, refine_Response, refine_Time)
    
    Judge_LLM.add_assistant(refine_response)

    time_lst.append(refine_Time)
    token_lst.append({
            'input_tokens': refine_input_tokens, 
            'output_tokens': refine_output_tokens,
            'total_tokens': refine_total_tokens,
            'reasoning_tokens': refine_reasoning_tokens
        })

    Score_lst.append(refine_true_response)

    return Score_lst, time_lst, token_lst



def add_step_head(lst, ans_id):
    new_lst = lst
    for step in STEP_num:
        new_lst[ans_id]["Step-" + str(step)] = {}
    return new_lst



def GoT_Score(Model_name, ans_id, single_response, file_core_name, temperature):
    
    inter_scores_name = os.path.join("Results", "inter_scores_in_ToT_and_GoT", file_core_name + "_" + Model_name + ".json")
    file_name = os.path.join("Results", "Raw_Results", file_core_name + "_" + Model_name + ".json")
    token_file_name = os.path.join("Results", 'Latency_Results', "[TOKEN]" + file_core_name + "_" + Model_name + ".json")
    time_file_name = os.path.join("Results", 'Latency_Results', "[TIME]" + file_core_name + "_" + Model_name + ".json")
    
    inter_scores = load_json_if_exists(inter_scores_name)
    final_score = load_json_if_exists(file_name)
    token_consumption_lst = load_json_if_exists(token_file_name)
    time_consumption_lst = load_json_if_exists(time_file_name)
    
    Score_LLM_1 = build_agent(model_name = Model_name, agent_name = 'Score_LLM_1', temperature = temperature)
    Score_LLM_2 = build_agent(model_name = Model_name, agent_name = 'Score_LLM_2', temperature = temperature)
    Score_LLM_3 = build_agent(model_name = Model_name, agent_name = 'Score_LLM_3', temperature = temperature)
    Judge_LLM = build_agent(model_name = Model_name, agent_name = 'Judge_LLM', temperature = temperature)
    
    final_result = {}

    inter_scores = add_step_head(inter_scores, ans_id)
    token_consumption_lst = add_step_head(token_consumption_lst, ans_id)
    time_consumption_lst = add_step_head(time_consumption_lst, ans_id)


    for step_num in STEP_num:

        Score_lst, time_lst, token_lst = ToT_Step_Score(Model_name, int(ans_id.split("FS")[1]), step_num, single_response, \
                                          [Score_LLM_1, Score_LLM_2, Score_LLM_3, Judge_LLM])

        final_Step_score = Score_lst[-1]
        
        token_consumption_lst[ans_id]["Step-" + str(step_num)] = token_lst
        time_consumption_lst[ans_id]["Step-" + str(step_num)] = time_lst
        inter_scores[ans_id]["Step-" + str(step_num)] = Score_lst[:-1]
        final_result.update(final_Step_score)
        print('=========================', model_name, ans_id, 'Step-' + str(step_num), '打分完成=========================')

    final_score[ans_id] = final_result


    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=4)
    with open(inter_scores_name, "w", encoding="utf-8") as f:
        json.dump(inter_scores, f, ensure_ascii=False, indent=4)
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
    
    # GoT
    Pure_responses = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))
    Pure_responses = dict(list(Pure_responses.items())[10:11])

    # ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4-mini']
    model_name = 'gemini-3.1-pro-preview'
    temperature = 0.2
            
    for ans_id in Pure_responses:
        GoT_Score(model_name, ans_id, Pure_responses[ans_id], 'GoT', temperature)