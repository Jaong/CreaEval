'''
该文件用于对温度参数进行预实验，包含 Phase 1
'''
import sys
import os
import random
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import numpy as np
from transformers import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

logging.set_verbosity_error()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import build_agent, parse_response


# 设置同一条件下的重复次数
REPEATS = 5

# 随机抽取 10 个作答进行预实验
RANDOM_ANS_ID = ['A10_FS6', 'B03_FS5', 'A01_FS7', 'B10_FS7', 'A04_FS10', 'B09_FS3', 'A05_FS5', 'A07_FS1', 'A06_FS8', 'A08_FS10']
# , 'A06_FS5', 'A03_FS4', 'A10_FS7', 'A07_FS5', 'B06_FS2', 'A02_FS1', 'A07_FS8', 'A06_FS1', 'A04_FS6', 'A09_FS10']

# 指定进行预实验的步骤和维度
INDICATED_STEPS_AND_DIMENSIONS = {
    'Step-1': ['Fluency', 'Flexbility', 'Elaboration', 'Originality'],
    'Step-2': ['Condition Phrase', 'Stem & KVP', 'Purpose', 'FS Parameters', 'Focus', 'Adequacy']
}

# 原始作答
PURE_RESPONSES = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))

# 所有 Judge LLM
MODEL_LST = ['qwen3.6-plus-2026-04-02', 'deepseek-v4-pro', 'gemini-3.1-pro-preview', 'gpt-5.4']

# 预选的温度参数
TEMPERATURE_LST = [0.2, 0.5, 0.8]

COLOR_LST = [
    '#4C72B0',  # 蓝色
    '#DD8452',  # 橙色
    '#55A868',  # 绿色
]


def add_arrow(ax): # 为图的左框线和右框线加粗且添加箭头
    # 添加左边框箭头
    arrow_props = dict(facecolor='black', arrowstyle='-|>', linewidth=1.5)
    left_arrow = FancyArrowPatch(( -0.04, 0), ( -0.04, 1.02), transform=ax.transAxes, **arrow_props)
    ax.add_patch(left_arrow)

    # 添加下边框箭头
    bottom_arrow = FancyArrowPatch((0, -0.04), (1.02, -0.04), transform=ax.transAxes, **arrow_props)
    ax.add_patch(bottom_arrow)

    # 左边框箭头，从(0, 1)往(0, 1.05)画，表示顶端箭头向上
    ax.annotate(
        '', 
        xy=(0, 1.05), xycoords='axes fraction',  # 箭头尖端稍微超出顶端
        xytext=(0, 0), textcoords='axes fraction',  # 箭头尾端顶端位置
        arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=1.5),
        annotation_clip=False
    )

    # 下边框箭头，从(1, 0)往(1.05, 0)画，表示末端箭头向右
    ax.annotate(
        '', 
        xy=(1.05, 0), xycoords='axes fraction',
        xytext=(0, 0), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=1.5),
        annotation_clip=False
    )
    ax.spines['top'].set_visible(False) # 隐藏坐标轴上边框
    ax.spines['right'].set_visible(False) # 隐藏坐标轴右边框
    return ax



def draw_temperature_lineplot(
        ddd,
        model_name,
        step_name,
        range,
        save_dir,
        metric
):
    """
    ddd:
        {
            temperature: {
                dimension: score
            }
        }
    range: (a, b) 表示折线图 y 轴范围

    画图：
    - 每条线 = 一个 temperature
    - 横轴 = dimension
    - 纵轴 = score
    """

    # 获取维度顺序（默认使用第一个temperature）
    dimensions = list(next(iter(ddd.values())).keys())

    # =========================
    # 使用 ax
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5))

    # 逐个temperature画线
    for i, (temperature, score_dict) in enumerate(ddd.items()):

        y = [score_dict[dim] for dim in dimensions]

        ax.plot(
            dimensions,
            y,
            COLOR_LST[i],
            marker='o',
            linewidth=1.5,
            label=f'temperature = {temperature}'
        )

    # 图像设置
    # ax.set_xlabel('Dimensions', fontsize=12, fontweight='bold')

    ppp = {
        'bert': 'BertScore',
        'bleu': 'Self-Bleu'
    }

    ax.set_ylabel(ppp[metric], fontsize=12, fontweight='bold')

    # ax.set_title(f'{model_name} - {step_name}', fontsize=14)

    ax.set_ylim(range[0], range[1])

    # ax.grid(alpha=0.3)

    ax.legend()
    ax = add_arrow(ax)

    fig.tight_layout()

    # 保存
    save_file_name = os.path.join(
        save_dir,
        f'{model_name}_{step_name}_{metric}.pdf'
    )

    fig.savefig(save_file_name, dpi=600, bbox_inches='tight')

    plt.close(fig)

    print(f'Saved to: {save_file_name}')


def collect_step_dimension_for_sot(step):
    '''
    为 SoT_LLM 整理维度说明 prompt
    '''
    step_config = json.load(open(r"jsons\steps_config_for_SoT_LLM.json", 'r', encoding='utf-8'))[step]
    Category_lst = "\n".join(json.load(open(r"jsons\category_lst.json", 'r', encoding='utf-8')))
    dimensions = ''
    for id, dim in enumerate(step_config['Step-rubrics'], start=1):
        dimensions += f"# 维度-{id}：{dim['dimension_name']} \n{dim['dimension_rubrics'].replace('{Category_lst}', Category_lst)}\n\n"

    prompt = "## {step_name}\n步骤要求：{step_description}\n\n{dimensions}".format(
        step_name = step_config['Step-Name'],
        step_description = step_config['Step-Description'],
        dimensions = dimensions
    )
    return prompt.strip()



def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {ans_id: {} for ans_id in RANDOM_ANS_ID}



def compute_self_bertscore(texts):

    pair_scores = []

    for i in range(len(texts)):

        cand = [texts[i]]

        refs = texts[:i] + texts[i+1:]

        # 与其它文本分别比较
        cands = cand * len(refs)

        P, R, F1 = score(
            cands,
            refs,
            lang='en',
            verbose=False
        )

        pair_scores.extend(F1.tolist())

    return np.mean(pair_scores)



def compute_self_bleu(texts):
    """
    texts: List[str]
    return: average self-BLEU
    """

    smoothie = SmoothingFunction().method1

    bleu_scores = []

    tokenized = [t.split() for t in texts]

    for i in range(len(tokenized)):

        hypothesis = tokenized[i]

        references = tokenized[:i] + tokenized[i+1:]

        score = sentence_bleu(
            references,
            hypothesis,
            smoothing_function=smoothie,
            weights=(0.8, 0.2)
        )

        bleu_scores.append(score)

    return np.mean(bleu_scores)




def update_dim_name(sot_file_name):
    '''
    将 SoT 结果中的 "dimension_i" 替换为对应的 <维度名称>
    '''
    final_sot = json.load(open(sot_file_name, 'r', encoding='utf-8'))
    save_file = sot_file_name.replace('.json', '_with_dim_name.json')
    new_data = {}

    dimension_map = INDICATED_STEPS_AND_DIMENSIONS
    
    for ans_id, ans_content in final_sot.items():
        new_data[ans_id] = {}

        for step, step_content in ans_content.items():
            new_data[ans_id][step] = {}

            sot = step_content["sot_evidence"]
            new_sot = {}

            dim_names = dimension_map[step]

            # ===== Step-1 / 3 / 4（有 item 层）=====
            if step in ['Step-1', 'Step-3', 'Step-4']:
                for item_id, item_content in sot.items():
                    new_item = {}

                    for i, dim_name in enumerate(dim_names, start=1):
                        key = f"dimension_{i}"
                        if key in item_content:
                            new_item[dim_name] = item_content[key]

                    new_sot[item_id] = new_item

            # ===== Step-2 / 5 / 6（无 item 层）=====
            else:
                for i, dim_name in enumerate(dim_names, start=1):
                    key = f"dimension_{i}"
                    if key in sot:
                        new_sot[dim_name] = sot[key]

            # 写回
            new_data[ans_id][step]["sot_evidence"] = new_sot
            new_data[ans_id][step]["task_status_memory"] = step_content["task_status_memory"]

    # return new_data
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(save_file, '文件已保存！')
    return save_file



def sot_extraction(model_name, temperature, ans_id, repeat_time):
    '''
    针对某一模型在特定参数下进行 SoT Extration
    '''
    def SoT_Extraction(model_name, step, FS_num, raw_text, pre_sot):
        dependency_step = {2:[1], 3:[2, 1], 4:[3, 2, 1], 5:[4, 3, 2, 1], 6:[5, 4, 3, 2, 1]} # 记录每个步骤需要之前步骤的依赖

        # assert 'gpt' in model_name or 'gemini' in model_name, 'SoT 模型选择不属于 gpt 或 gemini 系列！'
        SoT_LLM = build_agent(model_name = model_name, agent_name = 'SoT_LLM', temperature = temperature)

        prompts = json.load(open(r"jsons\SoT_LLM_prompt.json", 'r', encoding='utf-8'))
        fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
        score_dimensions_wo_rubrics = collect_step_dimension_for_sot(step)
        step_schema = json.load(open(r"jsons\step_schema.json", 'r', encoding='utf-8'))[step]


        user_prompt = prompts[1].format(step_num = step, \
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

    sot_file_name = os.path.join(r'Pre_Experiment\sots', 'sot_' + model_name + '_tem=' + str(temperature) + '_repeat=' + str(repeat_time) + '.json')
    final_sot = load_json_if_exists(sot_file_name)

    for step in INDICATED_STEPS_AND_DIMENSIONS:
            
        step_answer = PURE_RESPONSES[ans_id][step]

        SoT_result, _, _, _, _, _ = \
            SoT_Extraction(model_name, step, int(ans_id.split("FS")[1]), step_answer, final_sot[ans_id])
        
        if SoT_result.startswith("```json") and SoT_result.endswith("```"):
            SoT_result = SoT_result.strip("```json").strip("```")


        SoT_result = json.loads(SoT_result)

        final_sot[ans_id][step] = SoT_result
        
        print(ans_id + ' 的 ' + 'Step-' + str(step) + ' SoT 结果已生成')

    with open(sot_file_name, "w", encoding="utf-8") as f:
        json.dump(final_sot, f, ensure_ascii=False, indent=4)
    print('=========================', ans_id, 'repeat = ', repeat_time, 'SoT 提取完成=========================\n')



def cal_similarity(model_name, temperature):
    print('\n\n=======================', model_name, temperature, '=======================')

    # 对于每个维度
    all_lst = {}
    for step, dimensions in INDICATED_STEPS_AND_DIMENSIONS.items():

        print(step, '\n')

        sot_file_name_00 = os.path.join(r'Pre_Experiment\sots', 'sot_' + model_name + '_tem=' + str(temperature)+ '_repeat=0_with_dim_name.json')
        SOT_Results_00 = json.load(open(sot_file_name_00, 'r', encoding='utf-8'))

        if step == 'Step-2':

            step_lst = {}
            
            for dimension in dimensions:

                print(dimension)

                dim_lst = {}

                for ans_id in RANDOM_ANS_ID:

                    print(ans_id)

                    ans_lst = {}
                    SOTS_and_id = []

                    for repeat_time in range(REPEATS):

                        sot_file_name = os.path.join(r'Pre_Experiment\sots', 'sot_' + model_name + '_tem=' + str(temperature)+ '_repeat=' + str(repeat_time) + '_with_dim_name.json')
                        SOT_Results = json.load(open(sot_file_name, 'r', encoding='utf-8'))
                        response = SOT_Results[ans_id][step]['sot_evidence'][dimension]

                        SOTS_and_id.append(response)
                    
                    bleu_score = compute_self_bleu(SOTS_and_id)
                    bert_score = compute_self_bertscore(SOTS_and_id)

                    dim_lst[ans_id] = {}

                    dim_lst[ans_id]['bleu'] = round(bleu_score, 3)
                    dim_lst[ans_id]['bert'] = round(bert_score, 3)
                
                step_lst[dimension] = dim_lst
                print()

        elif step == 'Step-1':

            step_lst = {}
            
            for dimension in dimensions:

                print(dimension)

                dim_lst = {}

                for ans_id in RANDOM_ANS_ID:

                    print(ans_id)

                    ans_lst = {}

                    for item in SOT_Results_00[ans_id][step]['sot_evidence']: 

                        print(item)                           

                        SOTS_and_id = []

                        for repeat_time in range(REPEATS):

                            sot_file_name = os.path.join(r'Pre_Experiment\sots', 'sot_' + model_name + '_tem=' + str(temperature)+ '_repeat=' + str(repeat_time) + '_with_dim_name.json')
                            SOT_Results = json.load(open(sot_file_name, 'r', encoding='utf-8'))
                            response = SOT_Results[ans_id][step]['sot_evidence'][item][dimension]

                            SOTS_and_id.append(response)
                        
                        bleu_score = compute_self_bleu(SOTS_and_id)
                        bert_score = compute_self_bertscore(SOTS_and_id)

                        ans_lst[item] = {}
                        ans_lst[item]['bleu'] = round(bleu_score, 3)
                        ans_lst[item]['bert'] = round(bert_score, 3)
                    
                    dim_lst[ans_id] = ans_lst
                
                step_lst[dimension] = dim_lst
                print()

        all_lst[step] = step_lst

    with open(os.path.join(r'Pre_Experiment\Raw_Results', model_name + "_tem=" + str(temperature) + '.json'), "w", encoding="utf-8") as f:
        json.dump(all_lst, f, ensure_ascii=False, indent=4)



def Each_model_and_temperature(model_name, temperature):
    print('=================', model_name, temperature, '=================')

    for ans_id in RANDOM_ANS_ID:
        print('>>> ', ans_id)

        for repeat_time in range(REPEATS):

            print('>> repeat ', repeat_time)

            sot_extraction(model_name, temperature, ans_id, repeat_time)

    
    

def collate_results_step_2(model_name, metric, range, form):
    '''
    将 Raw_Results 文件夹里面指定模型的的结果整理为最终结果图
    指定指标: ['bleu', 'bert']
    '''
    step = 'Step-2'
    ddd = {}

    for temperature in TEMPERATURE_LST:

        raw_result = json.load(open(os.path.join(r'Pre_Experiment\Raw_Results', model_name + "_tem=" + str(temperature) + '.json'), 'r', encoding='utf-8'))
        ccc = {}

        for dimension in INDICATED_STEPS_AND_DIMENSIONS[step]:

            Dim_lst = raw_result[step][dimension]
            aaa_lst = []

            for ans_id in RANDOM_ANS_ID:

                aaa_lst.append(Dim_lst[ans_id][metric])

            ccc[dimension] = np.mean(np.array(aaa_lst))
        
        ddd[temperature] = ccc

    if form == 'table':
        NEW = {}
        for tem in ddd:
            new_dic = {}
            sub_dims = ['Condition Phrase', 'Stem & KVP', 'Purpose', 'FS Parameters']
            new_dic['Integrity'] = np.mean(np.array([ddd[tem][sub_dim] for sub_dim in ddd[tem] if sub_dim in sub_dims]))
            for dim in INDICATED_STEPS_AND_DIMENSIONS['Step-2']:
                if dim not in sub_dims:
                    new_dic[dim] = ddd[tem][dim]
            NEW[tem] = new_dic

        return NEW
    elif form == 'fig':
        draw_temperature_lineplot(ddd, model_name, step, range, r'Pre_Experiment\PLOTS', metric)


def collate_results_step_1(model_name, metric, range, form):
    '''
    将 Raw_Results 文件夹里面指定模型的的结果整理为最终结果图
    指定指标: ['bleu', 'bert']
    '''
    ddd = {}
    step = 'Step-1'

    for temperature in TEMPERATURE_LST:

        raw_result = json.load(open(os.path.join(r'Pre_Experiment\Raw_Results', model_name + "_tem=" + str(temperature) + '.json'), 'r', encoding='utf-8'))
        ccc = {}

        for dimension in INDICATED_STEPS_AND_DIMENSIONS[step]:

            Dim_lst = raw_result[step][dimension]
            bbb = {}

            for ans_id in RANDOM_ANS_ID:

                aaa_lst = []

                for item in Dim_lst[ans_id]:

                    aaa_lst.append(Dim_lst[ans_id][item][metric])

                bbb[ans_id] = np.mean(np.array(aaa_lst))
            
            ccc[dimension] = np.mean(np.array(list(bbb.values())))

        
        ddd[temperature] = ccc
    
    if form == 'table':
        return ddd
    elif form == 'fig':
        draw_temperature_lineplot(ddd, model_name, step, range, r'Pre_Experiment\PLOTS', metric)



def write_to_table(data, save_file_name):
    """
    data格式：
    {
        model_name: {
            step: {
                temperature: {
                    dimension: value
                }
            }
        }
    }

    输出表格格式：
    第一行：Step
    第二行：Dimension
    第一列：Model
    第二列：Temperature
    """

    # =========================
    # 收集所有 step 和 dimension
    # =========================
    step_dim_pairs = []

    first_model = next(iter(data.values()))

    for step, temp_dict in first_model.items():

        first_temp = next(iter(temp_dict.values()))

        for dim in first_temp.keys():
            step_dim_pairs.append((step, dim))

    # =========================
    # 构造多级列索引
    # =========================
    columns = [('Model', ''), ('Temperature', '')]

    for step, dim in step_dim_pairs:
        columns.append((step, dim))
    columns.append(('AVG', ''))

    multi_columns = pd.MultiIndex.from_tuples(columns)

    # =========================
    # 构造数据行
    # =========================
    rows = []

    for model_name, step_data in data.items():

        temperatures = sorted(
            next(iter(step_data.values())).keys()
        )

        for idx, temperature in enumerate(temperatures):

            row = []

            # 模型名称只在第一行显示
            if idx == 0:
                row.append(model_name)
            else:
                row.append('')

            row.append(temperature)

            # 写入各维度数值
            scores = []
            for step, dim in step_dim_pairs:

                value = round(step_data[step][temperature][dim], 4)
                row.append(value)
                scores.append(value)

            row.append(round(np.mean(scores), 4))
            rows.append(row)

    # =========================
    # DataFrame
    # =========================
    df = pd.DataFrame(rows, columns=multi_columns)

    # =========================
    # 保存Excel
    # =========================
    df.to_excel(save_file_name, index=True, merge_cells=True)

    print('Saved to:', save_file_name)

    return df




if __name__ == "__main__":
    '''
    提取 SoT
    '''
    # for model_name in MODEL_LST:

    #     print(model_name, '\n\n')
        
    #     # 先对某一模型所有温度进行 SoT Extraction
    #     for temperature in TEMPERATURE_LST:

    #         Each_model_and_temperature(model_name, temperature)



    '''
    更新 _dim_name
    '''

    # for model_name in MODEL_LST:

    #     print(model_name, '\n\n')
        
    #     # 先对某一模型所有温度进行 SoT Extraction
    #     for temperature in TEMPERATURE_LST:

    #         for repeat_time in range(REPEATS):
    #             sot_file_name = os.path.join(r'Pre_Experiment\sots', 'sot_' + model_name + '_tem=' + str(temperature)+ '_repeat=' + str(repeat_time) + '.json')

    #             update_dim_name(sot_file_name)

    
    '''
    计算相似度
    '''
    # for model_name in MODEL_LST:

    #     print(model_name, '\n\n')
        
    #     # 先对某一模型所有温度进行 SoT Extraction
    #     for temperature in TEMPERATURE_LST:

    #         cal_similarity(model_name, temperature)


            
    # '''
    # Raw_Results 文件夹中若某个模型存在三个温度参数文件，则可以计算该模型在两个步骤下的图像
    # '''
    # for model in MODEL_LST:
    #     for metric in ['bert', 'bleu']:
    #         collate_results_step_1(model, metric, (0, 1), 'fig') # 最后是图像展示的纵轴范围
    #         collate_results_step_2(model, metric, (0, 1), 'fig')

    
    '''
    将结果汇总为表格形式，共两个表格：bert.xlsx, bleu.xlsx
    '''
    for metric in ['bert', 'bleu']:
        ME_DIC = {}
        for model in MODEL_LST:
            mode_dic = {}
            step1 = collate_results_step_1(model, metric, (0, 1), 'table') # 最后是图像展示的纵轴范围
            step2 = collate_results_step_2(model, metric, (0, 1), 'table')
            mode_dic['Step-1'] = step1
            mode_dic['Step-2'] = step2
            ME_DIC[model] = mode_dic
        write_to_table(ME_DIC, os.path.join(r'Pre_Experiment\PLOTS', metric + '.xlsx'))