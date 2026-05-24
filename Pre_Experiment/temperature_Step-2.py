'''
该文件用于对温度参数进行预实验，包含 Phase 2
'''
import sys
import os
import random
import json
import numpy as np
from transformers import logging
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch

logging.set_verbosity_error()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import build_agent, parse_response


# 用指定温度的 sot 进行打分
INDICATED_TEMPERATURE = 0.2


# 设置同一条件下的重复次数
REPEATS = 2

# 随机抽取 10 个作答进行预实验
RANDOM_ANS_ID = ['A10_FS6', 'B03_FS5', 'A01_FS7', 'B10_FS7', 'A04_FS10', 'B09_FS3', 'A05_FS5', 'A07_FS1', 'A06_FS8', 'A08_FS10']
# , 'A06_FS5', 'A03_FS4', 'A10_FS7', 'A07_FS5', 'B06_FS2', 'A02_FS1', 'A07_FS8', 'A06_FS1', 'A04_FS6', 'A09_FS10']

# 指定进行预实验的步骤和维度
INDICATED_STEPS_AND_DIMENSIONS = {
    'Step-1': ['Fluency', 'Flexibility', 'Elaboration', 'Originality'],
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
        save_dir
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

    ax.set_ylabel('Variance', fontsize=12, fontweight='bold')

    # ax.set_title(f'{model_name} - {step_name}', fontsize=14)

    ax.set_ylim(range[0], range[1])

    # ax.grid(alpha=0.3)

    ax.legend()
    ax = add_arrow(ax)

    fig.tight_layout()

    # 保存
    save_file_name = os.path.join(
        save_dir,
        f'{model_name}_{step_name}.pdf'
    )

    fig.savefig(save_file_name, dpi=600, bbox_inches='tight')

    plt.close(fig)

    print(f'Saved to: {save_file_name}')




def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {ans_id: {} for ans_id in RANDOM_ANS_ID}





def judge_scoring(model_name, temperature, ans_id, repeat_time):
    '''
    针对某一模型在特定参数下进行打分
    '''
    def collect_sot_for_judge(sot_result):
        '''
        为每份作答拼接其用于打分的 sot 结果
        输入为一份作答的 sot，即 sot_results[ans_id]
        '''
        result = {}

        for step_name, step_content in sot_result.items():
            assert 'sot_evidence' in step_content, ans_id + ' 作答的 ' + step_name + ' 没有 sot_evidence 字段！'
            result[step_name] = step_content["sot_evidence"]

        return result
    
    def collect_step_dimension_for_Judge(step):
        '''
        为 Judge_LLM 整理维度说明 prompt，不涉及 Category_list 和 few-shot
        '''
        step_config = json.load(open(r"jsons\steps_config_for_Judge_LLM.json", 'r', encoding='utf-8'))[step]
        dimensions = ''
        for id, dim in enumerate(step_config['Step-rubrics'], start=1):
            dimensions += f"# 维度-{id}：{dim['dimension_name']} \n{dim['dimension_rubrics']}\n\n"

        prompt = "## {step_name}\n步骤要求：{step_description}\n\n{dimensions}\n\n".format(
            step_name = step_config['Step-Name'],
            step_description = step_config['Step-Description'],
            dimensions = dimensions
        )
        return prompt.strip()
    

    def collect_score_template(file_name):
        '''
        返回所有步骤的评分输出格式
        '''
        output_template = json.load(open(file_name, 'r', encoding='utf-8'))
        
        result = []
        for i, step in enumerate(range(1, len(INDICATED_STEPS_AND_DIMENSIONS) + 1)):
            content = output_template['Step-' + str(step)][1:-1].strip()
            
            if i != len(range(1, len(INDICATED_STEPS_AND_DIMENSIONS) + 1)) - 1:
                content += ','
            
            result.append(content)
        
        return '{\n' + '\n'.join(result) + '\n}'


    def Judge_Scoring(model_name, FS_num, sot_output, temperature):
        '''
        通过 Judge_LLM 对每份作答进行评分，输出最终的评分结果
        '''
        Judge_LLM = build_agent(model_name = model_name, agent_name = 'Judge_LLM', temperature = temperature)
        
        prompts = json.load(open(r"jsons\Judge_LLM_Pre_experiment_prompt.json", 'r', encoding='utf-8'))
        fs = json.load(open(r"jsons\future_scenarios_ch.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
        score_dimensions = '\n\n\n'.join([collect_step_dimension_for_Judge(step) for step in INDICATED_STEPS_AND_DIMENSIONS])
        output_template = collect_score_template(r"jsons\Score_Output_Template_ch.json")

        user_prompt = prompts[1].format(future_scenario = fs, \
                                        sot_output = sot_output, \
                                        score_dimensions = score_dimensions,
                                        output_template = output_template)
        
        Judge_LLM.add_system(prompts[0])
        Judge_LLM.add_user(user_prompt)
        response, Time = Judge_LLM.ask()

        return parse_response(model_name, response, Time)

 
    sot_file_name = os.path.join(r'Pre_Experiment\sots', 'sot_' + model_name + '_tem=' + str(INDICATED_TEMPERATURE) + '_repeat=0_with_dim_name.json')
    sot_outputs = json.load(open(sot_file_name, 'r', encoding='utf-8'))
    sot_output = collect_sot_for_judge(sot_outputs[ans_id])
    sot_output = json.dumps(sot_output, ensure_ascii=False, indent=4)
    
    score_file_name = os.path.join(r'Pre_Experiment\score_results', model_name + '_tem=' + str(temperature) + '_repeat=' + str(repeat_time) + '.json')
    final_score = load_json_if_exists(score_file_name)


    Judge_result, _, _, _, _, _ = \
        Judge_Scoring(model_name, int(ans_id.split("FS")[1]), sot_output, temperature)
        
    if Judge_result.startswith("```json") and Judge_result.endswith("```"):
        Judge_result = Judge_result.strip("```json").strip("```")

    Judge_result = json.loads(Judge_result)

    final_score[ans_id] = Judge_result
    
    # print(ans_id + '打分完成！')

    with open(score_file_name, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=4)
    # print('=========================', ans_id, 'repeat = ', repeat_time, '打分完成=========================\n')



def Each_model_and_temperature(model_name, temperature):
    print('=================', model_name, temperature, '=================')

    for ans_id in RANDOM_ANS_ID:
        print('++++++++++++++++', ans_id, '++++++++++++++++')

        for repeat_time in range(REPEATS):

            print('>> repeat ', repeat_time)

            judge_scoring(model_name, temperature, ans_id, repeat_time)



def cal_variance(model_name, temperature):
    print('=======================', model_name, temperature, '计算方差 =======================')

    # 对于每个维度
    all_lst = {}
    for step, dimensions in INDICATED_STEPS_AND_DIMENSIONS.items():

        score_file_name_00 = os.path.join(r'Pre_Experiment\score_results', model_name + '_tem=' + str(temperature)+ '_repeat=0.json')
        SCORE_Results_00 = json.load(open(score_file_name_00, 'r', encoding='utf-8'))

        if step == 'Step-2':

            step_lst = {}
            
            for dimension in dimensions:

                dim_lst = {}

                for ans_id in RANDOM_ANS_ID:

                    ans_lst = {}
                    SOTS_and_id = []

                    for repeat_time in range(REPEATS):

                        score_file_name = os.path.join(r'Pre_Experiment\score_results', model_name + '_tem=' + str(temperature)+ '_repeat=' + str(repeat_time) + '.json')
                        SCORE_Results = json.load(open(score_file_name, 'r', encoding='utf-8'))
                        response = SCORE_Results[ans_id][step][dimension]

                        SOTS_and_id.append(int(response))
                    
                    dim_lst[ans_id] = round(np.var(SOTS_and_id), 3)
                
                step_lst[dimension] = dim_lst

        elif step == 'Step-1':

            step_lst = {}
            
            for dimension in dimensions:

                if dimension == 'Flexibility':

                    dim_lst = {}

                    for ans_id in RANDOM_ANS_ID:

                        SOTS_and_id = []

                        for repeat_time in range(REPEATS):

                            score_file_name = os.path.join(r'Pre_Experiment\score_results', model_name + '_tem=' + str(temperature)+ '_repeat=' + str(repeat_time) + '.json')
                            SCORE_Results = json.load(open(score_file_name, 'r', encoding='utf-8'))
                            response = SCORE_Results[ans_id][step]['summary'][dimension]

                            SOTS_and_id.append(int(response))
                        
                        dim_lst[ans_id] = np.var(SOTS_and_id)


                else:

                    dim_lst = {}

                    for ans_id in RANDOM_ANS_ID:

                        ans_lst = {}

                        for item_num in range(len(SCORE_Results_00[ans_id][step]['each_challenge'])):                  

                            SOTS_and_id = []

                            for repeat_time in range(REPEATS):

                                score_file_name = os.path.join(r'Pre_Experiment\score_results', model_name + '_tem=' + str(temperature)+ '_repeat=' + str(repeat_time) + '.json')
                                SCORE_Results = json.load(open(score_file_name, 'r', encoding='utf-8'))
                                response = SCORE_Results[ans_id][step]['each_challenge'][item_num][dimension]

                                SOTS_and_id.append(1 if response == 'Yes' else 0 if response == 'No' else int(response))

                            ans_lst[SCORE_Results_00[ans_id][step]['each_challenge'][item_num]['id']] = np.var(SOTS_and_id)
                        
                        dim_lst[ans_id] = np.mean(list(ans_lst.values()))
                
                step_lst[dimension] = dim_lst

        all_lst[step] = step_lst

    with open(os.path.join(r'Pre_Experiment\Raw_Results_Scoring', model_name + "_tem=" + str(temperature) + '.json'), "w", encoding="utf-8") as f:
        json.dump(all_lst, f, ensure_ascii=False, indent=4)




def collate_results(model_name, range, form):
    '''
    将 Raw_Results_Scoring 文件夹里面指定模型的的结果整理为最终结果图
    '''
    AAA = {}
    for step in INDICATED_STEPS_AND_DIMENSIONS:
        ddd = {}

        for temperature in TEMPERATURE_LST:

            raw_result = json.load(open(os.path.join(r'Pre_Experiment\Raw_Results_Scoring', model_name + "_tem=" + str(temperature) + '.json'), 'r', encoding='utf-8'))
            ccc = {}

            for dimension in INDICATED_STEPS_AND_DIMENSIONS[step]:

                Dim_lst = raw_result[step][dimension]
                aaa_lst = []

                for ans_id in RANDOM_ANS_ID:

                    aaa_lst.append(Dim_lst[ans_id])

                ccc[dimension] = np.mean(np.array(aaa_lst))
            
            ddd[temperature] = ccc
        
        if form == 'plot':
            draw_temperature_lineplot(ddd, model_name, step, range, r'Pre_Experiment\PLOTS_Scoring')
        elif form == 'table':
            AAA[step] = ddd
        
        if step == 'Step-2':
            NEW = {}
            for tem in ddd:
                new_dic = {}
                sub_dims = ['Condition Phrase', 'Stem & KVP', 'Purpose', 'FS Parameters']
                new_dic['Integrity'] = np.mean(np.array([ddd[tem][sub_dim] for sub_dim in ddd[tem] if sub_dim in sub_dims]))
                for dim in INDICATED_STEPS_AND_DIMENSIONS['Step-2']:
                    if dim not in sub_dims:
                        new_dic[dim] = ddd[tem][dim]
                NEW[tem] = new_dic

        AAA[step] = NEW if step == 'Step-2' else ddd
        
    if form == 'table':
        return AAA


def dict_to_table(data, save_file_name):
    """
    data格式：
    {
        Step-1: {
            temp: {dim: value}
        },
        Step-2: {...}
    }
    """

    # Step-2中特殊维度
    integrity_keys = ['Condition Phrase', 'Stem & KVP', 'Purpose', 'FS Parameters']

    rows = []

    for temp in TEMPERATURE_LST:
        row = {'Temperature': temp}

        for step, step_data in data.items():
            # 收集该step在该temperature下的dimension
            dim_dict = step_data[temp]

            if step == 'Step-2':
                # 计算 Integrity
                integrity_vals = [dim_dict[k] for k in integrity_keys]
                row[f'{step}-Integrity'] = sum(integrity_vals) / len(integrity_vals)

            # 其他维度原样写入
            for k, v in dim_dict.items():
                if k not in integrity_keys:
                    row[f'{step}-{k}'] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    # Temperature作为第一列
    cols = ['Temperature'] + [c for c in df.columns if c != 'Temperature']
    df = df[cols]

    # 平均值
    df['AVG'] = df.drop(columns=['Temperature']).mean(axis=1)

    df.to_excel(save_file_name, index=False)
    print('Saved to: ', save_file_name)

    return df




def write_table(model_name):
    '''
    将 Raw_Results_Scoring 文件夹里面指定模型的的结果整理为最终表格形式
    '''
    save_file_name = os.path.join('Pre_Experiment\PLOTS_Scoring', model_name + '.xlsx')

    model_dict = {}

    for step in INDICATED_STEPS_AND_DIMENSIONS:
        step_dict = {}   # ⭐ step 层

        for temperature in TEMPERATURE_LST:
            raw_result = json.load(open(
                os.path.join(
                    r'Pre_Experiment\Raw_Results_Scoring',
                    model_name + "_tem=" + str(temperature) + '.json'
                ),
                'r',
                encoding='utf-8'
            ))

            ccc = {}  # temperature-level

            for dimension in INDICATED_STEPS_AND_DIMENSIONS[step]:
                Dim_lst = raw_result[step][dimension]
                aaa_lst = []

                for ans_id in RANDOM_ANS_ID:
                    aaa_lst.append(Dim_lst[ans_id])

                ccc[dimension] = np.mean(np.array(aaa_lst))

            step_dict[temperature] = ccc   # ⭐ 保存 temperature

        model_dict[step] = step_dict       # ⭐ 保存 step
    
    dict_to_table(model_dict, save_file_name)



    # AAA = {}
    # for model_name in MODEL_LST:

    #     for step in INDICATED_STEPS_AND_DIMENSIONS:
    #         ddd = {}

    #         for temperature in TEMPERATURE_LST:

    #             raw_result = json.load(open(os.path.join(r'Pre_Experiment\Raw_Results_Scoring', model_name + "_tem=" + str(temperature) + '.json'), 'r', encoding='utf-8'))
    #             ccc = {}

    #             for dimension in INDICATED_STEPS_AND_DIMENSIONS[step]:

    #                 Dim_lst = raw_result[step][dimension]
    #                 aaa_lst = []

    #                 for ans_id in RANDOM_ANS_ID:

    #                     aaa_lst.append(Dim_lst[ans_id])

    #                 ccc[dimension] = np.mean(np.array(aaa_lst))
                
    #             ddd[temperature] = ccc

    #     AAA[model_name] = ddd



if __name__ == "__main__":
    '''
    Judge-LLM 进行评分
    '''
    # for model_name in MODEL_LST:
        
    #     # 先对某一模型所有温度进行打分
    #     for temperature in TEMPERATURE_LST:

    #         Each_model_and_temperature(model_name, temperature)

    
    '''
    计算分数方差
    '''
    # for model_name in MODEL_LST:

    #     print(model_name, '\n\n')
        
    #     # 先对某一模型所有温度进行 SoT Extraction
    #     for temperature in TEMPERATURE_LST:

    #         cal_variance(model_name, temperature)


            
    '''
    Raw_Results 文件夹中若某个模型存在三个温度参数文件，则可以计算该模型在两个步骤下的图像
    '''
    ME_DIC = {}
    for model in MODEL_LST:
        ME_DIC[model] = collate_results(model, (0, 1), 'table') # 最后是图像展示的纵轴范围
    write_to_table(ME_DIC, os.path.join(r'Pre_Experiment\PLOTS_Scoring', 'ALL.xlsx'))

    # 所有结果写入表格
    # for model in MODEL_LST:
    #     write_table(model)