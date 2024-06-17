import json
import os

import dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()

def call_gemini():
    return ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.0)

def call_gpt4p0():
    return ChatOpenAI(model='gpt-4-0125-preview', temperature=0.0)

def call_gpt3p5():
    return ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0.0)

def evidence_prompt(medical_term_num: int) -> list:
    medical_term = {
        1: "주소증(Chief Complaints)",
        2: "발병일(Date of Onset)",
        3: "과거력(Past Medical History)",
        4: "가족력(Family History)",
        5: "현병력(Present Illness)"
    }
    
    try:
        user_text_1 = f'''
        이번 상담 내용에서 나온 {medical_term[medical_term_num]}의 근거를 알고싶다. 각 증상별로 연관성이 가장 높은 1문장을 상담내용에서 발췌해줘.
        '''
        return user_text_1
    
    except KeyError:
        print("입력된 번호가 유효하지 않습니다. 1부터 5까지의 숫자 중 하나를 입력해주세요.")
        return None

def inference_prompt(medical_term_num: int) -> str:
    medical_term = {
        1: "주소증(Chief Complaints)",
        2: "발병일(Date of Onset)",
        3: "과거력(Past Medical History)",
        4: "가족력(Family History)",
        5: "현병력(Present Illness)"
    }
    
    try:
        user_text_1 = f'''
        이번 상담 내용에서 나온 {medical_term[medical_term_num]}을 알려주세요.
        [Inference]에 명시된 단계별로 생각해서 단계별로 추론한 결과를 알려주세요.
        '''
        return user_text_1
    except KeyError:
        print("입력된 번호가 유효하지 않습니다. 1부터 5까지의 숫자 중 하나를 입력해주세요.")
        return None
    

def summary_prompt() -> list:
    user_text_1 = f'''
    이번 상담 내용을 요약해주세요.
    Let's think step by step.
    '''
    return user_text_1

def init_prompt(input_content: str) -> list:
    user_text_prompt_1 = f'''
    [상담 내용]

    {input_content}

    ---

    여기까지 [상담 내용]입니다. 잘 기억해두세요. 다음 지시를 사항을 줄 때까지 기다리세요.
    '''

    user_text_prompt_2 = f'''
    [First order]
    위의 [[상담 내용]]에 해당하는 내용은 환자와 의사가 나눈 상담 내용입니다. 당신은 반드시 [[상담 내용]]을 기억해야 하고 상담 내용에 대한 정보에 대해서 질문을 받으면 친절하게 답변해야 합니다.
    지금은 답변을 하는 시간이 아닙니다. 사용자가 질문을 할 때 까지 기다리고 계시면 됩니다.

    [Personality]
    당신은 이제부터 의료 기록 전문가입니다. 그러므로 환자 정보를 정리하는 차트에 대해서 전문가입니다. 
    글을 작성하는 스타일은 개조식으로 중요한 요점이나 단어를 나열하는 방식을 의미한다. 키워드나 개요 중심으로 서술하며 대개 글머리표나 번호를 붙이는 형태를 입니다.
    정리하면 당신은 의학전문가이면서 차트 전문가이고 글 작성을 개조식으로 작성하는 전문가입니다.

    [Medical Terminology Definitions]
        1. 주소증 (Chief Complaints, CC): 환자가 의사를 찾는 주된 이유로, 환자가 느끼고 있는 증상이나 문제를 말합니다. 예를 들어, 통증, 발열, 호흡곤란 등 환자가 경험하는 가장 두드러진 증상들이 이에 해당됩니다.
        2. 발병일 (Date of Onset, DOO): 환자가 처음으로 증상을 경험한 날짜를 의미합니다. 이 정보는 진단 과정에서 중요한 역할을 하며, 증상의 급성 또는 만성 여부를 판단하는 데 도움이 됩니다.
        3. 과거력 (Past Medical History, PMH): 환자의 과거 건강 상태, 이전 질병, 수술 경험, 입원 이력, 알레르기 반응, 지속적으로 복용하고 있는 약물 등을 포함합니다. 이는 현재의 건강 상태를 이해하는 데 필수적인 정보입니다.
        4. 가족력 (Family History, FH): 환자의 직계 가족(부모, 형제자매, 자녀) 및 때때로 친척들의 건강 이력을 말합니다. 특정 질병이 가족 내에서 유전적으로 전달되는 경향이 있을 수 있으므로, 이 정보는 유전적 요인을 평가하는 데 중요합니다.
        5. 현병력 (Present Illness, PI): 현재 환자가 겪고 있는 질병의 전체적인 이야기를 말합니다. 이는 주요 증상의 발병부터 현재까지의 경과, 증상의 변화, 그리고 그 증상이 환자의 일상생활에 미치는 영향 등을 포함합니다.

    [Inference]
        주소증, 발병일, 과거력, 가족력, 현병력을 정리하는 데 사용한 추론 과정은 다음과 같습니다:

        1. 주소증 (Chief Complaints)
            1.1 **상담 내용 분석**: 환자가 호소하는 주된 건강 문제 및 증상을 식별합니다.
            1.2 **증상 식별**: 환자의 설명과 의사의 진단을 통해 구체적인 증상을 정리합니다.
            1.3 **문맥적 해석**: 증상의 중요성과 상황에 따른 의미를 평가합니다.

        2. 발병일 (Date of Onset)
            2.1 **직접적 언급 검색**: 환자나 의사가 언급한 구체적인 날짜나 사건을 찾습니다.
            2.2 **간접적 정보 고려**: 증상 발생과 관련된 간접적인 정보를 평가합니다.
            2.3 **정보 부재 시 인정**: 발병일에 대한 명확한 정보가 없는 경우 이를 명시합니다.

        3. 과거력 (Past Medical History)
            3.1 **치료 경험 파악**: 과거에 받은 치료나 시술에 대한 정보를 찾습니다.
            3.2 **치료 결과 평가**: 과거 치료의 효과나 반응을 분석합니다.
            3.3 **치료 경험의 연결**: 현재 상태와 과거 치료 사이의 연관성을 평가합니다.

        4. 가족력 (Family History)
            4.1 **직접적 언급 탐색**: 가족의 건강 문제에 대한 언급을 찾습니다.
            4.2 **간접적 정보 평가**: 가족력과 관련된 간접적인 정보를 고려합니다.
            4.3 **명확한 정보 부재 인식**: 가족력에 대한 정보가 없으면 이를 명시합니다.

        5. 현병력 (History of Present Illness)
            5.1 **증상의 발달 추적**: 현재 증상의 발생과 진행 과정을 분석합니다.
            5.2 **현재 치료 반응 평가**: 현재 치료에 대한 반응과 효과를 평가합니다.
            5.3 **일상생활 영향 평가**: 증상이 환자의 일상에 미치는 영향을 고려합니다.
            5.4 **심리적, 정서적 상태 고려**: 환자의 정서적, 심리적 상태와 증상과의 관계를 평가합니다.

    [Overall Rules to follow]
        1. Contents은 환자와 의사가 나눈 상담 내용입니다. 당신은 반드시 [[상담 내용]]을 기억해야 하고 [[상담 내용]]에 대한 정보에 대해서 질문을 받으면 친절하게 답변해야 한다.
        2. [Medical Terminology Definitions]는 주소증, 발병일, 과거력, 가족력, 현병력에 대해서 설명하고 있는데 너는 이 내용을 아주 자세히 알고 있는 의료 기록 전문가이다.
        3. 이 차팅에 대해서 학습해라. 질문은 학습이 끝나면 드리겠습니다. 그 때까지 대기하세요.
        4. 너는 앞으로 답변할 때 인사말, 추가적인 문구를 생성하지 않는다.
        5. 질문에 대한 내용만 답변하면 된다.

    '''
    output_list=[user_text_prompt_1, user_text_prompt_2]
    return output_list

def make_knowledge_prompt(oneshot_path: str, input_content):
    '''
    file_path : 원샷 파일 경로(json 파일 형식)
    '''
    
    with open(oneshot_path, 'r', encoding='utf-8') as file:
        one_shot = json.load(file)
    
    knowledge_prompt = []
    content = one_shot.get('content')
    summary = one_shot.get('summary')
    charting = one_shot.get('charting')
    evidence = one_shot.get('evidence')
    
    user_text_1 = f'''
    {input_content}

    ---

    여기까지가 상담내용이다. 각 순서대로 요약[summary]을 해줘.
    다음은 예시[Example]이다.

    ###Example###

    {content}

    {summary}
    '''

    user_text_2 = f'''
    이번에는 charting을 정리해줘. charting은 주소증, 발병일, 과거력, 가족력, 현병력에 대한 내용이 포함된 것을 의미한다.
    다음은 예시[Example]이다.

    ###Example###

    {charting}
    '''

    user_text_3 = f"""너가 앞의 대답(주소증, 발병일, 과거력, 가족력, 현병력)들을
    생성한 근거[evidence]를 앞의 상담내용에서 연관성이 가장 높은 1문장만 발췌해줘.
    다음은 예시[Example]이다.

    ###Example###

    {evidence}
    """
    
    knowledge_prompt.append(user_text_1)
    knowledge_prompt.append(user_text_2)
    knowledge_prompt.append(user_text_3)
    
    return knowledge_prompt


def make_basic_prompt():
    prompt = []
    user_text_1 = "의사들이 작성하는 차트(주소증, 발병일, 과거력, 가족력, 현병력)를 작성해줘."
    prompt.append(user_text_1)
    return prompt

def make_cot_prompt(input_content):
    '''
    전체 CoT 과정 중에 첫번째 단계에 사용되는 prompt를 생성하는 함수
    '''
    cot_prompt = init_prompt(input_content)
    cot_prompt.append(summary_prompt())
    for medical_term_num in range(1, 6):
        cot_prompt.append(inference_prompt(medical_term_num))
        cot_prompt.append(evidence_prompt(medical_term_num))

    return cot_prompt

### 함수명 변경
def launch_DialogFlow(file_path: str,
                    oneshot_path: str, 
                    llm_name='gemini',
                    prompt_type='basic') -> ConversationChain:
    with open(file_path, 'r', encoding='utf-8') as file:
        input_content = file.read()

    if llm_name == 'gpt3':
        my_model = call_gpt3p5()
    if llm_name == 'gpt4':
        my_model = call_gpt4p0()
    if llm_name == 'gemini':    
        my_model = call_gemini()
    
    my_chat = ConversationChain(llm=my_model)
    
    print('RUN...')
    if prompt_type == "basic":
        conversation_text = make_basic_prompt()
    elif prompt_type == "knowledge":
        conversation_text = make_knowledge_prompt(oneshot_path, input_content)
    elif prompt_type == "cot":
        conversation_text = make_cot_prompt(input_content)
    else:
        raise ValueError("잘못된 prompt type을 입력하셨습니다. basic, knowledge, cot 중에서 선택주세요.")
    
    for content in conversation_text:
        my_chat.run(content)
    
    return my_chat

def synthesize_CoT(my_chat: ConversationChain) -> ConversationChain:
    print('RUN synthesize_CoT')
    
    temp_0 = my_chat.memory.chat_memory.messages
    
    total_charting_text = f'''
    [주소증(Chief Complaints)]
    {temp_0[7].content}

    [발병일(Date of Onset)]
    {temp_0[9].content}

    [과거력(Past Medical History)]
    {temp_0[11].content}

    [가족력(Family History)]
    {temp_0[13].content}

    [현병력(Present Illness)]
    {temp_0[15].content}

    ---

    위의 문서들은 추론에 의해 도출된 결과물들이다. 주소증, 발병일, 과거력, 가족력, 현병력 순으로 종합적으로 정리해서 의사들이 작성하는 차트로 만들어줘.
    '''
    
    my_chat.run(total_charting_text)
    my_chat.run("의사들이 작성하는 SOAP 방식으로 정리해줘.")
    
    return my_chat

def basic_save_txt(save_path, save_file_name, my_chat):
    full_name = os.path.join(save_path, save_file_name)
    temp_1 = my_chat.memory.chat_memory.messages
    with open(full_name, "w", encoding='utf-8') as file:
        file.write("======차팅======\n\n")
        file.write(temp_1[-1].content)
        print(f"Complete Save : {full_name}")
        
    return full_name


def knowledge_save_txt(save_path, save_file_name, my_chat):
    full_name = os.path.join(save_path, save_file_name)
    temp_1 = my_chat.memory.chat_memory.messages
    with open(full_name, "w", encoding='utf-8') as file:
        file.write("======요약======\n\n")
        file.write(temp_1[1].content)
        file.write("\n\n======차팅======\n\n")
        file.write(temp_1[3].content)
        file.write("\n\n======근거 정리======\n\n")
        file.write(temp_1[5].content)
        print(f"Complete Save : {full_name}")
        
    return full_name

def cot_save_txt(save_path, save_file_name, my_chat):
    full_name = os.path.join(save_path, save_file_name)
    temp_1 = my_chat.memory.chat_memory.messages
    with open(full_name, "w", encoding='utf-8') as file:
        file.write("======요약======\n\n")
        file.write(temp_1[5].content)
        file.write("\n\n======차팅======\n\n")
        file.write(temp_1[-3].content)
        file.write("\n\n======SOAP======\n\n")
        file.write(temp_1[-1].content)
        file.write("\n\n======근거 정리======\n\n")
        file.write("\n### 주소증\n")
        file.write(temp_1[17].content)
        file.write("\n### 발병일\n")
        file.write(temp_1[19].content)
        file.write("\n### 과거력\n")
        file.write(temp_1[21].content)
        file.write("\n### 가족력\n")
        file.write(temp_1[23].content)
        file.write("\n### 현병력\n")
        file.write(temp_1[25].content)
        print(f"Complete Save : {full_name}")
    
    return full_name 

def save_txt(save_path, save_file_name, my_chat, prompt_type):
    if prompt_type == "basic":
        basic_save_txt(save_path, save_file_name, my_chat)
    elif prompt_type == "knowledge":
        knowledge_save_txt(save_path, save_file_name, my_chat)
    elif prompt_type == "cot":
        cot_save_txt(save_path, save_file_name, my_chat)
    else:
        raise ValueError("잘못된 prompt type을 입력하셨습니다. basic, knowledge, cot 중에서 선택주세요.")

def processing_charting(file_path: str,
                        oneshot_path: str,
                        llm_name: str,
                        prompt_type: str,
                        save_path: str,
                        save_file_name: str) -> get_openai_callback:
    
    with get_openai_callback() as cb:
        my_chat = launch_DialogFlow(file_path=file_path,
                                    oneshot_path=oneshot_path,
                                    llm_name=llm_name,
                                    prompt_type=prompt_type)
        if prompt_type == 'cot':
            my_chat = synthesize_CoT(my_chat)

    save_txt(save_path=save_path,
            save_file_name=save_file_name,
            my_chat=my_chat,
            prompt_type=prompt_type)
    print("==API callBack Cost==")
    print(cb)
    cb.self_total_cost = cal_cost(cb=cb, llm_name=llm_name)

    return cb

def cal_cost(cb: get_openai_callback,
            llm_name: str) -> float:
    cost_values = 0.0
    if llm_name == "gpt3":
        cost_values = cb.prompt_tokens * 0.0005/1000 + \
            cb.completion_tokens * 0.0015/1000
    elif llm_name == "gpt4":
        cost_values = cb.prompt_tokens * 0.01/1000 + \
            cb.completion_tokens * 0.03/1000
    elif llm_name == "gemini":
        cost_values = 0.0
    print("==Estimated Cost==")
    print(f"The cost is : ${cost_values}")
    return cost_values

if __name__ == '__main__':
    processing_charting(
        file_path="../data/stt_text/treatment/treatment_local_balance/봄한의원.txt",
        oneshot_path="../data/sample/sample_data_one_short_content.json",
        llm_name="gpt3",
        prompt_type='basic',
        save_path="../data/",
        save_file_name="temp_temp.txt"
        )
    
