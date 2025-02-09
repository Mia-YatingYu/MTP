import os, time
import openai
import yaml
from dotenv import load_dotenv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

data_path = './'
# read file path
label_ASKG_path = data_path+'classes_ASKG_hmdb51_1.yml'
# write file path
xprompt_path = data_path+'classes_xprompt_hmdb51_1.yml'

with open(label_ASKG_path,'r') as label_ASKG_file:
    data = yaml.load(label_ASKG_file, Loader=yaml.FullLoader)

# continue from last run
last = 0

comments_KG_text = {
    'label': 'action label',
    # 'obj_en_li': 'object entity list',
    'obj_rel_triples': 'object-object relation triples',
    'act_obj_triples': 'action-object relation triples',
    # 'sub_act_en_li': 'sub-action entity list',
    'act_rel_triples': 'action-action relation triples',
}
# one-shot prompt
one_shot_label = 'drinking beer'
one_shot_user = {
      "role": "user",
      "content": "action label: drinking beer\n\nobject-object relation triples:\n1 - <beer, poured into, glass>\n2 - <beer, is consumed through, mouth>\n3 - <beer, in, bottle>\n4 - <glass, held by, hand>\n5 - <glass, placed on, coaster>\n6 - <bottle, is opened with, bottle opener>\n\naction-object relation triples:\n1 - <drinking beer, involves, beer>\n2 - <drinking beer, uses, glass>\n3 - <drinking beer, requires, mouth>\n4 - <drinking beer, needs, hand>\n5 - <drinking beer, utilizes, coaster>\n6 - <drinking beer, needs, bottle opener>\n7 - <drinking beer, involves, bottle>\n\naction-action relation triples:\n1 - <drinking beer, starts with, opening bottle>\n2 - <opening bottle, precedes, pouring beer>\n3 - <pouring beer, precedes, picking up the glass>\n4 - <picking up the glass, precedes, tilting the glass>\n5 - <tilting the glass, precedes, swallowing the beer>\n6 - <swallowing the beer, precedes, placing the glass back on the coaster>"
    }
one_shot_assistant = {
      "role": "assistant",
      "content": "# This is an example of drinking beer...\ndrinking beer:\n  label: drinking beer\n  xprompt_oo:\n    - where beer is poured into a glass\n    - where beer is consumed through the mouth\n    - where beer is contained in a bottle\n    - where a glass is held by the hand\n    - where a glass is placed on a coaster\n    - where a bottle is opened with a bottle opener\n  xprompt_ao:\n    - which involves beer\n    - which uses a glass\n    - which requires a mouth\n    - which needs a hand\n    - which utilizes a coaster\n    - which needs a bottle opener\n    - which involves a bottle\n  xprompt_aa:\n    - starting with opening a bottle\n    - where opening the bottle precedes pouring the beer\n    - where pouring the beer comes before picking up the glass\n    - where picking up the glass comes before tilting it\n    - where tilting the glass precedes swallowing the beer\n    - where swallowing the beer comes before placing the glass back on the coaster"
    }

with open(xprompt_path,'a+') as xprompt_file:
    for no, kv in enumerate(data.values()):
        if no >= last:
            user_input = ''
            for k,v in kv.items():
                if k in comments_KG_text.keys():
                    key = comments_KG_text[k]
                    if isinstance(v, str) or v is None:
                        user_input+=f'{key}: {v}'
                        # print(f'{key}: {v}')
                    elif isinstance(v, list):
                        user_input+=f'{key}:'
                        # print(f'{key}:')
                        for i,vi in enumerate(v):
                            user_input+=f'\n{i+1} - {vi}'
                            # print(i)
                    else:
                        user_input+=f'{key}: '
                        # print(f'{key}: ')
                    user_input+='\n\n'
                    # print()
            # print(user_input)
            start_time = time.time()
            response = chat_completion_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a commonsense knowledge base, especially for human actions. \nYou will be provided with some relation triples related to the action label below.\nUse the following step-by-step instructions to respond to user inputs:\n1 - Try to complete the whole sentence: \"This is an example of {given action label}, ...\", according to each relation triple.\n2 - There is a one-to-one correspondence between each generated completion and each relation triple.\n3 - You should avoid the presence of the action label in the generated completions, and only include objects (and subjects) and sub-actions in the triples.\n4 - The generated completions after the comma is all  we need. You should only output the content after the comma, instead of giving the whole sentence. Do not output \"This is an examples of {}\"\n 5 - Output the final answers in YAML format (with comments), reduce other prose. \n\nDesired format:\nShould include all these fields: \n[label (comments: action label), \nxprompt_oo (comments: object-object relation prompts), \nxprompt_ao (comments: action-object relation prompts), \nxprompt_aa (comments: action-action relation prompts)], under the root field \"action label name\"."
                    },
                    one_shot_user,
                    one_shot_assistant,
                    {
                        "role": "user",
                        "content": user_input.strip()
                    }
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            ans_time = time.time()
            consume_time = ans_time - start_time
            content = response.choices[0]["message"]["content"].strip()
            xprompt_file.write(content+'\n\n')

            print(content)
            print(f"##No.{no+1} time consuming : %.3f s##" % consume_time)
