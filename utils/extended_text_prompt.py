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

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")


data_path = './'
# ASKG dataset path
en_rel_path = data_path + 'classes_ASKG_hmdb51.yml'

# dataset label path
ucf101_label_path = data_path + 'classes_label_ucf101.yml'
hmdb51_label_path = data_path + 'classes_label_hmdb51.yml'
k400_label_path = data_path + 'classes_label_k400.yml'

# load the dataset label
with open(hmdb51_label_path,'r') as hmdb51_label_file:
  info = yaml.load(hmdb51_label_file, Loader=yaml.FullLoader)
  classes = info['classes']

# continue from last run
classes = classes[:]

# one-shot prompt: drinking beer
one_shot_label = 'drinking beer'
## user prompt
one_shot_user = {
  "role": "user",
  "content": "action entity: ["+ one_shot_label+ "]"
}
## assistant prompt
one_shot_assistant = {
      "role": "assistant",
      "content": "drinking beer:\n  label: drinking beer\n  obj_en_li: \n  - beer\n  - glass\n  - mouth\n  - hand\n  - coaster\n  - bottle opener\n  - bottle\n  obj_rel_triples: \n  - <beer, poured into, glass>\n  - <beer, is consumed through, mouth>\n  - <beer, in, bottle>\n  - <glass, held by, hand>\n  - <glass, placed on, coaster>\n  - <bottle, is opened with, bottle opener>\n  act_obj_triples: \n  - <drinking beer, involves, beer>\n  - <drinking beer, uses, glass>\n  - <drinking beer, requires, mouth>\n  - <drinking beer, needs, hand>\n  - <drinking beer, utilizes, coaster>\n  - <drinking beer, needs, bottle opener>\n  - <drinking beer, involves, bottle>\n  sub_act_en_li:\n  - opening bottle\n  - pouring beer\n  - picking up the glass\n  - tilting the glass\n  - swallowing the beer\n  - placing the glass back on the coaster\n  act_rel_triples: \n  - <drinking beer, starts with, opening bottle>\n  - <opening bottle, precedes, pouring beer>\n  - <pouring beer, precedes, picking up the glass>\n  - <picking up the glass, precedes, tilting the glass>\n  - <tilting the glass, precedes, swallowing the beer>\n  - <swallowing the beer, precedes, placing the glass back on the coaster>"
}

# save the content in yaml format
with open(en_rel_path,'a+') as en_rel_file:
  for c in classes:
    start_time = time.time()
    response = chat_completion_with_backoff(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "You are a commonsense knowledge base, especially for human actions. \n\nYou will be provided with an action entity name  below, which is delimited with square brackets. \n\nUse the following step-by-step instructions to respond to user inputs:\n\n1 - Return the object entity list contained Top K most relevant objects involved in the given action (5<=K<=10).\n\n2 - What are the relations among these object entities? Find the proper predicate names that concisely describing the relationship between each object pair chosen from the object entity list.\n\n3 - What are the relations between the given action entity and these object entities? Choose the proper predicate names that concisely describing the relationship between the given action entity and each object entity listed above.\n\n4 - What sub-actions does the given action entity involve?  Return each sub-action name in the right processing order.\n\n5 - Generate the action category info based on the instructions above in YAML format (with comments), reduce other prose. \nShould include these fields: \n[label (i.e., the given action name), \nobj_en_li (i.e., object entity list), \nobj_rel_triples (i.e., object-object relation triples), \nact_obj_triples (i.e., action-object relation triples), \nsub_act_en_li (i.e., sub-action entity list),\nact_rel_triples (i.e., action-action relation triples)], under the root \"given action name\"."
        },
        one_shot_user,
        one_shot_assistant,
        {
          "role": "user",
          "content": "action entity: [" + c.lower() + "]"
        }
      ],
      temperature=0,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    ans_time = time.time()
    consume_time = ans_time - start_time
    content = response.choices[0]["message"]["content"].strip()
    en_rel_file.write(content+'\n\n')

    print(content)
    print("##time consuming : %.3f s##" % consume_time)

