import time
from typing import List, Tuple, Dict
import threading
from fastchat.model import get_conversation_template


class LanguageModel:
    def __init__(self, model_name, max_n_tokens=1400, temperature=0.1, top_p=1.0):
        self.max_n_tokens = max_n_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.roles = ["user", "assistant"]
        _, self.template = get_model_path_and_template(model_name)

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float,
                                   is_get_attention: bool = False):
        """
        Generates response by multi-threads for each requests
        """
        raise NotImplementedError



from openai import OpenAI
class CommercialAPI(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20



    def direct_response(self, conv, prompt):
        conv.append_message(conv.roles[0], prompt)
        final_prompt = conv.to_openai_api_messages()
        return self.generate(final_prompt, max_n_tokens=3072, temperature=0.1, top_p=1.0)




    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''

        output = self.API_ERROR_OUTPUT
        if "gpt" in self.model_name:

            client = OpenAI(
                base_url="xxx",
                api_key="xxx"
            )
        elif "local" in self.model_name:
            client = OpenAI(api_key='TOKEN',
                            base_url='xxx')

        else:
            client = OpenAI(api_key='xxx',
                            base_url='xxx')

        def conversation_to_messages(conv):
            return [
                {"role": message[0].strip('<|>'), "content": message[1]}
                for message in conv.messages
            ]


        for _ in range(self.API_MAX_RETRY):
            try:
            # if True:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation_to_messages(conv) if not isinstance(conv, list) else conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # output = response["choices"][0]["message"]["content"]
                output = response.choices[0].message.content
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

    def batched_generate_by_thread(self,
                                      convs_list: List[List[Dict]],
                                      max_n_tokens: int,
                                      temperature: float,
                                      top_p: float = 1.0,
                                      is_get_attention: bool = False):
          # multi-threading
          threads = []
          results = []
          attentions = []

          # each thread handles one conversation and save the result to results
          def thread_func(conv, max_n_tokens, temperature, top_p):
                result = self.generate(
                 conv, max_n_tokens, temperature, top_p)
                results.append(result)

          for conv in convs_list:
                thread = threading.Thread(target=thread_func, args=(
                 conv, max_n_tokens, temperature, top_p))

                time.sleep(1)
                threads.append(thread)
                thread.start()

          for thread in threads:
                thread.join()

          return results, attentions





def get_model_path_and_template(model_name):
    """
    TODO: Add the template you want.
    """
    full_model_dict = {
        "vicuna-api": {
            "path": "vicuna_v1.1",
            "template": "vicuna_v1.1"
        },
        "llama2-api": {
            "path": "llama-2",
            "template": "llama-2"
        },
        "chatglm-api": {
            "path": "chatglm",
            "template": "chatglm"
        },
        "chatglm2-api": {
            "path": "chatglm-2",
            "template": "chatglm-2"
        },
        "phi2-api": {
            "path": "phi2",
            "template": "phi2"
        },
        "zephyr-api": {
            "path": "zephyr",
            "template": "zephyr"
        },
        "baichuan-api": {
            "path": "baichuan2-chat",
            "template": "baichuan2-chat"
        },
        "one-shot": {
            "path": "one_shot",
            "template": "one_shot"
        },
        "zhipu": {
            "path": "zhipu",
            "template": "zhipu"
        },
        "douyin": {
            "path": "douyin",
            "template": "douyin"
        },
        "wenxinyiyan": {
            "path": "wenxinyiyan",
            "template": "wenxinyiyan"
        },
        "kuaishou": {
            "path": "kuaishou",
            "template": "kuaishou"
        },
        "baichuan": {
            "path": "baichuan",
            "template": "baichuan"
        },
        "zero-shot": {
            "path": "zero_shot",
            "template": "zero_shot"
        },
        "airoboros-1": {
            "path": "airoboros_v1",
            "template": "airoboros_v1"
        },
        "airoboros-2": {
            "path": "airoboros_v2",
            "template": "airoboros_v2"
        },
        "airoboros-3": {
            "path": "airoboros_v3",
            "template": "airoboros_v3"
        },
        "koala-1": {
            "path": "koala_v1",
            "template": "koala_v1"
        },
        "alpaca": {
            "path": "alpaca",
            "template": "alpaca"
        },
        "chatglm": {
            "path": "chatglm",
            "template": "chatglm"
        },
        "chatglm-2": {
            "path": "chatglm-2",
            "template": "chatglm-2"
        },
        "dolly-v2": {
            "path": "dolly_v2",
            "template": "dolly_v2"
        },
        "oasst-pythia": {
            "path": "oasst_pythia",
            "template": "oasst_pythia"
        },
        "oasst-llama": {
            "path": "oasst_llama",
            "template": "oasst_llama"
        },
        "tulu": {
            "path": "tulu",
            "template": "tulu"
        },
        "stablelm": {
            "path": "stablelm",
            "template": "stablelm"
        },
        "baize": {
            "path": "baize",
            "template": "baize"
        },
        "chatgpt": {
            "path": "chatgpt",
            "template": "chatgpt"
        },
        "bard": {
            "path": "bard",
            "template": "bard"
        },
        "falcon": {
            "path": "falcon",
            "template": "falcon"
        },
        "baichuan-chat": {
            "path": "baichuan_chat",
            "template": "baichuan_chat"
        },
        "baichuan2-chat": {
            "path": "baichuan2_chat",
            "template": "baichuan2_chat"
        },
        "falcon-chat": {
            "path": "falcon_chat",
            "template": "falcon_chat"
        },
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "gpt-4-turbo": {
            "path": "gpt-4-turbo",
            "template": "gpt-4-turbo"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo",
            "template": "gpt-3.5-turbo"
        },
        "text-davinci-003": {
            "path": "text-davinci-003",
            "template": "text-davinci-003"
        },
        "gpt-3.5-turbo-instruct": {
            "path": "gpt-3.5-turbo-instruct",
            "template": "gpt-3.5-turbo-instruct"
        },
        "vicuna": {
            "path": "vicuna-api",
            "template": "vicuna_v1.1"
        },
        "llama2-chinese": {
            "path": "llama2_chinese",
            "template": "llama2_chinese"
        },
        "llama-2": {
            "path": "llama2-api",
            "template": "llama-2"
        },
        "claude-instant-1": {
            "path": "claude-instant-1",
            "template": "claude-instant-1"
        },
        "claude-2": {
            "path": "claude-2",
            "template": "claude-2"
        },
        "palm-2": {
            "path": "palm-2",
            "template": "palm-2"
        },
        "gpt-4o-mini-2024-07-18": {
            "path": "gpt-4o-mini-2024-07-18",
            "template": "gpt-4o-mini-2024-07-18"
        },
        "claude-3-haiku-20240307":{
            "path": "claude-3-haiku-20240307",
            "template": "claude-3-haiku-20240307"
        },
        "gpt-4o-mini": {
            "path": "gpt-4o-mini",
            "template": "gpt-4o-mini"
        },

    }


    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template



def conv_template(template_name):
    template = get_conversation_template(template_name)
    return template


