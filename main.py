import json
import re
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import logging
import time
import functools
import logging
from typing import List, Tuple, Dict
import requests
from tqdm import tqdm
from functools import lru_cache
from language_models import conv_template,CommercialAPI
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import os
import numpy as np
import torch
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StanceEnhancer:
    def __init__(self, api_key: str):
        """
        初始化stance enhancer
        Args:
            api_key: API密钥
        """
        self.api_key = api_key
        self.api_url = "xxx"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    @lru_cache(maxsize=1024)
    def _get_embedding(self, text: str) -> List[float]:
        payload = {
            "model": "Pro/BAAI/bge-m3",
            "encoding_format": "float",
            "input": text
        }
        
        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return None
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        if not emb1 or not emb2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
        
    def calculate_stance_entropy1(self, text: str, target: str, language='en') -> float:
        if language == 'zh':
            stance_templates = [
                f"我对{target}的态度是中立",
                f"我对{target}的态度是反对",
                f"我对{target}的态度是支持"
            ]
        elif language == 'en':
            stance_templates = [
                f"My stance towards {target} is neutral",
                f"My stance towards {target} is against",
                f"My stance towards {target} is in favor"
            ]
        
        text_embedding = self._get_embedding(text)
        if not text_embedding:
            return 0.0
        
        similarities = []
        for template in stance_templates:
            template_embedding = self._get_embedding(template)
            if template_embedding:
                similarity = self._calculate_similarity(text_embedding, template_embedding)
                similarities.append(similarity)
            
        return np.std(similarities) if similarities else 0.0

    def calculate_stance_entropy(self, text: str, target: str, language='en') -> float:
        if language == 'zh':
            stance_templates = [
                f"我对{target}的态度是中立",
                f"我对{target}的态度是反对",
                f"我对{target}的态度是支持"
            ]
        elif language == 'en':
            stance_templates = [
                f"My stance towards {target} is neutral",
                f"My stance towards {target} is against",
                f"My stance towards {target} is in favor"
            ]

        text_embedding = self._get_embedding(text)
        if not text_embedding:
            return 0.0

        similarities = []
        for template in stance_templates:
            template_embedding = self._get_embedding(template)
            if template_embedding:
                similarity = self._calculate_similarity(text_embedding, template_embedding)
                similarities.append(similarity)

        if not similarities:
            return 0.0

        # Normalize similarities to form a probability distribution
        total_similarity = sum(similarities)
        probabilities = [sim / total_similarity for sim in similarities]

        # Calculate entropy (positive version to match standard deviation logic)
        entropy = sum(p * np.log2(p) for p in probabilities if p > 0)  # Avoid log(0)

        return entropy

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？!?]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_target_relevance(self, sentence: str, target: str, language='en') -> float:
        if language == "zh":
            target_templates = [
                target,
                f"关于{target}",
                f"谈论{target}",
                f"{target}相关"
            ]
        elif language == "en":
            target_templates = [
                target,
                f"about {target}",
                f"discussing {target}",
                f"{target} related"
            ]
        
        sentence_emb = self._get_embedding(sentence)
        if not sentence_emb:
            return 0.0
            
        max_similarity = 0.0
        for template in target_templates:
            template_emb = self._get_embedding(template)
            if template_emb:
                similarity = self._calculate_similarity(sentence_emb, template_emb)
                max_similarity = max(max_similarity, similarity)
                
        return max_similarity
    
    def enhance_texts_with_knowledge(self, 
                                   texts: List[str], 
                                   knowledge_json_file: str, 
                                   target: str,
                                   language: str = "en",
                                   relevance_threshold: float = 0.6,
                                   ) -> List[str]:
        if os.path.exists(knowledge_json_file):
            with open(knowledge_json_file, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
        else:
            logger.error(f"Knowledge JSON 文件 {knowledge_json_file} 不存在。")
            return texts
            
        terms_dict = {item['term']: item['explanation'] for item in knowledge_data}
        
        enhanced_texts = []
        for text in tqdm(texts, desc="Processing texts"):
            original_entropy = self.calculate_stance_entropy(text, target)
            best_text = text
            best_entropy = original_entropy
            best_knowledge = []
            
            for term, explanation in terms_dict.items():
                if language == "en":
                    pattern = r'\b{}\b'.format(re.escape(term))
                else:
                    pattern = re.escape(term)
                    
                if re.search(pattern, text):
                    knowledge_text = f"{text}\n=== context ===\n{term}: {explanation}"
                    current_entropy = self.calculate_stance_entropy(knowledge_text, target)
                    
                    if current_entropy > best_entropy:
                        best_entropy = current_entropy
                        best_knowledge.append((term, explanation))
            
            sentences = self._split_sentences(text)
            if not sentences:
                enhanced_texts.append(best_text)
                continue
                
            relevance_scores = []
            for i, sentence in enumerate(sentences):
                relevance = self._calculate_target_relevance(sentence, target)
                if relevance >= relevance_threshold:
                    relevance_scores.append((i, relevance))
            
            if not relevance_scores:
                enhanced_texts.append(best_text)
                continue
                
            relevance_scores.sort(key=lambda x: x[1], reverse=True)
            best_subset = []
            best_subset_entropy = best_entropy
            
            for k in range(1, len(relevance_scores) + 1):
                current_indices = [x[0] for x in relevance_scores[:k]]
                current_indices.sort()
                
                current_text = ""
                last_idx = -1
                for idx in current_indices:
                    if last_idx == -1:
                        current_text = "..."
                    elif idx - last_idx > 1:
                        current_text += "..."
                    current_text += sentences[idx]
                    last_idx = idx
                current_text += "..."
                
                if best_knowledge:
                    knowledge_str = "\n=== context ===\n" + "|".join(
                        [f"{term}: {exp}" for term, exp in best_knowledge]
                    )
                    current_text += knowledge_str
                
                current_entropy = self.calculate_stance_entropy(current_text, target)
                
                if current_entropy > best_subset_entropy:
                    best_subset_entropy = current_entropy
                    best_subset = current_indices
            
            if best_subset:
                final_text = ""
                last_idx = -1
                for idx in best_subset:
                    if last_idx == -1:
                        final_text = "..."
                    elif idx - last_idx > 1:
                        final_text += "..."
                    final_text += sentences[idx]
                    last_idx = idx
                final_text += "..."
                
                if best_knowledge:
                    knowledge_str = "\n=== context ===\n" + "\n".join(
                        [f"{term}: {exp}" for term, exp in best_knowledge]
                    )
                    final_text += knowledge_str
            else:
                final_text = best_text
                
            enhanced_texts.append(final_text)
            
        return enhanced_texts

    def process_batch(self, 
                     texts: List[str], 
                     knowledge_json_file: str,
                     target: str,
                     batch_size: int = 32,
                     language: str = "zh",
                     relevance_threshold: float = 0.3) -> List[str]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.enhance_texts_with_knowledge(
                batch, 
                knowledge_json_file, 
                target, 
                language,
                relevance_threshold
            )
            results.extend(batch_results)
        return results

class InvalidOutputError(Exception):
    pass

def retry_on_invalid_output(max_retries=3, delay=2, backoff=2):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            _delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except InvalidOutputError as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts due to invalid output.")
                        raise
                    else:
                        logger.warning(f"Invalid output on attempt {attempt} for function {func.__name__}. Retrying in {_delay} seconds...")
                        time.sleep(_delay)
                        _delay *= backoff

        return wrapper_retry

    return decorator_retry

class TextProcessor:
    def __init__(self, models: List[str]):
        self.models = models
        self.label2id = {
            "支持": 2, "中立": 1, "反对": 0,
            "FAVOR": 2, "NONE": 1, "AGAINST": 0,
            "favor": 2, "neutral": 1, "against": 0
        }
        self.enhancer = StanceEnhancer(api_key="api_key")

    def extract_jsonl(self, output: str) -> List[Dict]:
        try:
            json_str = re.search(r'\[.*\]', output, re.DOTALL).group()
            data = json.loads(json_str)
            return data
        except (AttributeError, json.JSONDecodeError):
            return []

    def extract_json(self, output: str) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return {}
        except json.JSONDecodeError:
            return None



    def call_api_with_retry(self, lm, conv, prompt: str) -> List[Dict]:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                conv_temp = conv.copy()
                output = lm.direct_response(conv_temp, prompt).strip()
                print(f"Model Output on attempt {attempt + 1}")
                data = self.extract_json(output)
                if not data:
                    raise InvalidOutputError(f"Failed to extract JSON from output.")
                return data
            except InvalidOutputError as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

    def distribution_similarity(self, dist1, dist2):
        return np.dot(dist1, dist2) / (np.linalg.norm(dist1) * np.linalg.norm(dist2))

    def enhance_texts_with_keywords(self, texts: List[str], knowledge_json_file: str, language: str = "zh") -> List[str]:
        if os.path.exists(knowledge_json_file):
            with open(knowledge_json_file, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
        else:
            logger.error(f"Knowledge JSON 文件 {knowledge_json_file} 不存在。")
            return texts

        terms_dict = {item['term']: item['explanation'] for item in knowledge_data}

        enhanced_texts = []
        for text in texts:
            matched_terms = []
            for term, explanation in terms_dict.items():
                if language == "en":
                    # 英文完全匹配，使用单词边界
                    pattern = r'\b{}\b'.format(re.escape(term))
                else:
                    # 中文直接查找子串
                    pattern = re.escape(term)
                if re.search(pattern, text):
                    matched_terms.append((term, explanation))

            if matched_terms:
                context_pieces = []
                for term, explanation in matched_terms:
                    context_str = f"{term}: {explanation}"
                    context_pieces.append(context_str)
                additional_context = "\n=== context ===\n" + "|".join(context_pieces)
                enhanced_text = text + additional_context
            else:
                enhanced_text = text
            enhanced_texts.append(enhanced_text)
        return enhanced_texts



class KnowledgeExtractor(TextProcessor):
    def __init__(self, models: List[str], checkpoint_file: str):
        super().__init__(models)
        self.checkpoint_file = checkpoint_file
        self.knowledge_data = self.load_checkpoint()
        # 可以在这里添加特定于 KnowledgeExtractor 的初始化代码


    def load_checkpoint(self) -> List[Dict]:
        """
        从 checkpoint JSON 文件加载已保存的知识数据，如果文件存在。
        :return: 已保存的知识数据（列表格式），如果不存在则返回空列表。
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.error("Checkpoint 文件格式错误，无法解析。")
                    return []
        # 初始化为空列表
        return []

    def save_checkpoint(self):
        """
        将当前的知识数据保存到 checkpoint JSON 文件中。
        """
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Checkpoint 数据已保存到 {self.checkpoint_file}")

    def extract_knowledge(self, model: str, texts: List[str], target: str, output_json_file: str, language: str = "zh", batch_size=4) -> None:
        """
        从输入的文本列表中提取知识，并保存到 JSON 文件中。
        """
        lm = CommercialAPI(model)

        # 根据语言选择 Prompt
        if language == "zh":
            knowledge_prompt_template = '''
                请阅读以下与目标 "{target}" 相关的文本：

                {texts}

                根据这些文本，请提取以下类型的信息，并按照指定格式返回：

                1. **关键词**：与目标相关的重要词汇或短语，这些词汇在表达情感或立场时具有代表性。请确保这些关键词与文本中的术语完全一致，不进行任何替代、扩展或修改。

                2. **事件**：与目标相关的重大事件或活动。事件名称应与文本中的描述完全一致。

                3. **近期新闻**：最近发生的、与目标相关的新闻报道。新闻标题和内容应与文本中的信息保持一致。

                4. **社交媒体讨论**：在社交媒体上关于目标的讨论热点或话题。讨论内容应与原文描述一致。

                对于每个提取的信息，请提供详细的解释，包括其重要性和与目标的关系。

                请以以下 JSON 格式返回结果，其中每个条目都是一个字典，包含 "term"、"explanation" 和 "catalogue" 键：

                [
                    {
                        "term": "关键词1",
                        "explanation": "关键词1的详细解释",
                        "catalogue": "keyword"
                    },
                    {
                        "term": "事件1",
                        "explanation": "事件1的详细解释",
                        "catalogue": "event"
                    },
                    {
                        "term": "新闻1",
                        "explanation": "新闻1的详细解释",
                        "catalogue": "recent_news"
                    },
                    {
                        "term": "讨论1",
                        "explanation": "讨论1的详细解释",
                        "catalogue": "social_media_discussion"
                    },
                    ...
                ]

                请**仅**输出上述 JSON 数据，使用双引号括起所有的键和值，确保 JSON 格式正确，可以被标准的 JSON 解析器解析，不要包含任何额外的文本。
                '''
        else:
            # 英文 Prompt
            knowledge_prompt_template = '''
                Please read the following texts related to the target "{target}":

                {texts}

                Based on these texts, please extract the following types of information and return them in the specified format:

                1. **Keywords**: Important words or phrases related to the target that are representative in expressing emotions or stances. Please ensure that these keywords exactly match the terms in the text without any substitution, expansion, or modification.

                2. **Events**: Significant events or activities related to the target. Event names should exactly match the descriptions in the text.

                3. **Recent News**: Recent news reports related to the target. News titles and contents should align precisely with the information in the text.

                4. **Social Media Discussions**: Hot topics or discussions about the target on social media. Discussion content should be consistent with the descriptions in the text.

                For each piece of information extracted, please provide a detailed explanation, including its importance and relationship with the target.
                
                For example, the following samples you should consider: Braidleigh, BetOnRed, PatriotsWillRise, SpankAFeminist

                Please return the results in the following JSON format, where each entry is a dictionary containing "term", "explanation", and "catalogue" keys:

                [
                    {{
                        "term": "Keyword1",  
                        "explanation": "Detailed explanation of Keyword1",
                        "catalogue": "keyword"
                    }},
                    {{
                        "term": "Event1",
                        "explanation": "Detailed explanation of Event1",
                        "catalogue": "event"
                    }},
                    {{
                        "term": "News1",
                        "explanation": "Detailed explanation of News1",
                        "catalogue": "recent_news"
                    }},
                    {{
                        "term": "Discussion1",
                        "explanation": "Detailed explanation of Discussion1",
                        "catalogue": "social_media_discussion"
                    }},
                    ...
                ]

                Please **only** output the above JSON data, using double quotes for all keys and string values, ensuring the JSON format is correct and can be parsed by standard JSON parsers. Do not include any additional text.
                '''

        total_batches = math.ceil(len(texts) / batch_size)
        all_knowledge_data = []



        def process_batch(batch_texts, batch_idx):
            """
            单个批次的处理函数。
            """
            texts_str = '\n'.join(batch_texts)
            knowledge_prompt = knowledge_prompt_template.format(target=target, texts=texts_str)

            # try:
            if True:

                retry_count = 0
                max_retries = 3
                knowledge_output = None

                while retry_count < max_retries:
                    try:
                        conv = conv_template(lm.template)
                        knowledge_output = lm.direct_response(conv, knowledge_prompt).strip()
                        print(knowledge_output)
                        new_knowledge_data = self.extract_jsonl(knowledge_output)
                        
                        if new_knowledge_data is None:
                            raise InvalidOutputError(f"Failed to extract JSON from the model output for batch {batch_idx + 1}.")
                        break  # 成功提取数据，跳出重试循环
                    except Exception as e:
                        retry_count += 1
                        print(f"Retry {retry_count}/{max_retries} for batch {batch_idx + 1} due to error: {e}")
                        if retry_count == max_retries:
                            raise InvalidOutputError(f"Failed to extract JSON after {max_retries} attempts for batch {batch_idx + 1}.")

                # 删除 new_knowledge_data[i]['term'] 中单词>3的，且总长度<=20的（英文）
                if len(new_knowledge_data)==0:
                    return []
                new_knowledge_data = [item for item in new_knowledge_data if
                                      len(item['term'].split()) <= 3 and len(item['term']) <= 20]
                
                return new_knowledge_data


        # 使用 ThreadPoolExecutor 进行多线程处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_batch, texts[batch_idx * batch_size: (batch_idx + 1) * batch_size],
                                batch_idx): batch_idx
                for batch_idx in range(total_batches)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                # if True:
                    batch_result = future.result()
                    # print(batch_result)
                    all_knowledge_data.extend(batch_result)

                except Exception as e:
                    print(f"Error processing batch {batch_idx + 1}: {e}")


        # 合并所有批次提取的知识数据
        self.update_knowledge_data(all_knowledge_data, model, language)

        # 保存 checkpoint
        self.save_checkpoint()

        # 保存最终结果
        self.save_final_output(output_json_file)

    def update_knowledge_data(self, new_data: List[Dict], model: str, language: str):
        """
        合并新提取的知识数据到现有的知识数据中，并处理冗余。
        """
        # 创建一个 term 到 explanation 的映射，便于快速查找
        existing_terms = {item['term']: item for item in self.knowledge_data}

        for item in new_data:
            term = item.get('term')
            explanation = item.get('explanation')
            catalogue = item.get('catalogue')

            if term in existing_terms:
                # 如果 term 已存在，比较解释
                existing_expl = existing_terms[term]['explanation']
                better_expl = self.compare_explanations(term, existing_expl, explanation, model, language)
                existing_terms[term]['explanation'] = better_expl
            else:
                # 添加新项
                self.knowledge_data.append({
                    'term': term,
                    'explanation': explanation,
                    'catalogue': catalogue
                })
                existing_terms[term] = self.knowledge_data[-1]  # 更新映射


    def compare_explanations(self, term: str, expl1: str, expl2: str, model: str, language: str) -> str:
        """
        使用大模型比较两个解释，返回更好的一个。
        """
        lm = CommercialAPI(model)
        if language == "zh":
            compare_prompt = f'''
            针对术语“{term}”有两个解释：

            解释1：{expl1}

            解释2：{expl2}

            请判断哪个解释更全面、准确，有助于理解针对目标的情感和立场。如果解释1更好，返回“1”；如果解释2更好，返回“2”。
            '''
        else:
            compare_prompt = f'''
            There are two explanations for the term "{term}":

            Explanation 1: {expl1}

            Explanation 2: {expl2}

            Please determine which explanation is more comprehensive, accurate, and helpful in understanding the sentiment and stance towards the target. Return "1" if Explanation 1 is better, or "2" if Explanation 2 is better.
            '''

        try:
            conv = conv_template(lm.template)
            compare_output = lm.direct_response(conv, compare_prompt).strip()
            better_option = re.search(r'\b(1|2)\b', compare_output)
            if better_option:
                return expl1 if better_option.group() == "1" else expl2
            else:
                # 如果无法判断，默认保留第一个解释
                return expl1
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            # 如果比较失败，默认保留第一个解释
            return expl1

    def save_final_output(self, output_json_file: str):
        """
        将最终的知识数据保存到指定的 JSON 文件。
        """


        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_data, f, ensure_ascii=False, indent=4)
        logger.info(f"完整的知识数据已保存到 {output_json_file}")



class StanceDetector(TextProcessor):
    def __init__(self, models: List[str]):
        super().__init__(models)
        self.models = models

        self.label2id = {
            "支持": 2, "中立": 1, "反对": 0,
            "FAVOR": 2, "NONE": 1, "AGAINST": 0,
            "favor": 2, "neutral": 1, "against": 0
        }


    def process_batch_item(self, model: str, data_batch, target: str, true_labels: list, language: str = "en", zero_shot: bool = False, knowledge_base = './datasets/knowledge_data.json', class_num=3) -> Dict:
        """
        Batch process stance detection for multiple texts.

        Args:
            model (str): Model identifier
            data_batch: Batch of data containing texts
            target (str): Target for stance detection
            true_labels (list): List of true labels for evaluation
            language (str): Language of the texts (default: "en")
            zero_shot (bool): Whether to use zero-shot approach (default: False)

        Returns:
            Dict: Dictionary containing results of stance detection
        """
        lm = CommercialAPI(model)

        # Extract texts from data batch
        texts = data_batch.get('text', data_batch.get('text_a')).tolist()
        target = data_batch.get('Target', data_batch.get('target')).tolist()
        if len(target)==1:
            target = target[0]
        

        # Prepare prompt based on language
        if class_num==3:
            if language == "en":
                if isinstance(target, list):
                    '''(**Note that they could belong to the same stance**)'''
                    # print("hello world!")
                    batch_detection_prompt = '''You are a stance detection assistant. Given the following list of texts and their corresponding targets, please analyze the stance each text potentially holds toward its corresponding target. Pay special attention to subtle or ironic expressions of stance and their contexts.

                    Texts and Targets:
                    {texts}

                    Please evaluate whether each text holds a favor, against, or (irrelevant/neutral/no stance expressed) stance toward its corresponding target.
                    Analytical reference:
                    {{
                    Target: christians
                    Text: This discussion is a classic NYT attempt to change the narrative. The shooter was a muslim who swore allegiance to ISIS. Last I checked, no mainstream Christian religion advocates killing gays. The massacre is not the responsibility of Christians, the NRA, or Republicans, but rather a radical Islamic ideology. But neither the NYT or the President can apparently say that.
                    Stance: Neutral
                    Explanation: The text distinguishes mainstream Christian ideology from the radical Islamic ideology blamed for the violence. It clarifies that Christians are not responsible, without expressing explicit support, thus maintaining a neutral stance by delineating ideological differences.
                    }}
                    {{
                    Target: guns
                    Text: Military and police, yes. They're sworn and trained to uphold the country and the laws of the country. The rest is an ego trip. As for hunters, Ogden Nash said it best, '---------------------------the hunter with pluck and luck is trying to outwit a duck'.
                    Stance: Against
                    Explanation: The text implies a negative view toward non-official use of guns, describing it as an "ego trip" or trivial pursuit, aligning with a critical stance on guns.
                    }}
                    {{
                    Target: stability
                    Text: Tenure does not mean a teacher cannot lose their job. It requires due process before termination. Before tenure is achieved, a teacher can be fired without due process. In the Atlanta School District administrators, fearing that low test scores would cost them their jobs, instructed teachers to change student test responses. Without tenure and due process, teachers risked being fired if they didn't follow instructions.
                    Stance: Favor
                    Explanation: The text supports tenure and due process as mechanisms that provide stability and job security for teachers, contrasting it with the risks faced without these protections.
                    }}


                    Please respond in the following JSON format, where each analysis result is an item in a list:
                    {{
                        "results": [
                            {{
                                "text_id": Text ID (an integer starting from 0),
                                "explanation": "A brief explanation of your assessment",
                                "favor_probability": Probability of a favor stance (a decimal between 0 and 1),
                                "neutral_probability": Probability of a irrelevant/neutral/no stance (a decimal between 0 and 1),
                                "against_probability": Probability of an against stance (a decimal between 0 and 1)
                            }},
                            ...
                        ]
                    }}'''
                else:
                    '''(**Note that they could belong to the same stance**)'''
                    batch_detection_prompt = '''You are a stance detection assistant. Given the following texts and a target, please analyze the stance each text potentially holds toward its corresponding target "{target}". Pay special attention to subtle or ironic expressions of stance.
                    
                    {texts}

                    Please evaluate whether each text holds a favor, against, or (neutral/no stance expressed) stance toward its corresponding target.
                    
                    Please respond in the following JSON format, where each analysis result is an item in a list:
                    {{
                        "results": [
                            {{
                                "text_id": Text ID (an integer starting from 0),
                                "explanation": "A brief explanation of your assessment",
                                "favor_probability": Probability of a favor stance (a decimal between 0 and 1),
                                "neutral_probability": Probability of a neutral/no stance (a decimal between 0 and 1),
                                "against_probability": Probability of an against stance (a decimal between 0 and 1)
                            }},
                            ...
                        ]
                    }}'''
            else:
                batch_detection_prompt = '''你是一名立场检测助手。给定以下多个文本和目标，请分析每个文本对其对应目标"{target}"潜在持有的立场，注意隐晦、反讽表达的立场。

                {texts}
            
                请评估每个文本对其对应目标持支持、反对立场或者（话题无关\没发表观点）的可能性。
            
                请以以下JSON格式回答，其中每个文本的分析结果都是一个列表项：
                {{
                    "results": [
                        {{
                            "text_id": 文本ID（从0开始的整数）,
                            "explanation": "简短解释您的评估",
                            "favor_probability": 支持立场的概率（0-1之间的小数）,
                            "neutral_probability": 话题无关或者单纯是新闻报道的概率（0-1之间的小数）,
                            "against_probability": 反对立场的概率（0-1之间的小数）
                        }},
                        ...
                    ]
                }}'''
        
        else:
            if language == "en":
                #  (**Note that they could belong to the same stance**)
                if isinstance(target, list):
                    batch_detection_prompt = '''You are a stance detection assistant. Given the following list of texts and their corresponding targets, please analyze the stance each text potentially holds toward its corresponding target. Pay special attention to subtle or ironic expressions of stance and their contexts.

                    Texts and Targets:
                    {texts}

                    Please evaluate whether each text holds a favor or against stance toward its corresponding target.

                    Please respond in the following JSON format, where each analysis result is an item in a list:
                    {{
                        "results": [
                            {{
                                "text_id": Text ID (an integer starting from 0),
                                "explanation": "A brief explanation of your assessment",
                                "favor_probability": Probability of a favor stance (a decimal between 0 and 1),
                                "against_probability": Probability of an against stance (a decimal between 0 and 1)
                            }},
                            ...
                        ]
                    }}'''
                else:
                    '''(**Note that they could belong to the same stance**)'''
                    batch_detection_prompt = '''You are a stance detection assistant. Given the following texts and a target, please analyze the stance each text potentially holds toward its corresponding target "{target}". Pay special attention to subtle or ironic expressions of stance.

                    {texts}

                    Please evaluate whether each text holds a favor or against stance toward its corresponding target.

                    Please respond in the following JSON format, where each analysis result is an item in a list:
                    {{
                        "results": [
                            {{
                                "text_id": Text ID (an integer starting from 0),
                                "explanation": "A brief explanation of your assessment",
                                "favor_probability": Probability of a favor stance (a decimal between 0 and 1),
                                "against_probability": Probability of an against stance (a decimal between 0 and 1)
                            }},
                            ...
                        ]
                    }}'''
            else:
                batch_detection_prompt = '''你是一名立场检测助手。给定以下多个文本和目标，请分析每个文本对其对应目标"{target}"潜在持有的立场，注意隐晦、反讽表达的立场。

                {texts}
            
                请评估每个文本对其对应目标持支持或者反对立场的可能性。
            
                请以以下JSON格式回答，其中每个文本的分析结果都是一个列表项：
                {{
                    "results": [
                        {{
                            "text_id": 文本ID（从0开始的整数）,
                            "explanation": "简短解释您的评估",
                            "favor_probability": 支持立场的概率（0-1之间的小数）,
                            "against_probability": 反对立场的概率（0-1之间的小数）
                        }},
                        ...
                    ]
                }}'''

        # Format texts string
        if isinstance(target, str):
            texts_str = "\n".join([f"```No.{i}:{text}\n```" for i,text in enumerate(self.enhance_texts_with_keywords(texts, knowledge_base, language=language))])
            batch_detection_prompt_formatted = batch_detection_prompt.format(target=target, texts=texts_str)
        else:
            texts_str = "\n".join([f"```No.{i}:\nText: {text_t[0]},\n Target: {text_t[1]}```\n" for i, text_t in enumerate(zip(self.enhance_texts_with_keywords(texts, knowledge_base, language=language), target))])
            batch_detection_prompt_formatted = batch_detection_prompt.format(texts=texts_str)


        # -----------------------
        # # # 情感分析
        import os
        from train_model import load_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_file = './saved_model/stanceberta_classifier'        
        model, tokenizer = load_model(model_file, device)
        model.eval()

        # -----------------------


        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                # Call API and get results
                conv = conv_template(lm.template)
                batch_detection_data = self.call_api_with_retry(lm, conv, batch_detection_prompt_formatted)
                results = batch_detection_data.get('results')
                if len(results) != len(texts):
                    raise ValueError(f'结果长度不匹配: {texts_str},texts_str|{results}, results|text:{len(texts)}|results:{len(results)}')
                break  # 成功获取结果，跳出重试循环
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e  # 超过最大重试次数，抛出异常
        processed_results = []


        for idx, result in enumerate(results):
            # Get probabilities from large model
            favor_prob = result.get('favor_probability')
            neutral_prob = result.get('neutral_probability',0)
            against_prob = result.get('against_probability')
            
            # -----------------------
            # 使用小模型对 explanation 进行立场检测
            explanation_text = result.get('explanation')
            if explanation_text:
                encoded_explanation = tokenizer([explanation_text], return_tensors='pt', padding=True, truncation=True, max_length=168).to(device)  # 128
                with torch.no_grad():
                    output = model(encoded_explanation['input_ids'], encoded_explanation['attention_mask'])
                    probabilities_expl = output[1]
                    small_model_expl_probs = probabilities_expl.cpu().numpy().tolist()[0]
            else:
                small_model_expl_probs = np.array([0.0, 0.0, 0.0])

            large_model_probs = np.array([against_prob, neutral_prob, favor_prob])

            inter_sim = self.distribution_similarity(large_model_probs, small_model_expl_probs)
            # inter_sim = 1
            # -----------------------

            # Determine final stance
            pred_label = self.determine_stance(against_prob, neutral_prob, favor_prob)


            ori_text_entropy = self.enhancer.calculate_stance_entropy(texts[idx], target)
            exp_entropy = self.enhancer.calculate_stance_entropy(result.get('explanation'), target)
            
            processed_results.append({
                "text_id": result.get('text_id'),
                "text": texts[idx],
                "aspect": target if isinstance(target,str) else target[idx],
                "explanation": result.get('explanation'),
                "pred_label": pred_label,
                "true_label": true_labels[idx],
                "favor_probability": favor_prob,
                "neutral_probability": neutral_prob,
                "against_probability": against_prob,
                "combined_favor": small_model_expl_probs[2],
                "combined_neutral": small_model_expl_probs[1],
                "combined_against": small_model_expl_probs[0],
                "inter_sim": inter_sim,
                "exter_sim": exp_entropy-ori_text_entropy,
            })

        return {
            "results": processed_results
        }

    def process_single_item(self, model: str, data_batch, true_label: int, language='en', knowledge_base='./datasets/knowledge_data.json', class_num=3) -> Dict:
        
        text = data_batch.get('text', data_batch.get('text_a')).tolist()[0]
        if 'Target' in data_batch and len(set(data_batch['Target'])) == 1:
            target = data_batch['Target'].tolist()[0]
        elif 'target' in data_batch and len(set(data_batch['target'])) == 1:
            target = data_batch['target'].tolist()[0]
        else:
            raise "not single data"
        

        lm = CommercialAPI(model)


        if language == "en":


            aspect_prompt = '''
            Given the following text:
            {text}
            Please extract and analyze the common characteristics of the text towards the target '{target}', such as common aspects or shared prior conditions.

            Please answer in the following JSON format:
            {{
                "aspect": "Aspects (comma-separated, 1-3 core-aspects of target '{target}')",
                "explanation": "A brief explanation (string)"
            }}
            '''
        else:
            aspect_prompt = '''
            Given the following text:
            {text}
            请提取并分析这些文本的共同特点，例如共同方面，共同先验条件。
    
            Please answer in the following JSON format:
            {{
                "aspect": "共同特点", # 逗号分隔
                "explanation": "简短解释 (1-3个方面)"
            }}
            '''

        

        if class_num == 3:
            if language == "zh":
                # case 添加中立， 以及gpt容易错的example
                stance_analysis_prompt = '''你是一名立场检测助手。给定以下文本和目标，请分析文本对该目标潜在持有的立场，注意隐晦、反讽表达的立场。

                文本: "{text}"
                目标: "{aspect}"

                请评估文本对该目标持支持、反对立场或者（话题无关\没发表观点）的可能性。

                请以以下JSON格式回答：
                {{
                    "explanation": "简短解释您的评估",
                    "favor_probability": 支持立场的概率（0-1之间的小数）,
                    "neutral_probability": 话题无关或者单纯是新闻报道的概率（0-1之间的小数）,
                    "against_probability": 反对立场的概率（0-1之间的小数）  
                }}'''
            else:
                stance_analysis_prompt = '''You are a stance detection assistant. Given the following text and the aspect of target "{target}", please analyze the potential stance the text holds toward the target, paying attention to subtle or sarcastic expressions of stance.

                Text: "{text}"
                Aspect: "{aspect}"

                Please evaluate the likelihood of the text showing a favorable, unfavorable, or (irrelevant/no opinion) stance toward the target '{target}'.


                
                Respond in the following JSON format:
                {{
                    "explanation": "A brief explanation of your assessment",
                    "favor_probability": The probability of a favorable stance (a decimal between 0 and 1),
                    "neutral_probability": The probability of being irrelevant or just reporting news (a decimal between 0 and 1),
                    "against_probability": The probability of an unfavorable stance (a decimal between 0 and 1)
                }}'''
        else:
            if language == "zh":
                stance_analysis_prompt = '''你是一名立场检测助手。给定以下文本和目标，请分析文本对该目标潜在持有的立场，注意隐晦、反讽表达的立场。

                文本: "{text}"
                目标: "{aspect}"

                请评估文本对该目标持支持或反对立场的可能性。

                请以以下JSON格式回答：
                {{
                    "explanation": "简短解释您的评估",
                    "favor_probability": 支持立场的概率（0-1之间的小数）,
                    "against_probability": 反对立场的概率（0-1之间的小数）
                }}'''
            else:
                stance_analysis_prompt = '''You are a stance detection assistant. Given the following text and the aspect of target "{target}", please analyze the potential stance the text holds toward the target, paying attention to subtle or sarcastic expressions of stance.

                Text: "{text}"
                Aspect: "{aspect}"

                Please evaluate the likelihood of the text showing a favorable or unfavorable stance toward the target '{target}'.
                
                Respond in the following JSON format:
                {{
                    "explanation": "A brief explanation of your assessment",
                    "favor_probability": The probability of a favorable stance (a decimal between 0 and 1),
                    "against_probability": The probability of an unfavorable stance (a decimal between 0 and 1)
                }}'''




        # Stage 1: Aspect Extraction
        aspect_complete_prompt = aspect_prompt.format(target=target,text=text)
        try:
            conv = conv_template(lm.template)
            aspect_data = self.call_api_with_retry(lm, conv, aspect_complete_prompt)
        except InvalidOutputError as e:
            logger.error(f"Aspect extraction failed: {e}")
            raise

        aspect = aspect_data.get('aspect')
        # aspect = 'iPhoneSE的'+aspect
        explanation = aspect_data.get('explanation')

        if not aspect:
            logger.debug(f"Aspect extraction data: {aspect_data}")
            raise InvalidOutputError("Aspect or explanation missing in the response.")



        # k_text = self.enhancer.enhance_texts_with_knowledge(text,knowledge_base,target,language)
        k_text = self.enhance_texts_with_keywords([text], knowledge_base, language=language)[0]
        # k_text = text

        stance_analysis_prompt_formatted = stance_analysis_prompt.format(text=k_text, target=target, aspect=aspect)

        


        try:

            conv = conv_template(lm.template)
            stance_analysis_data = self.call_api_with_retry(lm, conv, stance_analysis_prompt_formatted)
        except InvalidOutputError as e:
            logger.error(f"Probability calculations failed: {e}")
            raise

        favor_probability = stance_analysis_data.get('favor_probability')
        neutral_probability = stance_analysis_data.get('neutral_probability',0)
        against_probability = stance_analysis_data.get('against_probability')
        stance_explanation = stance_analysis_data.get('explanation')



        pred_label2 = self.determine_stance(against_probability, neutral_probability, favor_probability)

        ori_text_entropy = self.enhancer.calculate_stance_entropy(text, target)
        exp_entropy = self.enhancer.calculate_stance_entropy(explanation, target)

        # -----------------------
        # # 情感分析
        import os
        from train_model import load_model

        # -----------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_file = './saved_model/stanceberta_classifier'
        model, tokenizer = load_model(model_file, device)
        model.eval()
        # -----------------------


        # 使用小模型对 explanation 进行立场检测
        large_model_probs = np.array([against_probability, neutral_probability, favor_probability])
        explanation_text = explanation
        if explanation_text:
            encoded_explanation = tokenizer([explanation_text], return_tensors='pt', padding=True, truncation=True,
                                max_length=168).to(device)
            with torch.no_grad():
                output = model(encoded_explanation['input_ids'], encoded_explanation['attention_mask'])
                probabilities_expl = output[1]
                small_model_expl_probs = probabilities_expl.cpu().numpy().tolist()[0]
        else:
            small_model_expl_probs = np.array([0.0, 0.0, 0.0])
        inter_sim = self.distribution_similarity(large_model_probs, small_model_expl_probs)
        
        # -----------------------
        

        results = []
        results.append({
            "text": text,
            "aspect": f'{target} : {aspect}',
            "explanation": explanation,
            "true_label": true_label,
            "pred_label": pred_label2,
            "favor_probability": favor_probability,
            "neutral_probability": neutral_probability,
            "against_probability": against_probability,
            "stance_explanation": stance_explanation,
            "combined_favor": small_model_expl_probs[2],
            "combined_neutral": small_model_expl_probs[1],
            "combined_against": small_model_expl_probs[0],
            "inter_sim": inter_sim,
            "exter_sim": exp_entropy-ori_text_entropy,
            # 'unbias': stance_unbias
        })

        return {"results":results,
        }



    
    def batch_run_detection(self, val_data: pd.DataFrame, batch_size: int = 4, target = 'tweet stance detection', language='en', knowledge_base='./datasets/knowledge_data.json', class_num=3):
        # for model in self.models:
        #     print(f"Processing with model: {model}")

        total_attempts = 0  # 记录总尝试数

        if True:
            # 设置样本数量和批次大小
            sample_size = min(64, len(val_data))  # 取 10 个样本或者 val_data 的长度，以小的为准
            # sample_size = len(val_data)


            unclassified_data = val_data.iloc[:sample_size].copy()
            all_results = []
            retry_count = 0
            max_retries = 3
            # 维护一个存储低置信度的样本列表
            low_inter_sample = {}
            while not unclassified_data.empty and retry_count < max_retries:
                model = self.models[min(retry_count, len(self.models) - 1)]
                batch_size = int(max(1, batch_size / (2 ** retry_count)))
                print(f"Processing with model: {model}")
                print(f"batch_size: {batch_size}")

                # 设置样本数量和批次大小
                sample_size = len(unclassified_data)
                data_batches = [unclassified_data.iloc[i:min(i + batch_size, sample_size)] for i in range(0, sample_size, batch_size)]

                with ThreadPoolExecutor(max_workers=4) as executor:
                    if batch_size>1:
                        batch_futures = [executor.submit(self.process_batch_item, model, data_batches[idx], target, data_batches[idx]['label'].tolist(), language, False, knowledge_base, class_num) for idx, batch in enumerate(data_batches)]
                        # total_attempts += 1  # 每次处理任务时增加尝试计数
                        total_attempts += len(data_batches)
                    else:
                        # print(data_batches)
                        batch_futures = [executor.submit(self.process_single_item, model, data_batches[idx], batch['label'].tolist()[0], language, knowledge_base, class_num) for idx, batch in enumerate(data_batches)]
                        total_attempts += len(data_batches)

                    for future in as_completed(batch_futures):
                        
                        try:
                            for result in future.result().get('results'):

                                # print(result)
                                if result['exter_sim'] < 0:
                                    continue
                                if result["inter_sim"] < 0.95: # 0.6
                                    if low_inter_sample.get(result['text']) is None:
                                        low_inter_sample[result['text']] = [result]
                                    else:
                                        low_inter_sample[result['text']].append(result)
                                    continue
                                # print(result)
                                all_results.append(result)
                        except Exception as e:
                            print(f"Error processing batch: {e}")

                # 检查是否有未分类的样本
                classified_indices = {result['text'] for result in all_results}


                unclassified_data = unclassified_data[~unclassified_data['text'].isin(classified_indices)]

                retry_count += 1

            texts = [result['text'] for result in all_results]
            
            # 按照一致性加权聚合low_inter_simple中的样本
            for text,result_list in low_inter_sample.items():
                if text not in texts:

                    vote_counts = {'favor': 0, 'neutral': 0, 'against': 0}
                    for result in result_list:
                        pred_label = self.determine_stance(
                            result['against_probability'], result['neutral_probability'], result['favor_probability']
                        )
                        if pred_label == 2:
                            vote_counts['favor'] += 1
                        elif pred_label == 1:
                            vote_counts['neutral'] += 1
                        elif pred_label == 0:
                            vote_counts['against'] += 1
                    total_weight = sum(res['inter_sim'] for res in result_list)

                    if total_weight > 0:
                        weighted_against = sum(
                            result['against_probability'] * result['inter_sim'] for result in result_list) / total_weight
                        weighted_neutral = sum(
                            result['neutral_probability'] * result['inter_sim'] for result in result_list) / total_weight
                        weighted_favor = sum(
                            result['favor_probability'] * result['inter_sim'] for result in result_list) / total_weight

                        weighted_combined_favor = sum(
                            result['combined_favor'] * result['inter_sim'] for result in result_list) / total_weight
                        weighted_combined_against = sum(
                            result['combined_against'] * result['inter_sim'] for result in result_list) / total_weight
                        weighted_combined_neutral = sum(
                            result['combined_neutral'] * result['inter_sim'] for result in result_list) / total_weight


                        final_label = self.determine_stance(weighted_against, weighted_neutral, weighted_favor)

                        # Create a final result for the aggregated low-confidence sample
                        aggregated_result = {
                            "text_id": result_list[-1].get('text_id',0),  # Take the first result's ID for consistency
                            "text": text,
                            "aspect": result_list[-1]['aspect'],
                            "explanation": result_list[-1]['explanation'],
                            # Take the explanation from the first entry or aggregate if needed
                            "pred_label": final_label,
                            "true_label": result_list[-1]['true_label'],
                            "favor_probability": weighted_favor,
                            "neutral_probability": weighted_neutral,
                            "against_probability": weighted_against,
                            "combined_favor": weighted_combined_favor,
                            "combined_neutral": weighted_combined_neutral,
                            "combined_against": weighted_combined_against,
                            "inter_sim": np.mean([r['inter_sim'] for r in result_list]),  # Average the inter_sim values
                            "exter_sim": np.mean([r['exter_sim'] for r in result_list])  # Average the exter_sim values,
                        }

                        # Add this aggregated result to all_results
                        all_results.append(aggregated_result)



            if True:
                y_true, y_pred = self.process_results(all_results)
                self.print_metrics(y_true, y_pred, model)
            print(f"总尝试数: {total_attempts}")  # 打印总尝试数



    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix


    def determine_stance(self, against: float, neutral: float, favor: float) -> int:
        probabilities = [against, neutral, favor]
        return probabilities.index(max(probabilities))



    def stance_f1(self, y_true, y_pred):
        # 将 y_true 和 y_pred 转换为 NumPy 数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 计算 F1 scores for 'AGAINST' (label=0) and 'FAVOR' (label=2)
        f1_against = f1_score(y_true, y_pred, labels=[0], average='macro', zero_division=0)
        f1_favor = f1_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)

        # 计算这两个 F1 分数的平均值
        f1_average = (f1_against + f1_favor) / 2

        # 打印预测结果的混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        print("Confusion Matrix (y_true vs. y_pred):")
        print(cm)

        # 打印ground truth的混淆矩阵（y_true vs. y_true）
        cm_ground_truth = confusion_matrix(y_true, y_true, labels=[0, 1, 2])
        print("\nGround Truth Confusion Matrix (y_true vs. y_true):")
        print(cm_ground_truth)

        # 打印每个标签的选中数量和样本数量
        for label in [0, 1, 2]:
            selected_count = sum((y_pred == label).astype(int))
            true_count = sum((y_true == label).astype(int))
            print(f"\nLabel {label}: selected count = {selected_count}, true count = {true_count}")

        return f1_average

    def print_metrics(self, y_true: List[int], y_pred: List[int], model: str):
        if len(y_true) > 0 and len(y_pred) > 0:
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            average_f1 = self.stance_f1(y_true, y_pred)

            print(f"Model: {model}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Average F1 Score: {average_f1:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")

        else:
            print(f"Model: {model} - Unable to calculate metrics. Possible issue with label range.")


    def process_results(self,all_results):
        y_true, y_pred = [], []
        output_data = []  # 用于存储输出信息
        for idx, result in enumerate(all_results):
            y_true.append(result["true_label"])
            y_pred.append(result["pred_label"])

            output_entry = {
                "text": result['text'],
                "aspect": result['aspect'],
                "explanation": result['explanation'],
                "favor_probability": result['favor_probability'],
                "neutral_probability": result['neutral_probability'],
                "against_probability": result['against_probability'],
                "combined_favor": result['combined_favor'],
                "combined_neutral": result['combined_neutral'],
                "combined_against": result['combined_against'],
                "exter_sim": result["exter_sim"],
                "inter_sim": result["inter_sim"],
                "true_label": result['true_label'],
                "pred_label": result['pred_label']
            }
            output_data.append(output_entry)

            print(f"Text: {result['text']}")
            print(f"Aspect: {result['aspect']}")
            print(f"Explanation: {result['explanation']}")
            print(f"Favor: {result['favor_probability']:.2f}, "
                    f"Neutral: {result['neutral_probability']:.2f}, "
                    f"Against: {result['against_probability']:.2f}")
            print(f"Favor: {result['combined_favor']:.2f}, "
                    f"Neutral: {result['combined_neutral']:.2f}, "
                    f"Against: {result['combined_against']:.2f}")
            print(f'''exter_sim: {result["exter_sim"]:.2f}", inter_sim: {result["inter_sim"]:.2f}''')
            print(f"True Label: {result['true_label']}, Predicted Label: {result['pred_label']}")

        # 将输出信息保存为jsonl格式
        with open('./datasets/eval.jsonl', 'a', encoding='utf-8') as f:  # 变成a，加模式写入
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print("-" * 50)
            
        return y_true, y_pred


class VASTStanceDetector(StanceDetector):
    def __init__(self, models: List[str]):
        super().__init__(models)

        # 更新label2id映射
        self.id2id = {
            1:2,
            0:0,
            2:1
        }



    def process_results(self,all_results):
        y_true, y_pred = [], []
        output_data = []  # 用于存储输出信息
        for idx, result in enumerate(all_results):
            y_true.append(self.id2id[result["true_label"]])
            y_pred.append(result["pred_label"])

            output_entry = {
                "text": result['text'],
                "aspect": result['aspect'],
                "explanation": result['explanation'],
                "favor_probability": result['favor_probability'],
                "neutral_probability": result['neutral_probability'],
                "against_probability": result['against_probability'],
                "combined_favor": result['combined_favor'],
                "combined_neutral": result['combined_neutral'],
                "combined_against": result['combined_against'],
                "exter_sim": result["exter_sim"],
                "inter_sim": result["inter_sim"],
                "true_label": result['true_label'],
                "pred_label": result['pred_label']
            }
            output_data.append(output_entry)

            print(f"Text: {result['text']}")
            print(f"Aspect: {result['aspect']}")
            print(f"Explanation: {result['explanation']}")
            print(f"Favor: {result['favor_probability']:.2f}, "
                    f"Neutral: {result['neutral_probability']:.2f}, "
                    f"Against: {result['against_probability']:.2f}")
            print(f"Favor: {result['combined_favor']:.2f}, "
                    f"Neutral: {result['combined_neutral']:.2f}, "
                    f"Against: {result['combined_against']:.2f}")
            print(f'''exter_sim: {result["exter_sim"]:.2f}", inter_sim: {result["inter_sim"]:.2f}''')
            print(f"True Label: {self.id2id[result['true_label']]}, Predicted Label: {result['pred_label']}")
            # self.id2id[result["true_label"]]
        # 将输出信息保存为jsonl格式
        with open('./datasets/eval.jsonl', 'a', encoding='utf-8') as f:  # 变成a，加模式写入
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print("-" * 50)
            
        return y_true, y_pred


class PstanceDetector(StanceDetector):
    def __init__(self, models: List[str]):
        super().__init__(models)

        # 更新label2id映射
        self.id2id = {
            1:2,
            0:0,
            2:1
        }


    def process_results(self, all_results):
        y_true, y_pred = [], []
        output_data = []  # 用于存储输出信息
        for idx, result in enumerate(all_results):
            y_true.append(self.id2id[result["true_label"]])
            y_pred.append(result["pred_label"])

            # 修改为只有Favor和Against标签，中立默认为0
            favor_probability = result['favor_probability']
            against_probability = result['against_probability']
            neutral_probability = 0.0  # 中立默认为0

            output_entry = {
                "text": result['text'],
                "aspect": result['aspect'],
                "explanation": result['explanation'],
                "favor_probability": favor_probability,
                "neutral_probability": neutral_probability,
                "against_probability": against_probability,
                "combined_favor": result['combined_favor'],
                "combined_neutral": neutral_probability,  # 中立默认为0
                "combined_against": result['combined_against'],
                "exter_sim": result["exter_sim"],
                "inter_sim": result["inter_sim"],
                "true_label": result['true_label'],
                "pred_label": result['pred_label']
            }
            output_data.append(output_entry)

            print(f"Text: {result['text']}")
            print(f"Aspect: {result['aspect']}")
            print(f"Explanation: {result['explanation']}")
            print(f"Favor: {favor_probability:.2f}, "
                    f"Neutral: {neutral_probability:.2f}, "
                    f"Against: {against_probability:.2f}")
            print(f"Favor: {result['combined_favor']:.2f}, "
                    f"Neutral: {neutral_probability:.2f}, "
                    f"Against: {result['combined_against']:.2f}")
            print(f'''exter_sim: {result["exter_sim"]:.2f}", inter_sim: {result["inter_sim"]:.2f}''')
            print(f"True Label: {self.id2id[result['true_label']]}, Predicted Label: {result['pred_label']}")
            # self.id2id[result["true_label"]]
        # 将输出信息保存为jsonl格式
        with open('./datasets/eval.jsonl', 'a', encoding='utf-8') as f:  # 变成a，加模式写入
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print("-" * 50)
            
        return y_true, y_pred





if __name__ == "__main__":

    # os.environ["http_proxy"] = "http://127.0.0.1:7891"
    # os.environ["https_proxy"] = "http://127.0.0.1:7891"
    models= ["gpt-4o-mini"]

    # 随机洗牌
    data_path = './datasets/semeval2016/hc'
    knowledge_base = f'{data_path}/knowledge_data.json'
    val_data = pd.read_csv(f"{data_path}/test.csv")
    val_data = val_data.sample(frac=1).reset_index(drop=True)


    # 实例化模型
    if 'VAST' in data_path:
        val_data = val_data[val_data['seen?'] == 0]
        original_count = val_data.shape[0]
        val_data = val_data[['text', 'target', 'label']].drop_duplicates(subset='text')
        new_count = val_data.shape[0]
        print(f"去重前数量: {original_count}, 去重后数量: {new_count}")
        detector = VASTStanceDetector(models)
    elif 'pstance' in data_path: 
        detector = PstanceDetector(models)
    else:
        detector = StanceDetector(models)

        
    if 'pstance' in data_path:
        class_num = 2
    else:
        class_num = 3
    target = 'stance detection target'
    detector.batch_run_detection(val_data, batch_size=32, target=target, language='en', knowledge_base=knowledge_base,class_num=class_num)
