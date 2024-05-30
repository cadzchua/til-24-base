import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from typing import Dict

class NLPManager:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForQuestionAnswering.from_pretrained('cadzchua/distil-bert-ft-qa-model-7up-v6')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def qa(self, context: str) -> Dict[str, str]:
        # Perform NLP question-answering
        context = context.lower()
        hdg_ans = self.find_heading(context)
        hdg_ans = self.heading_text_to_string(hdg_ans)
        if "drone catcher" in context:
            tool_ans = "drone catcher"
        elif "interceptor jets" in context:
            tool_ans = "interceptor jets"
        elif "(emp)" in context:
            tool_ans = "EMP"
        else:
            tool_ans = self.predict_answer("What is the tool to be deployed?", context)
            tool_ans = self.clean_tool_answer(tool_ans)

        tgt_ans = self.predict_answer("What is the target?", context)
        tgt_ans = self.clean_target_answer(tgt_ans, context)
        return {"heading": hdg_ans, "tool": tool_ans, "target": tgt_ans}

    def find_heading(self, context: str) -> str:
        words = word_tokenize(context)
        pos_tags = pos_tag(words)

        heading = []
        pos_set = {'CD', 'NN', 'JJ', 'NNS'}
        trigger_words = {'heading', 'head'}
        connector_words = {'of', 'at', 'to'}

        # Look for trigger words at the beginning of the context
        for i, (word, pos) in enumerate(pos_tags):
            if word.lower() in trigger_words:
                if i + 1 < len(pos_tags) and pos_tags[i + 1][0].lower() in connector_words:
                    i += 1
                if i + 1 < len(pos_tags) and pos_tags[i + 1][1] in pos_set:
                    if i + 2 < len(pos_tags) and pos_tags[i + 2][1] in pos_set:
                        if i + 3 < len(pos_tags) and pos_tags[i + 3][1] in pos_set:
                            heading.extend([pos_tags[i + 1][0], pos_tags[i + 2][0], pos_tags[i + 3][0]])
                            break
                    break

        # If no trigger words found, look for cardinal numbers
        if not heading:
            for i, (word, pos) in enumerate(pos_tags):
                if pos == 'CD':
                    if i + 1 < len(pos_tags) and pos_tags[i + 1][1] in pos_set:
                        heading.extend([word, pos_tags[i + 1][0]])
                        if i + 2 < len(pos_tags) and pos_tags[i + 2][1] in pos_set:
                            heading.append(pos_tags[i + 2][0])
                            break
                    break

        return " ".join(heading)

    def heading_text_to_string(self, heading_text: str) -> str:
        heading_mapping = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9"
        }
        words = heading_text.split()
        string_heading = "".join([heading_mapping.get(word, "") for word in words])
        return string_heading.zfill(3)  # Ensure the heading is always three digits

    def predict_answer(self, question: str, context: str) -> str:
        inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True, padding="max_length").to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():  # Disable gradient calculation for faster inference
            outputs = self.model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))
        return answer.strip()

    def clean_tool_answer(self, tool_ans: str) -> str:
        if "-" in tool_ans:
            tool_ans = self.remove_spaces_around_hyphens(tool_ans)
        if tool_ans.lower() in ['emp', 'emp tool']:
            tool_ans = 'EMP'
        if tool_ans.lower() == "machine guns":
            tool_ans = "machine gun"
        if 'with' in tool_ans:
            tool_ans = tool_ans.split('with', 1)[1].strip()
        if 'using' in tool_ans:
            tool_ans = tool_ans.split('using', 1)[1].strip()
        return tool_ans

    def clean_target_answer(self, tgt_ans: str, context: str) -> str:
        tgt_ans = tgt_ans.replace("planes", "plane")
        if '.' in tgt_ans:
            tgt_ans = tgt_ans.split('.', 1)[0].strip()
        if tgt_ans == '':
            tgt_ans = self.extract_reference_phrase(context)
        return tgt_ans

    def remove_spaces_around_hyphens(self, text: str) -> str:
        return re.sub(r'\s*-\s*', '-', text)

    def extract_reference_phrase(self, context: str) -> str:
        pattern = r'reference\s(.*?)(?=[,\.])'
        match = re.search(pattern, context)
        if match:
            return match.group(1).strip()
        else:
            return ""
