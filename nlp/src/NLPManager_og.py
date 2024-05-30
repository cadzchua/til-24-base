from typing import Dict
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import re

class NLPManager:
    def __init__(self):
        # initialize the model here
        
        # Load a pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') 
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        hdg_ans = self.predict_answer("What is the heading?", context) # "According to the instructions, what heading should the turret target?"
        hdg_ans = self.heading_text_to_string(hdg_ans)
        tool_ans = self.predict_answer("What is the tool to be deployed?", context) # "Based on the instructions, what tool or weapon should the turret deploy?"
        if "-" in tool_ans:
            tool_ans = self.remove_spaces_around_hyphens(tool_ans)
        if tool_ans == 'emp' or tool_ans == 'emp tool':
            tool_ans = 'EMP'
        tgt_ans = self.predict_answer("What is the target?", context) # "According to the instructions, what is the description of the target?"
        
        return {"heading": hdg_ans, "tool": tool_ans, "target": tgt_ans}

    def heading_text_to_string(self, heading_text):
        """
        Convert heading text to string representation with leading zeros.

        Args:
            heading_text (str): Heading text in format "zero six five".

        Returns:
            str: String representation of the heading with leading zeros.
        """
        # Define a mapping from text to integer representations
        heading_mapping = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9"
        }

        # Split the heading text into words and convert each word to its integer representation
        words = heading_text.split()
        string_heading = ""
        for word in words:
            if word in heading_mapping:
                string_heading += heading_mapping[word]

        return string_heading
    
    def predict_answer(self, question, context):
        inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt', add_special_tokens=True).to(self.device)
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']

        outputs = self.model(input_ids, token_type_ids=token_type_ids)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))
        return answer
    
    def remove_spaces_around_hyphens(self, text: str) -> str:
        # Use regular expression to find spaces around hyphens and remove them
        cleaned_text = re.sub(r'\s*-\s*', '-', text)
        return cleaned_text