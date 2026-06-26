import os
import openai


def _legacy_chat_completion(model, messages, temperature):
    """Fallback call path for the legacy openai<1.0 SDK."""
    chat_completion = getattr(openai, 'ChatCompletion', None)
    if chat_completion is None:
        raise RuntimeError(
            "Validator requires either an injected OpenAI v1 client "
            "(pass `client=...`) or the legacy openai<1.0 SDK with "
            "openai.api_key configured."
        )
    return chat_completion.create(model=model, messages=messages, temperature=temperature)


class Validator(object):
    """
    LLM Validator: validate plans using LLM
    arg: argument parameters
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    client: optional openai.OpenAI v1 client (injected by the OMNISAFE adapter).
            When omitted, falls back to the legacy openai.ChatCompletion path
            so the upstream CLI still works against openai<1.0.
    """
    def __init__(self, arg, is_log_example=False, temperature=0, client=None):

        self.arg = arg
        self.model = arg.model
        self.temperature = temperature
        self.messages = None
        self.log_dir = arg.logdir
        self.log_file_path = self.log_dir + "/validator_log.txt"
        self.is_log_example = is_log_example
        self.client = client

        # root for prompt examples
        if self.arg.domain == 'blocksworld':
            self.max_examples = 6
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
            if arg.method == "LLM_no_trans" or arg.method == "LLM_no_trans_self_feedback":
                self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocksworld_no_trans_examples")
            else:
                self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocksworld_examples")
        elif self.arg.domain == 'ballmoving':
            self.max_examples = 5
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ballmoving_examples")
        elif self.arg.domain == 'cooking':
            self.max_examples = 4
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cooking_examples")
        elif self.arg.domain == 'household':
            self.max_examples = 4
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "household_examples")

        # initialize messages
        self.init_messages()

    # Write content to file
    def write_content(self, content, is_append):

        if is_append == False:

            with open(self.log_file_path, "w") as f:
                f.write(content+"\n")

        else:

            with open(self.log_file_path, "a") as f:
                f.write(content+"\n")

    # Initialize messages include opening and few-shot examples
    def init_messages(self, is_reinitialize = False):

        # opening setup
        file_path = self.prompt_example_root + "/opening.txt"
        with open(file_path, 'r') as f:
            contents = f.read()
            opening_message =  {"role": "system", "content": contents}
            self.messages = [opening_message]
            # record content
            if self.is_log_example == True and is_reinitialize == False:
                self.write_content(content= contents, is_append=False)

        # load few-shot examples
        for i in range(self.num_valid_example):

            file_path = self.prompt_example_root + "/example"+str(i)+".txt"
            with open(file_path, 'r') as f:
                contents = f.read().split('Answer:', 1)
                question = contents[0]
                answer = 'Answer:' + contents[1]

                question_message = {"role": "system", "name":"example_user", "content": question}
                self.messages.append(question_message)
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= question, is_append=True)

                answer_message = {"role": "system", "name":"example_assistant", "content": answer}
                self.messages.append(answer_message)
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= answer, is_append=True)


    # Query question message
    def query(self, content, is_append=False):

        # add new question to message list
        question_message = {"role": "user", "content": content}
        if is_append == False:
            question = self.messages.copy()
        else:
            question = self.messages
        question.append(question_message)
        self.write_content(content=content, is_append=True)

        if self.client is not None:
            raw = self.client.chat.completions.create(
                model=self.model, messages=question, temperature=self.temperature
            )
            response_content = raw.choices[0].message.content
            response = {"choices": [{"message": {"content": response_content}}]}
        else:
            response = _legacy_chat_completion(self.model, question, self.temperature)
            response_content = response["choices"][0]["message"]["content"]

        self.write_content(content=response_content, is_append=True)

        return response
