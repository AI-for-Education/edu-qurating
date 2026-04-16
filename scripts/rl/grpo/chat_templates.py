import re
from dataclasses import dataclass


@dataclass
class Reasoning:
    reasoning_start: str = "<start_working_out>"  # Acts as think-open tag
    reasoning_end: str = "<end_working_out>"  # Acts as think-close tag
    response_start: str = "<start_response>"
    response_end: str = "<end_response>"

    system_prompt: str = (
        "You are given a problem."
        "Think about the problem and provide your working out."
        "Place it between {reasoning_start} and {reasoning_end}."
        "Then, provide your solution between {response_start}{response_end}"
    )

    def render_system_prompt(self):
        return self.system_prompt.format(
            reasoning_start=self.reasoning_start,
            reasoning_end=self.reasoning_end,
            response_start=self.response_start,
            response_end=self.response_end,
        )

    def chat_template(self):
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{{ '{system_prompt}' + eos_token }}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
            "{% endif %}"
        )

        template = template.replace(
            "'{system_prompt}'", f"'{self.render_system_prompt()}'"
        ).replace("'{reasoning_start}'", f"'{self.reasoning_start}'")
        
        return template

    def formatter(self, tokenizer) -> re.Pattern:
        response_end_regex = (
            rf"{self.response_end}[\s]{{0,}}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
        )

        match_format = re.compile(
            rf"(.*?){self.reasoning_end}"
            rf"{self.response_start}(.+?){response_end_regex}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )
        
        return match_format


@dataclass
class ReasoningStudentMaterial(Reasoning):
    material_needed_start: str = "<start_material_needed>"
    material_needed_end: str = "<end_material_needed>"
    student_material_start: str = "<start_student_material>"
    student_material_end: str = "<end_student_material>"

    system_prompt: str = (
        "You are given a task for an early grade teacher."
        "\nThink about the task and provide your working out."
        "\nPlace it between {reasoning_start} and {reasoning_end}."
        "\nThen, if you need to generate material that should be shown to a student,"
        '\nplace "yes" between {material_needed_start} and {material_needed_end}, otherwise place "no".'
        "\nThen, if material is needed, place it between {student_material_start} and {student_material_end}."
        "Finally, provide your response between {response_start} and {response_end}."
        "Your response is the only part the teacher will see - everything else is just for you"
        "- so make sure to include everything that you want the teacher to see in your response."
    )

    def render_system_prompt(self):
        return self.system_prompt.format(
            reasoning_start=self.reasoning_start,
            reasoning_end=self.reasoning_end,
            material_needed_start=self.material_needed_start,
            material_needed_end=self.material_needed_end,
            student_material_start=self.student_material_start,
            student_material_end=self.student_material_end,
            response_start=self.response_start,
            response_end=self.response_end,
        )

    def formatter(self, tokenizer) -> re.Pattern:
        response_end_regex = (
            rf"{self.response_end}[\s]{{0,}}" + "(?:" + re.escape(tokenizer.eos_token) + ")*?"
        )

        match_format = re.compile(
            rf"(.*?){self.reasoning_end}"
            rf"{self.material_needed_start}(yes|no){self.material_needed_end}"
            rf"{self.student_material_start}(.*?){self.student_material_end}"
            rf"{self.response_start}(.+?){response_end_regex}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )
        
        return match_format