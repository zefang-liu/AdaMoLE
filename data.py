"""
Loading and Preprocessing Datasets
"""
import os

from datasets import load_dataset, concatenate_datasets, DatasetDict


def format_text(example, data_name: str, prompt_only: bool = True):
    """
    Format an example into one text
    """
    if data_name == 'boolq':
        """
        Passage: Windows Movie Maker -- Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) 
        is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and 
        offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, 
        and Flickr.
        Question: is windows movie maker part of windows essentials
        Choices:
        A. No
        B. Yes
        Answer: B
        """
        text = f"Passage: {example['passage']}\nQuestion: {example['question']}\nChoices:\n"
        text += "A. No\nB. Yes\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'cb':
        """
        Text: It was a complex language. Not written down but handed down. One might say it was peeled down.
        Hypothesis: the language was peeled down
        Question: Does the text entail the hypothesis, contradict it, or is it neutral?
        Choices:
        A. Entailment
        B. Contradiction
        C. Neutral
        Answer: A
        """
        text = f"Text: {example['premise']}\nHypothesis: {example['hypothesis']}\n" \
               f"Question: Does the text entail the hypothesis, contradict it, or is it neutral?\nChoices:\n"
        text += "A. Entailment\nB. Contradiction\nC. Neutral\n"
        text += "Answer: "
        example['answer'] = ['A', 'B', 'C'][example['label']]
        example['num_choices'] = 3

    elif data_name == 'copa':
        """
        Premise: My body cast a shadow over the grass.
        Question: What’s the cause for this?
        Choices:
        A. The sun was rising.
        B. The grass was cut.
        Answer: A
        """
        text = f"Premise: {example['premise']}\nQuestion: What’s the {example['question']} for this?\nChoices:\n"
        text += f"A. {example['choice1']}\nB. {example['choice2']}\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'multirc':
        """
        Paragraph: While this process moved along, diplomacy continued its rounds. Direct pressure on the Taliban had 
        proved unsuccessful. As one NSC staff note put it, "Under the Taliban, Afghanistan is not so much a state 
        sponsor of terrorism as it is a state sponsored by terrorists." ...
        Question: What did the high-level effort to persuade Pakistan include?
        Candidate Answer: Children, Gerd, or Dorian Popa
        Choices:
        A. False
        B. True
        Answer: A
        """
        text = f"Paragraph: {example['paragraph']}\nQuestion: {example['question']}\n" \
               f"Candidate Answer: {example['answer']}\nChoices:\n"
        text += f"A. False\nB. True\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'record':
        raise NotImplementedError

    elif data_name == 'rte':
        """
        Text: No Weapons of Mass Destruction Found in Iraq Yet.
        Hypothesis: Weapons of Mass Destruction Found in Iraq.
        Question: Does the text entail the hypothesis or not?
        Choices:
        A. Entailment
        B. Not entailment
        Answer: B
        """
        text = f"Text: {example['premise']}\nHypothesis: {example['hypothesis']}\n" \
               f"Question: Does the text entail the hypothesis or not?\nChoices:\n"
        text += "A. Entailment\nB. Not entailment\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'wic':
        """
        Context 1: Do you want to come over to my <place> later?
        Context 2: A political system with no <place> for the less prominent groups.
        Question: Is the word in brackets used with the same meaning in both contexts?
        Choices:
        A. False
        B. True
        Answer: A
        """
        sentence1 = example['sentence1']
        sentence2 = example['sentence2']
        marked_sentence1 = sentence1[:example['start1']] + '<' + sentence1[example['start1']:example['end1']] \
                           + '>' + sentence1[example['end1']:]
        marked_sentence2 = sentence2[:example['start2']] + '<' + sentence2[example['start2']:example['end2']] \
                           + '>' + sentence2[example['end2']:]
        text = f"Context 1: {marked_sentence1}\nContext 2: {marked_sentence2}\n" \
               f"Question: Is the word in brackets used with the same meaning in both contexts?\nChoices:\n"
        text += "A. False\nB. True\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'wsc.fixed':
        """
        Text: <Mark> told Pete many lies about himself, which Pete included in his book. <He> should have been more 
        skeptical.
        Question: Is the pronoun in brackets referring to the correct entity as intended in the context?
        Choices:
        A. False
        B. True
        Answer: A
        """
        tokens = example['text'].split()
        span1_start = example['span1_index']
        span1_end = example['span1_index'] + len(example['span1_text'].split())
        span2_start = example['span2_index']
        span2_end = example['span2_index'] + len(example['span2_text'].split())
        marked_tokens = tokens[:span1_start] + ['<' + example['span1_text'] + '>'] + tokens[span1_end:span2_start] \
                        + ['<' + example['span2_text'] + '>'] + tokens[span2_end:]
        marked_text = ' '.join(marked_tokens)
        text = f"Text: {marked_text}\n" \
               f"Question: Is the pronoun in brackets referring to the correct entity as intended in the context?\n" \
               f"Choices:\n"
        text += "A. False\nB. True\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'commonsense_qa':
        """
        Question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the 
        school had made to change?
        Choices:
        A. ignore
        B. enforce
        C. authoritarian
        D. yell at
        E. avoid
        Answer: A
        """
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 5

    elif data_name == 'cosmos_qa':
        """
        Context: Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . 
        seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to 
        them ; I think it 's the fact that they seemed so happy themselves when they played .
        Question: In the future , will this person go to see other bands play ?
        Choices:
        A. None of the above choices .
        B. This person likes music and likes to see the show , they will see other bands play .
        C. This person only likes Good Old War and Person L , no other bands .
        D. Other Bands is not on tour and this person can not see them .
        Answer: B
        """
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
        text += f"A. {example['answer0']}\n"
        text += f"B. {example['answer1']}\n"
        text += f"C. {example['answer2']}\n"
        text += f"D. {example['answer3']}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + example['label'])
        example['num_choices'] = 4

    elif data_name == 'social_i_qa':
        """
        Context: Cameron decided to have a barbecue and gathered her friends together.
        Question: How would Others feel as a result?
        Choices:
        A. like attending
        B. like staying home
        C. a good friend to have
        Answer: A
        """
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
        text += f"A. {example['answerA']}\n"
        text += f"B. {example['answerB']}\n"
        text += f"C. {example['answerC']}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + int(example['label']) - 1)
        example['num_choices'] = 3

    elif data_name == 'piqa':
        """
        Question: When boiling butter, when it's ready, you can
        Choices:
        A. Pour it onto a plate
        B. Pour it into a jar
        Answer: B
        """
        text = f"Question: {example['goal']}\nChoices:\n"
        text += f"A. {example['sol1']}\n"
        text += f"B. {example['sol2']}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + example['label'])
        example['num_choices'] = 2

    elif data_name == 'openbookqa':
        """
        Fact: the sun is the source of energy for physical cycles on Earth
        Question: The sun is responsible for
        Choices:
        A. puppies learning new tricks
        B. children growing up and getting old
        C. flowers wilting in a vase
        D. plants sprouting, blooming and wilting
        Answer: D
        """
        text = f"Fact: {example['fact1']}\nQuestion: {example['question_stem']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 4

    elif data_name == 'ai2_arc':
        """
        Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most 
        heat?
        Choices:
        A. dry palms
        B. wet palms
        C. palms covered with oil
        D. palms covered with lotion
        Answer: A
        """
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 4

    elif data_name == 'scienceqa':
        """
        Question: Which tense does the sentence use?
        Mona will print her name with care.
        Choices:
        A. present tense
        B. future tense
        C. past tense
        Answer: B
        """
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for index, choice in enumerate(choices):
            text += f"{chr(ord('A') + index)}. {choice}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + example['answer'])
        example['num_choices'] = 5  # undefined

    else:
        raise NotImplementedError

    if not prompt_only:
        text += f"{example['answer']}"
    example['data_name'] = data_name
    example['text'] = text
    return example


def get_formatted_datasets(data_path: str, prompt_only: bool):
    """
    Get formatted datasets
    """
    data_name = os.path.basename(data_path).lower()

    # Load and format datasets
    if data_name == 'super_glue':
        data_names = ['boolq', 'cb', 'copa', 'rte', 'wic']
        splits = ['train', 'validation', 'test']
        formatted_datasets = {split: [] for split in splits}

        # Load and format datasets
        for _data_name in data_names:
            _datasets = load_dataset(path='super_glue', name=_data_name)
            print(f'Datasets: {_datasets}')
            _formatted_datasets = _datasets.map(
                lambda example: format_text(example, _data_name, prompt_only=prompt_only),
                batched=False, load_from_cache_file=False)
            for split in splits:
                formatted_datasets[split].append(
                    _formatted_datasets[split].select_columns(['data_name', 'text', 'num_choices', 'answer']))

        # Concatenate datasets
        for split in splits:
            formatted_datasets[split] = concatenate_datasets(formatted_datasets[split])
        formatted_datasets = DatasetDict(formatted_datasets)
        print(f'Formatted datasets: {formatted_datasets}')
        print(f"Text example:\n{formatted_datasets['train']['text'][0]}")
    else:
        # Load datasets
        if data_name in [
            'axb', 'axg', 'boolq', 'cb', 'copa', 'multirc',
            'record', 'rte', 'wic', 'wsc', 'wsc.fixed',
        ]:
            datasets = load_dataset(path='super_glue', name=data_name)
        elif data_name == 'openbookqa':
            datasets = load_dataset(path=data_path, name='additional')
        elif data_name == 'ai2_arc':
            datasets = load_dataset(path=data_path, name='ARC-Challenge')
        elif data_name == 'scienceqa':
            datasets = load_dataset(path=data_path)
            datasets = datasets.filter(lambda example: example["image"] is None)
        else:
            datasets = load_dataset(path=data_path)
        print(f'Datasets: {datasets}')
        print(f"Example: {datasets['train'][0]}")

        # Format datasets
        formatted_datasets = datasets.map(
            lambda example: format_text(example, data_name, prompt_only=prompt_only),
            batched=False, load_from_cache_file=False)
        print(f'Formatted datasets: {formatted_datasets}')
        print(f"Formatted example: {formatted_datasets['train'][0]}")
        print(f"Text example:\n{formatted_datasets['train']['text'][0]}")

    return formatted_datasets


if __name__ == '__main__':
    data_path = 'cb'
    _ = get_formatted_datasets(data_path=data_path, prompt_only=False)
