import os, sys
from templates import *
import json
from global_methods import *
import pickle as pkl

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json, os
from bert_score import score
import regex
import string
import torch
from tqdm import tqdm

def prepare_for_rag(database, query):

     
    # check if embeddings exist
    if not os.path.exists(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_observation.pkl'))):

        observations = []
        date_times = []
        context_ids = []
        for i in range(1,50):

            if 'session_%s' % i not in data:
                break
            if len(data['session_%s' % i]) == 0:
                continue    

            if 'session_%s_observation' % i not in data:
                print("Getting observations for session %s" % i)
                facts = get_session_facts(args, data, data, i)
                data['session_%s_observation' % i] = facts
                with open(ann_file, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                facts = data['session_%s_observation' % i]

            date_time = data['session_%s_date_time' % i]
            for k, v in facts.items():
                for fact, dia_id in v:
                    observations.append(fact)
                    context_ids.append(dia_id)
                    date_times.append(date_time)

        print("Getting embeddings for %s observations" % len(observations))
        embeddings = get_embeddings(args, observations, 'context')
        print(embeddings.shape)
        database = {'embeddings': embeddings,
                            'date_time': date_times,
                            'dia_id': context_ids,
                            'context': observations}

        with open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_observation.pkl')), 'wb') as f:
            pickle.dump(database, f)

    else:
        database = pickle.load(open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_observation.pkl')), 'rb'))
    
    print("Getting embeddings for %s questions" % len(data['qa']))
    question_embeddings = get_embeddings(args, [q['question'] for q in data['qa']], 'query')

    return database, question_embeddings


def get_rag_context(context_database, query_vector, args):

    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    sorted_context = [context_database['context'][idx] for idx in sorted_outputs[:args.top_k]]
    
    sorted_context_ids = [context_database['dia_id'][idx] for idx in sorted_outputs[:args.top_k]]
    sorted_date_times = [context_database['date_time'][idx] for idx in sorted_outputs[:args.top_k]]
    if args.rag_mode in ['dialog', 'observation']:
        query_context = '\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])
    else:
        query_context = '\n\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])

    return query_context, sorted_context_ids


def get_persona(name, age, gender, role, preferences=None):
    "Function to accept the choices and inputs entered by user in the first page of the system and convert it into a persona"

    preferences_string = "" #TODO
    persona = PERSONA_FROM_CHOICES.format(name, age, gender, preferences_string)
    return persona


def initialize_database(session_identifier, out_dir):
    "Function to initialize the database (facts about the user and their interactions and their embeddings for retrieval)"

    database_file = os.path.join(out_dir, session_identifier  + '.pkl')
    with open(database_file, 'wb') as f:
        pkl.dump({'history': {}, 'embeddings': {}, 'timestamps': {}}, f)


def get_session_facts(prompt_dir, conversation, session_idx):
    "Function to convert a series of dialogs into a set of facts about each speaker that can be saved to a database"

    # TODO: IMPORTANCE SCORE

    # Step 1: get events
    task = json.load(open(os.path.join(prompt_dir, 'fact_generation_examples_new.json')))
    query = CONVERSATION2FACTS_PROMPT
    examples = [[task['input_prefix'] + e["input"], json.dumps(e["output"], indent=2)] for e in task['examples']]
    
    input = task['input_prefix'] + conversation
    facts = run_json_trials(query, num_gen=1, num_tokens_request=500, use_16k=False, examples=examples, input=input)

    agent_a_embeddings = get_embedding(facts[agent_a['name']])
    agent_b_embeddings = get_embedding(facts[agent_b['name']])

    if session_idx > 1:
        with open(args.emb_file, 'rb') as f:
            embs = pkl.load(f)
    
        embs[agent_a['name']] = np.concatenate([embs[agent_a['name']], agent_a_embeddings], axis=0)
        embs[agent_b['name']] = np.concatenate([embs[agent_b['name']], agent_b_embeddings], axis=0)
    else:
        embs = {}
        embs[agent_a['name']] = agent_a_embeddings
        embs[agent_b['name']] = agent_b_embeddings
    
    with open(args.emb_file, 'wb') as f:
        pkl.dump(embs, f)
    
    return facts


def get_session_reflection(args, agent_a, agent_b, session_idx):

    # Step 1: get conversation
    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for dialog in agent_a['session_%s' % session_idx]:
        # if 'clean_text' in dialog:
        #     writer.write(dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n')
        # else:
        conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n'

    # Step 2: Self-reflections
    if session_idx == 1:
        agent_a_self = run_json_trials(SELF_REFLECTION_INIT_PROMPT.format(conversation, agent_a['name']), model='chatgpt', num_tokens_request=300)
        agent_b_self = run_json_trials(SELF_REFLECTION_INIT_PROMPT.format(conversation, agent_b['name']), model='chatgpt', num_tokens_request=300)
    else:
        agent_a_self = run_json_trials(SELF_REFLECTION_CONTINUE_PROMPT.format(agent_a['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_a['name']), model='chatgpt', num_tokens_request=300)
        agent_b_self = run_json_trials(SELF_REFLECTION_CONTINUE_PROMPT.format(agent_b['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_b['name']), model='chatgpt', num_tokens_request=300)

    # Step 3: Reflection about other speaker
    if session_idx == 1:
        agent_a_on_b = run_json_trials(REFLECTION_INIT_PROMPT.format(conversation, agent_a['name'], agent_b['name']), model='chatgpt', num_tokens_request=300)
        agent_b_on_a = run_json_trials(REFLECTION_INIT_PROMPT.format(conversation, agent_b['name'], agent_a['name']), model='chatgpt', num_tokens_request=300)
    else:
        agent_a_on_b = run_json_trials(REFLECTION_CONTINUE_PROMPT.format(agent_a['name'], agent_b['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_a['name'], agent_b['name']), model='chatgpt', num_tokens_request=300)
        agent_b_on_a = run_json_trials(REFLECTION_CONTINUE_PROMPT.format(agent_b['name'], agent_a['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_b['name'], agent_a['name']), model='chatgpt', num_tokens_request=300)

    reflections = {}
    reflections['a'] = {'self': agent_a_self, 'other': agent_a_on_b}
    reflections['b'] = {'self': agent_b_self, 'other': agent_b_on_a}

    return reflections


def init_llava(debug):

    if debug:
        return None, None
    else:
        # init model
        print("Initializing LlaVA model")
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model = model.cuda()
        return model, processor


def get_next_response(history, context_database, query_vectors, image=None, rag_mode=False):

    if args.rag_mode:
        assert args.batch_size == 1, "Batch size need to be 1 for RAG-based evaluation."
        context_database, query_vectors = prepare_for_rag(args, in_data, ann_file)
    else:
        context_database, query_vectors = None, None


    if image is not None:
        # use llava
        pass
    else:
        # get LLM response
        pass

    return response