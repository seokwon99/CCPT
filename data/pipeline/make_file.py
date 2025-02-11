import pandas as pd
import spacy
from tqdm import tqdm
tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

def df_preprocess(df):
    if 'root_rel_prompt' in df.columns:
        del df['root_rel_prompt']
    if 'modifier_rel_prompt' in df.columns:
        del df['modifier_rel_prompt']
    if 'noun_phrase_rel_prompt' in df.columns:
        del df['noun_phrase_rel_prompt']

    df['noun_phrase'] = df['noun_phrase'].apply(lambda x: " ".join(x.split()))
    df['sentence'] = df['sentence'].apply(lambda x: " ".join(x.split()))
    
    # filter if digit in noun phrase
    df = df[~df['noun_phrase'].str.contains(r'\d')]
    
    # filter if property is not descriptive
    df = df[df['property'].progress_apply(lambda x: any([token for token in nlp(x) if token.pos_ in ['ADJ', 'ADV', 'VERB']]))]
    
    # turn properties to original form
    df['property'] = df['property'].progress_apply(lambda x: nlp(x)[0].lemma_)
    
    df = df.drop_duplicates(subset=['noun_phrase', 'property'])
    return df

emergent_candidates = pd.read_csv("data/pipeline/241212_vera_processed.csv")
emergent_candidates = emergent_candidates[(emergent_candidates['individual_max'] < 0.5)]
emergent_candidates['emergence'] = emergent_candidates.apply(lambda x: (x['noun_phrase_rel'] - x['individual_max']), axis=1)
emergent_candidates = df_preprocess(emergent_candidates)
emergent_candidates = emergent_candidates[emergent_candidates['emergence'] > 0.2].reset_index(drop=True)
emergent_candidates = emergent_candidates.groupby('noun_phrase').apply(lambda x: x.nlargest(3, 'emergence'))
print(f"Emergent candidates: {len(emergent_candidates)}")
emergent_candidates.to_csv("data/pipeline/emergent_seeds.csv", index=False)

canceled_candidates = pd.read_csv("data/pipeline/241227_expanded.csv")
canceled_candidates = canceled_candidates[(canceled_candidates['individual_max'] > 0.5)]
canceled_candidates['cancellation'] = canceled_candidates.apply(lambda x: (x['individual_max'] - x['noun_phrase_rel']), axis=1)
canceled_candidates = df_preprocess(canceled_candidates)
canceled_candidates = canceled_candidates[canceled_candidates['cancellation'] > 0.4].reset_index(drop=True)
canceled_candidates = canceled_candidates.groupby('noun_phrase').apply(lambda x: x.nlargest(2, 'cancellation'))
print(f"Canceled candidates: {len(canceled_candidates)}")
canceled_candidates.to_csv("data/pipeline/canceled_seeds.csv", index=False)