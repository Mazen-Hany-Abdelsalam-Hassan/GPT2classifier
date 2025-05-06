from src import TOKENIZER  , PADDINGTOKEN
from torch import  tensor
from config import DEVICE ,LABEL_DICTIONARY_IMDB ,ALLOWED_SEQ_LENGTH
def predict(text: 'str', model, tokenizer = TOKENIZER,
            padding_token:int= PADDINGTOKEN ,label_dict:dict= LABEL_DICTIONARY_IMDB ,
            Max_length:int = ALLOWED_SEQ_LENGTH):
    assert len(text) > 0 , "Empty String"
    encoded_text = tokenizer.encode(text)[:Max_length]
    encoded_text.append(padding_token)
    encoded_text = tensor([encoded_text])
    encoded_text = encoded_text.to(DEVICE)
    model.to(DEVICE)
    model.eval()
    result = model(encoded_text)[:,-1,:].argmax(axis=-1 ).to('cpu')
    inverse_label_dict = {value : key  for key , value in LABEL_DICTIONARY_IMDB.items()}
    print(result.item())
    return  inverse_label_dict[result.item()]

