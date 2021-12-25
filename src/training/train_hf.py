from transformers import FSMTForConditionalGeneration, FSMTTokenizer


if __name__ == '__main__':

    mname = "facebook/wmt19-en-ru"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    print(len(tokenizer.get_vocab()))

    input = "Machine learning is great, isn't it?"
    # input_ids = tokenizer.encode(input, return_tensors="pt")
    # outputs = model.generate(input_ids)
    # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(outputs.shape)
    # print(outputs)
    # print(decoded) # Машинное обучение - это здорово, не так ли?
    # print(model)