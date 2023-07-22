from proteinbert import load_pretrained_model

seq = 'ACDEFGHIKLMNPQRSTVWY'

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "D:\\github\\Machine-Learning-2023\\test\\bert\\" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 512)

input_ids = input_encoder.encode_X(seq , 512)

local_representations, global_representation = model.predict(input_ids)

print(local_representations.shape) # (20, 512, 26)
print(global_representation.shape) # (20, 8943)