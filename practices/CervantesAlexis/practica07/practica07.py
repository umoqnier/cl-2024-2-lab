# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="pYEzGgxBWYvL" outputId="b978ead5-7888-4a88-b459-b80d58bb53a6"
#from google.colab import drive
#drive.mount('/content/drive')

# + id="S8O8KPGqYHW8" colab={"base_uri": "https://localhost:8080/"} outputId="a7e1e90f-c861-4fd8-8862-a84caf0c1b3d"
# #%cd /content/drive/MyDrive/carpeta

# + id="AaZdT4fGeUW2"
# Creación del archivo de configuración
# Usando valores pequeños en vista de que tenemos un corpus limitado
# Para datasets grandes deberian aumentar los valores:
# train_steps, valid_steps, warmup_steps, save_checkpoint_steps, keep_checkpoint
SRC_DATA_NAME = "corpus.nah-filtered.nah.subword"
TARGET_DATA_NAME = "corpus.es-filtered.es.subword"

config = f'''# config.yaml

## Where the samples will be written
save_data: run

# Rutas de archivos de entrenamiento
#(previamente aplicado subword tokenization)
data:
    corpus_1:
        path_src: {SRC_DATA_NAME}.train
        path_tgt: {TARGET_DATA_NAME}.train
        transforms: [filtertoolong]
    valid:
        path_src: {SRC_DATA_NAME}.dev
        path_tgt: {TARGET_DATA_NAME}.dev
        transforms: [filtertoolong]

# Vocabularios (serán generados por `onmt_build_vocab`)
src_vocab: run/source.vocab
tgt_vocab: run/target.vocab

# Tamaño del vocabulario
#(debe concordar con el parametro usado en el algoritmo de subword tokenization)
src_vocab_size: 50000
tgt_vocab_size: 50000

# Filtrado sentencias de longitud mayor a n
# actuara si [filtertoolong] está presente
src_seq_length: 150
src_seq_length: 150

# Tokenizadores
src_subword_model: source.model
tgt_subword_model: target.model

# Archivos donde se guardaran los logs y los checkpoints de modelos
log_file: train.log
save_model: models/model.enfr

# Condición de paro si no se obtienen mejoras significativas
# despues de n validaciones
early_stopping: 4

# Guarda un checkpoint del modelo cada n steps
save_checkpoint_steps: 1000

# Mantiene los n ultimos checkpoints
keep_checkpoint: 3

# Reproductibilidad
seed: 3435

# Entrena el modelo maximo n steps
# Default: 100,000
train_steps: 3000

# Corre el set de validaciones (*.dev) despues de n steps
# Defatul: 10,000
valid_steps: 1000

warmup_steps: 1000
report_every: 100

# Numero de GPUs y sus ids
world_size: 1
gpu_ranks: [0]

# Batching
bucket_size: 262144
num_workers: 0
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 2048
max_generator_batches: 2
accum_count: [4]
accum_steps: [0]

# Configuración del optimizador
model_dtype: "fp16"
optim: "adam"
learning_rate: 2
# warmup_steps: 8000
decay_method: "noam"
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: "tokens"

# Configuración del Modelo
encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: 6
dec_layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout_steps: [0]
dropout: [0.1]
attention_dropout: [0.1]
'''

with open("/content/drive/MyDrive/neural_machine_translation/config.yaml", "w+") as config_yaml:
  config_yaml.write(config)

# + id="cMXhpanwN_uo"
# #!gdown --folder https://drive.google.com/drive/folders/1LXnb3iqeGJLAcIAtdMh80MciRhW8pnQG?usp=sharing -O /content/drive/MyDrive/prueba

# + id="3in0y5dUWjrL"
# !pip install OpenNMT-py -U

# + colab={"base_uri": "https://localhost:8080/"} id="0opBGtVeY2Cq" outputId="7318ce4a-cfa7-4f5f-89f9-9cc24ab53a81"
# %%time
# !onmt_translate -model models/model.enfr_step_3000.pt -src corpus.nah-filtered.nah.subword.test -output test.es-practice.translated -gpu 0 -min_length 1

# + colab={"base_uri": "https://localhost:8080/"} id="Ywn1TXU9bFEe" outputId="a7f392a6-2f0e-4799-d076-90bed8fb907e"
# !tail corpus.nah-filtered.nah.subword.test

# + colab={"base_uri": "https://localhost:8080/"} id="7sdIMMnPbQNs" outputId="9d7b20e2-5798-4188-dec6-0cfc4f38bba3"
# !python MT-Preparation/subwording/3-desubword.py source.model corpus.nah-filtered.nah.subword.test

# + colab={"base_uri": "https://localhost:8080/"} id="ickmrlxqdJr6" outputId="6859c131-e8fd-4877-c48b-5dcebba13423"
# !python MT-Preparation/subwording/3-desubword.py target.model test.es-practice.translated

# + colab={"base_uri": "https://localhost:8080/"} id="rsd9da3ndhc3" outputId="e5e3bebf-e2fb-4990-f1fb-048bb2dad29f"
# !tail corpus.es-filtered.es.subword.test.desubword

# + colab={"base_uri": "https://localhost:8080/"} id="y4YUw6tSdUX7" outputId="9b131e94-ab11-4ddb-a4e3-f4239618cf8a"
# !tail  test.es-practice.translated.desubword

# + colab={"base_uri": "https://localhost:8080/"} id="mIqLde8XdXjD" outputId="7cf63c38-8107-4eae-dfca-9703d2644348"
# !python translation_eval.py --system_output test.es-practice.translated.desubword --gold_reference corpus.es-filtered.es.subword.test.desubword --detailed_output

# + [markdown] id="FATKYBwqd5JK"
# ## Baseline
# ### Language	BLEU	ChrF (0-1) 
#
# ### Náhuatl	0.33	0.182
#
# -


