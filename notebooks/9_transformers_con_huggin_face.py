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

# + [markdown] id="ebHxr0OmvE26"
# # 9. Transformers via Hugging Face

# + [markdown] id="O1P8rpCsw0IH"
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia2.giphy.com%2Fmedia%2FTw0rSdQs4fbJC%2Fgiphy.gif&f=1&nofb=1&ipt=f07b4382bc4ac4352b1ae008a847fca249f95f46816f6427add49ed63c1b0062&ipo=images)

# + [markdown] id="sH32qza812QV"
# ## Transformers
#
# - Es una arquitectura alternativa a redes recurrentes y convolucionales
# - Introducida en 2017 en el paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
# - Esta arquitectura influenció una gran cantidad de modelos. Podemos agruparlos como se muestra a continuación:
#     - GPT-like
#     - BERT-like
#     - BART/T5-like

# + [markdown] id="XVFeLYQS2sAP"
# ![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg)
# > Tomada de [Huggin Face - NLP Course](https://huggingface.co/learn/nlp-course/chapter1/4)

# + [markdown] id="gdZXkXBA3V1R"
# ### Los transformers son modelos del lenguaje

# + [markdown] id="EgNjMWYu3bBC"
# - Se entrenan de forma self-supervised
# - El objetivo se determina automaticamente computado de los inputs
#   - Precisa grandes corpus de entrenamiento
#   - Permite entendimiento estadistico del lenguaje
#

# + [markdown] id="yb5_xEXs4SAv"
# ### El problema de los transformers

# + [markdown] id="AJzG3Qo64fea"
# ![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png)

# + [markdown] id="ngswUktA9yZq"
# - Entrenarlos no es sencillos, requiere muchos datos y computo
# - Tienen un [impacto ambiental considerable](https://www.youtube.com/watch?v=ftWlj4FBHTg)
# - Compartir los modelos es una forma de reducir tiempo, costos e impacto ambiental
#     - [Hugging Face](https://huggingface.co/)
#     - https://paperswithcode.com/
#     - https://www.kaggle.com/models

# + [markdown] id="5PJFMrxS4jlb"
# ![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint.svg)

# + [markdown] id="-ynuF2Aj35Ss"
# ### Transfer learning y fine-tuning

# + [markdown] id="mqpxlOlT38L1"
# - Estos modelos pre-entrenados pasan por un proceso llamado *transfer learning*
# - Durante este se realiza el proceso de *fine-tuning* con datos anotados por humanos (aprendizaje supervisado)
#   - Por ejemplos: Si se tiene un modelo entrenado con un gran corpus en español se puede realizar el *fine-tuning* con un corpus de textos cientificos obtendremos un modelo especializado en *science/research*

# + [markdown] id="Q-V_wV-e_og9"
# ### Arquitectura

# + [markdown] id="fcjCbpf6_p7J"
# ![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg)

# + [markdown] id="e1fjwV2t_tSp"
# - Solo encoder: Tareas que precisan entendimiento del input como etiquetadores
# - Solo decoder: Tareas Que realizan generación de texto
# - Encoder-decoder (seq2seq transformers): Tareas de generación de texto que requiere el input como traducción automática

# + [markdown] id="a3DxstVNkxas"
# ### Atención

# + [markdown] id="aadHyMbQkzeM"
# - Es un componente crucial que consiste en son capas especiales llamadas *attention layers*
# - A grosso modo, esta capa le dirá al modelo como prestar atención solo a algunas partes de la oración cuando se crea la representación vectorial de cada palabra
#   - En el ejemplo "it" hace alusión a "animal" por lo que se tiene que poner más atención a esa sección de la oración
#

# + [markdown] id="pCO911x41TcO"
# ![](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)
#
# > Tomada de [Illustated Transformer](https://jalammar.github.io/illustrated-transformer/)

# + colab={"base_uri": "https://localhost:8080/"} id="1TB65hI_cX2y" outputId="2453bd1f-881b-4c63-e2a5-7e766a1f4f71"
# !pip install -U transformers
# !pip install datasets
# !pip install evaluate

# + [markdown] id="l93pHYnICa1o"
# ### Ejemplos
#
# - [Tareas disponibles](https://huggingface.co/docs/transformers.js/en/pipelines#available-tasks)
# - Modelos chiquitos - https://huggingface.co/distilbert

# + id="6_3zHGzJZE1B"
import transformers
from transformers import pipeline

transformers.logging.set_verbosity_error()

# + colab={"base_uri": "https://localhost:8080/"} id="zcSVe8EZZa8t" outputId="262e9e50-7428-44e1-ca33-4deebe875baf"
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
classifier(["i think this is amazing", "my bag is awful", "i want pizza"])

# + colab={"base_uri": "https://localhost:8080/"} id="ecktsxM5bCiA" outputId="93bd1349-2f1f-4d35-e048-cc2d80d0d16a"
generator = pipeline("text-generation", model="distilbert/distilgpt2", num_return_sequences=2, max_length=30)
generator("I think that")

# + colab={"base_uri": "https://localhost:8080/"} id="_GtKbRGcczcv" outputId="d32d044c-ff35-451e-f47b-ae5bbc7f47f2"
question_answerer = pipeline("question-answering", model="distilbert/distilbert-base-uncased-distilled-squad")
question_answerer(
    question="What is the course about?",
    context="I think this course Computer Linguistic was life changing stuff!!!",
)

# + id="MA1AXZ7AAQiE" colab={"base_uri": "https://localhost:8080/"} outputId="fc96a0ae-58ff-4520-883f-1b451334d339"
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
ner("My name is Diego and I live on Mexico City. I work at Mercado Libre doing software development and I teach a course at Universidad Nacional Autonoma de Mexico.")

# + [markdown] id="ciLjrNk9Ae3R"
# ### Explicando que hay detras de `pipeline()`

# + [markdown] id="SSkQsY7gjwdS"
# ![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline.svg)

# + id="LqX0z25vlzix"
raw_inputs = [
    "I think my bike is awesome",
    "I hate when you talk like goofy :("
]

# + colab={"base_uri": "https://localhost:8080/", "height": 180, "referenced_widgets": ["728d40cf5e3e45679a925218ed911856", "3542c6da26754f358218c6dd3544f54e", "1f454f92d85849d995802bce7552de5f", "3a94b1726b8044978256752dad242632", "c6d02c2d27b145409c9ea0330429bfce", "0248bdded7f94fd89c9f1949e584a668", "a860a01ab14946cdbd555f8e27e7b40c", "b5393bd336fa41e9966753c5a6626c0c", "ba8bda9415c2460697d92d26c87306d8", "cd4615e5c4ba41bcb469053fffc01884", "092caf32a530405684f148aa68d9dc3c", "74d6207b119f4ae1854c42f25f59cb27", "54b00173b6d44b3f9dc97acb1018a1aa", "3c86c858d50f4ddb8d02817f6eec2fd5", "63c6bc1505b248cf8d06f8840c790761", "eeed835a9a7b481da1a31e6d78ce2823", "e266f137c66748e2a95802f7b9b9bae7", "c3079692e6fb444f9a535bf47fd590e0", "9294a57207d641fbb65a8d0ef80b19b2", "696c05fbca43452691ebe9a952b40d60", "5ffd36a993e64abd932702dac4a27770", "7be15c69bbe9468090fe92ac47666b9b", "8e19ae853da34e7ea2f82018ac9cddb9", "f6816bc09f7e4e3cbd418cb891f0cdb2", "a6732c7fa19e4ab682726a5429131cd6", "40f166f1457f4c1b81e2f64ec6c3226c", "8e00e7ba77034ca1a8d890a6a3c2ed87", "f2c095a05bf14fd983a2bb544bdb7eb4", "8b3b93a956a94711be190c22939446bf", "44eb60e3d9074c23b2e4407a9bed4f7d", "ad525465602b4820a422b08eb10382c4", "2c6fd2aa9b30458c8d0b2d96bc2ad11e", "4dec0920ae6440ea858356980dc97fc0", "fe39192e3bba41bdaf40697430d5e923", "d7904d2515cf4b829c348f31cc815aca", "4c5c722a44f44ee5b098bde9ca8a7ada", "16e0912335394e82ab8a64ca5d38687f", "5171c7ca9ffd4551a3e88da351438dab", "b4b13946354f47bd9920aea8ab90e544", "9631e27f630342b5b3357490c9b22471", "17cd4c220fa941eb9c3bd08052fab78f", "35b6e98e88aa48c380a0528e0ac014d8", "ddadfcfa03994e3a858d85af08f573b9", "0065af2142c1488b8462d3607ec6fbb5"]} id="Toq9q51jj7AX" outputId="a14f41c8-1991-49b5-f146-de5578d6f07c"
classifier = pipeline("sentiment-analysis")
classifier(raw_inputs)

# + [markdown] id="4r4q92aiePwv"
# Vamos a desmenuzar esto suavecito. Vemos que en el `pipeline()` hay tres elementos importantes:

# + [markdown] id="MP3lagGJl87v"
# - Preprocesamiento
# - Modelo
#
# - Posprocesamiento

# + [markdown] id="BitOHVqNoZi0"
# ### 1. Preprocesamiento y tokenización

# + [markdown] id="7up5TFD-ogtJ"
# - Tokenización
# - Mapeo de tokens a números
# - Agregar elemento que sean útiles al modelo
#
# Estos pasos deben ser exactamente los mismos que los del modelo pre-entrenado. Para obtener esta información utilizaremos `AutoTokeninzer`

# + id="sE5SrDmbl7sP"
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# + colab={"base_uri": "https://localhost:8080/"} id="fcTlLQ7bmNr-" outputId="aaa2a25e-c866-47e1-bff8-ac740982c1d0"
inputs = tokenizer(raw_inputs, padding=True, truncation=False, return_tensors="pt")
inputs

# + [markdown] id="vSHn1RGYmjLA"
# #### Tokenizers

# + [markdown] id="UfOEJrYLpS2m"
# Los transformers solo aceptan tensores como entrada. Para modificar el tipo de tensor usamos `return_tensors=`

# + [markdown] id="fsrcBYveViO0"
# Recordemos que hay varios tipos de tokenizadores:
#
# - Word base
# - Character base
# - Subword tokenizers
#
# Cargar y guardar tokenizadores puede realizarse mediante los métodos `from_pretrained` y `save_pretrained` respectivamente.

# + colab={"base_uri": "https://localhost:8080/"} id="BJrNg9mJXAfF" outputId="9267f3d9-20b7-41ac-de46-74ac51b6c025"
tokenizer.save_pretrained("my_tokenizer")

# + colab={"base_uri": "https://localhost:8080/"} id="uH937SUcXPy6" outputId="d15f30e4-142a-4967-d7cb-6f96df88c775"
# !ls my_tokenizer

# + [markdown] id="MptSFEmUXgFv"
# ##### Encoding tokenization

# + [markdown] id="hRD3bLRtXjPP"
# Convertir los token a números es conocido como un proceso de *encoding* (NO confundir con *encoders* de los transformers) y consta de dos pasos:
#
# 1. Tokenizar
# 2. Mapear los tokens a números únicos (*ids*).
#
# Hay muchas formas de obtener los tokens por lo que es importante utilizar el mismo proceso de tokenizado con el que fue entrenado el modelo que queramos utilizar.

# + colab={"base_uri": "https://localhost:8080/"} id="QrV3Xy8RYSdw" outputId="f542459d-3569-499b-b180-30c300c3f90f"
sentence = raw_inputs[0] + " like tokenizers"
tokens = tokenizer.tokenize(sentence)
tokens

# + [markdown] id="c3WUQsC_Ywc8"
# Para mapear de tokens a números se necesita el vocabulario (*vocab.txt*). Es importante tener el mismo vocabulario con el que el modelo fue creado.

# + colab={"base_uri": "https://localhost:8080/"} id="rQGAO6DuebHn" outputId="202ac713-2ff9-48fb-ff92-d76e20fcc0ad"
ids = tokenizer.convert_tokens_to_ids(tokens)
ids

# + [markdown] id="fZ8auSl6e3Oy"
# ##### Decoding tokenization

# + [markdown] id="BYHJhsOWe6hu"
# Decodificar es tomar los ids y pasarlos a su representación textual. Notemos que además de pasar a texto mergeamos los subwords

# + colab={"base_uri": "https://localhost:8080/", "height": 34} id="0jTCQqh-e5V5" outputId="1ad6a46d-a37f-46c6-b363-6457c2043656"
decoded_string = tokenizer.decode(ids)
decoded_string

# + [markdown] id="280ffR4lpvM1"
# ### 2. El Modelo

# + [markdown] id="EFlWE77Hp0Rz"
# Asi como obtuvimos el tokenizador podemos obtener el modelo con el que queramos trabajar. Para ello usaremos `AutoModel`

# + colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["00decf2bce3f4f71a9757216be8492b5", "0a5aa270e9384aa9931226e837524457", "b18a66e345874077950917a6a0fcafcb", "a83a62d6847943f6b428954573ee3e1d", "7e483d4bb57943b286845f9574475f74", "87e7b36483254f228c926608673e2090", "c85e50644dfb435ca43aa33a2e8e9ef1", "9dfad9460493478da9dc3a7729c12306", "c67f8036070e45f5bb07b736087e1a2e", "a4f349461efd48cdbd558a9941f73c0f", "d52909c1e0304c46a3897032749d39f2"]} id="ZSouYz2_px3R" outputId="3ca50721-4470-418b-fa1a-0fa5435c0ce2"
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

# + [markdown] id="gzztg0gEsRzX"
# El modelo nos regresará los estados ocultos (*hidden states*) que será la representación **contextual** del texto de entrada.
#
# La salida contendrá usualmente:
#
# - Batch size: Oraciones procesadas
# - Sequence length: Longitud de la representación númerica de las oraciones
# - Hidden size: ?

# + colab={"base_uri": "https://localhost:8080/"} id="09XYnm_0s23f" outputId="2eecceb2-6f44-40e7-875b-d7eb12ec1c60"
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# + [markdown] id="WfsDCK-ZtRcb"
# ![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg)

# + [markdown] id="mcaiptrlwDF5"
# Dependiendo de la tarea a resolver cambiará la capa de *Head*. Siendo más específicos cambiaremos nuestro código para usar `AutoModelForSequenceClassification`

# + colab={"base_uri": "https://localhost:8080/"} id="pWG_hIoxwJWh" outputId="33fd0b1a-9a38-4886-ff26-98336dd7fdf1"
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)

# + [markdown] id="ZYQTscHE3Tjc"
# Se puede instanciar un modelo para entrenarlo desde cero (aunque ya vimos que no es recomendable por la cantidad de datos y computo necesarios)

# + colab={"base_uri": "https://localhost:8080/"} id="bnfnxvKF3b4C" outputId="360f900e-a1e1-420d-c380-37405cda11dc"
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

print(config)

# + [markdown] id="4iqw1fWR3tMq"
# Generalmente obtendremos mejores resultados si cargamos un modelo previamente entrenado

# + id="Tpi_WDPA30h8"
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")

# + [markdown] id="eMjmPcW6385v"
# Guardar un modelo es sencillo, solo usamos el análogo a `from_pretrained` que es `save_pretrainded`

# + id="ZOPJ_jto38BV"
model.save_pretrained("my_model")

# + id="TsiZsjDk4Gx3" outputId="dc468bf5-6de9-4f7e-9a8f-c78abed66a5a" colab={"base_uri": "https://localhost:8080/"}
# !ls my_model

# + [markdown] id="Ly6bbPBhjyQP"
# #### Padding y Attention Masks

# + [markdown] id="BGIRkhHPpade"
# ##### Padding

# + [markdown] id="iTXsm8S0j1cQ"
# Hasta ahora nuestro modelo realizó una predicción tomando en cuenta un par de oraciones de entrada:

# + colab={"base_uri": "https://localhost:8080/"} id="KDRZRbY_kAg8" outputId="f216a557-2ff6-49f5-b818-536ca347ed3e"
raw_inputs

# + colab={"base_uri": "https://localhost:8080/"} id="i-oFaS47nOiZ" outputId="5a8f9583-f0b7-4e34-c4ee-4b82e313f966"
inputs

# + colab={"base_uri": "https://localhost:8080/"} id="IgYzcJVwm3Fn" outputId="e121251a-ef66-4185-f85d-6e0afb94b728"
print(inputs["input_ids"])
print(inputs["attention_mask"])

# + colab={"base_uri": "https://localhost:8080/"} id="V0az08lYm-VG" outputId="f16ae3ea-7b2c-4429-a25a-3a951286c7bc"
tokenizer.pad_token_id

# + [markdown] id="Ah02u9AInoaS"
# ¿Qué pasa cuando nuestras oraciones no tienen la misma longitud? Nuestros tensores deben tener una forma rectangular para poder ser creados a partir de un *batch*.
#
# Para ello usamos un *padding* que será de utilidad para lograr tener vectores de forma rectangular

# + colab={"base_uri": "https://localhost:8080/", "height": 176} id="W362eWy4n0WK" outputId="7e5bcc46-6c16-4db2-de1f-543d76844c44"
another_ids = [
    [200, 200, 200],
    [200, 200]
]
torch.tensor(another_ids)

# + colab={"base_uri": "https://localhost:8080/"} id="s5mMj2LcoF-q" outputId="2fc837f3-42b6-44cc-f52a-a1dd4aa6ee6a"
another_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id]
]
torch.tensor(another_ids)

# + [markdown] id="iBYk_nklo07c"
# Mandamos los ids de nuestras secuencias de forma individual y en batch:

# + colab={"base_uri": "https://localhost:8080/"} id="CJueThfaouKc" outputId="09345760-ec10-49f0-ebd8-dae849b66c4a"
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

# + [markdown] id="qT6QIAIbo66G"
# Podemos ver que no estamos obteniendo el mismo resultado para los logits de la secuencia 2 ¿Porqué?

# + [markdown] id="5QEEWTxypUUg"
# ##### Attention mask

# + [markdown] id="xuHp1JEZpvHY"
# Las *attention masks* son vectores del mismo tamaño que los vectores de ids con 1's y 0's para indicar que tokens seran tomados en cuenta.

# + colab={"base_uri": "https://localhost:8080/"} id="qCeOfe3gpq7c" outputId="f8b91b13-24e2-486d-cf90-dfa4c7b6721d"
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)

# + [markdown] id="pye7IZl1sYCH"
# #### Truncation

# + [markdown] id="pfQjHt06sZO5"
# Los modelos de transformers tienen limitaciones en cuanto a la longitud de la oración se refiere. Muchos modelos solo pueden lidiar con entre 512 y 1024 tokens. Es importante tomar en cuenta esto:
#
# ```python
# sequence = sequence[:max_sequence_length]
# ```

# + colab={"base_uri": "https://localhost:8080/"} id="_gbRj0h6tfJD" outputId="c4a99ea7-8e9f-41bc-9d97-6f5de3b0f45b"
tokenizer.model_max_length

# + [markdown] id="lPWelRvTwfOn"
# ### 3. Posprocesamiento

# + colab={"base_uri": "https://localhost:8080/"} id="o8EySXrZwhZ5" outputId="5904a411-13b3-4269-b576-b86e29da44df"
outputs.logits

# + [markdown] id="xSsLCjWqwky8"
# Necesitamos posprocesar estos valores llamados *logits* (scores no normalizados). Para obtener probabilidades necesitamos usar la capa [*SoftMax*](https://en.wikipedia.org/wiki/Softmax_function).

# + colab={"base_uri": "https://localhost:8080/"} id="_bq70S6cxATQ" outputId="2610befd-5eb1-4723-ba66-2fb1c9f5f2a6"
import torch

logits = outputs.logits
softmax = torch.nn.Softmax(dim=-1)
probs = softmax(logits)
print(probs)

# + colab={"base_uri": "https://localhost:8080/"} id="Whg3qJZCxdwo" outputId="ac8e3b17-0fb5-4ba8-948b-a05051f3dd60"
model.config.id2label

# + [markdown] id="9a5IncphxlXn"
# Con esta información podemos concluir que el modelo asigno las siguientes labels:
# 1. *NEGATIVE*: 0.00001, *POSITIVE*: 0.99982
# 2. *NEGATIVE*: 0.99283, *POSITIVE*: 0.00071

# + [markdown] id="-K2tlEE_BJOr"
# ### Fine-tuneando modelos

# + [markdown] id="HdpVVyQ0xZeK"
# Para el fine-tuning usaremos el modelo de `bert-base-uncased`. Podemos encontrar la información del modelo en su [model-card](https://huggingface.co/google-bert/bert-base-uncased)

# + id="44Pga_JJx_nw"
checkpoint = "bert-base-uncased"

# + colab={"base_uri": "https://localhost:8080/", "height": 440, "referenced_widgets": ["cb223665d934471f9a96e9a1c77345ac", "33500ab045be445ca4108e4938a5d54d", "ce70ed58aa01448f81b0de41f9ec325c", "79ee8186151945c8a112c39405cfe935", "91528463873444d1a06780005cad4e6e", "f67dc5b2f3c74be697e7632c825b4716", "75909d67331c4de883d0c35ebc03e31d", "e52d4047d5ab4bbda28a185e738349cd", "13ac802362a343aeb66a96f565b32cd7", "4066079a5c67474b881ae6c376ce3645", "2f61dcdcdbe54386ad7951b540c289be", "f7ab1d06f5754a18aa83687f032eecb0", "e94cf79b2229419eb30033f0d7115e9a", "695f5097f0fe43f7a92f4d4db56a9109", "1a1ce615ac184fd399a3b1240c1bc699", "07c06ad940c94e6d885d7755111ce58a", "594199663b34495aa60d657a98fa3a35", "d6f44e6e6daa4a2aac08b45e33526e1b", "49dcbc4fbc4a444e9b40e93b96bb52c4", "aee46d82b8ec42e3ad0de0a1ec4b9919", "b4ba896f346d44b299051b0b52371d10", "3474c7aa4f534e50ace542a97b61b26b", "14e8e4dd0f1542a9b18a63bff47b4d6b", "7da5de7953634bc8b3855e25c6acd828", "37105aad0ac94d9c8094cb0efa8f2fb6", "94889516336947b0aef6dbaf85b01cdd", "c96b55b6e7fc4a0d81db0d84170afd48", "c1711944aadf4917971f5504d743dd07", "d9fd3d97ccfe4bf68e4b7d165e63f2f2", "0f7a9857041a4dec9d721778400573bf", "835a59f583f2487597dce57a6a141412", "ad66ec46beac4c43be88d1766712a750", "021bba246c3d45e8a69287dac7386460", "5f5060058f2a42b4a58f0e5b4b51f262", "30bd6d7e66b048a8965501a021fc6673", "f37520dc8aa347ac9cfec236d9416964", "aba626c644284f4cbea7673ddea47d14", "72bc0ae37a404e0bbdc6cfc7d78957d4", "e63e658f80cb408bb2942793dcb26bbc", "8530e38a57ae4d70b925f72bda61c744", "61884da109f84a8d84a358114f31f9bf", "48b5ff76a88d42e9a196fa78f803c513", "d8e0782503cb4b0fb02090c2cfe055a2", "dc47a89c7d004ff49a721642e37b45da", "d715206f4ae546b184a13487df2c37a6", "a8977780ad4249518faffa2178b2ff10", "5196f083687a477bb63cdcdc3c272d25", "b8d0848dd5ed4bad86d0f9de0a7616ba", "751d6aea99754d149fb1e81cfb0cb7b0", "92fe6185188c43c591cb176572be1f8f", "bcf85a28fc39439082f93dc6948db838", "f377b7cd3f9e4049beea3a7f18c7b3bf", "fde3568d4fbf49fbb6871a10385c0379", "42607dce37a840b0b28d9d7ba05e74c7", "eb400edcb68844e78ff87334ce840a15"]} id="NDYS8q4uawFj" outputId="293c35a0-372d-4752-8bc7-101257e45b5d"
from transformers import pipeline

unmasker = pipeline("fill-mask", model=checkpoint)
unmasker("This linguistics course will teach you all about [MASK] models", top_k=2)

# + [markdown] id="KDRsz_IXvYjv"
# #### Dataset

# + [markdown] id="hYduyC6VBIjT"
# El dataset que utilizaremos sera el **MRPC (Microsoft Research Paraphrase Corpus)** introducido en este [paper](https://www.aclweb.org/anthology/I05-5002.pdf).
#
# Consiste en ~5.8k oraciones con sus etiquetas indicando si son oraciones parafraseadas o no (o sea que ambas oraciones quieren decir lo mismo). Mas infromación en la [página del dataset](https://huggingface.co/datasets/nyu-mll/glue#mrpc).

# + colab={"base_uri": "https://localhost:8080/"} id="eXn-WRR8BpMl" outputId="14e30c2a-e22d-44c2-90dc-125b742adb23"
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

# + colab={"base_uri": "https://localhost:8080/"} id="Wv3ull86DI-a" outputId="b377eb03-3898-4a4b-b243-074262bb7d6b"
train_dataset = raw_datasets["train"]
train_dataset[0]

# + [markdown] id="b5xEL-3-Dfw2"
#
# Obtengamos mas información hacerca del `'label'`

# + colab={"base_uri": "https://localhost:8080/"} id="5Jdy7Tm0Dk14" outputId="eb21615a-c29e-43a3-a684-7735de94d476"
train_dataset.features

# + colab={"base_uri": "https://localhost:8080/", "height": 34} id="OQfYIrarE7D5" outputId="7fa6009d-93d1-4bfb-c562-58924161fb94"
checkpoint

# + [markdown] id="9q3KPoHoD5GH"
# #### Preprocesamiento

# + [markdown] id="YcneUoQOFK0C"
# La tarea requiere que le pasemos al modelo un par de oraciones para determinar si efectivamente si quieren decir lo mismo. `tokenizer()` puede lidiar con este caso

# + colab={"base_uri": "https://localhost:8080/"} id="RO2oYFI0Fb82" outputId="c9713589-3f30-466f-c717-5f6b5fe35a28"
inputs = tokenizer("The first sentence", "The second sentence!!!")
inputs

# + [markdown] id="GIgnrW6KFpm0"
# `token_types_ids` le indica al modelo que tokens corresponden a la primera oración y cuales a la segunda

# + colab={"base_uri": "https://localhost:8080/"} id="2eJiBDNzFype" outputId="85dfb021-d580-48bf-b33c-1ef733d377b4"
tokenizer.convert_ids_to_tokens(inputs["input_ids"])

# + [markdown] id="YCM9yQIVGHyt"
# Vemos que además el tokenizador agrego los tokens especiales `[CLS]` y `[SEP]`

# + id="-L4OkcBcH0-V"
from datasets.dataset_dict import DatasetDict

def do_tokenization(example: DatasetDict) -> dict:
    # Dejamos fuera padding=True por una buena razón ¿Cúal?
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# + [markdown] id="9HICxu7bIKXL"
# Queremos mantener los datos del tipo `DatasetDict` por ellos utilizaremos el método `Dataset.map()` aplicando una función de tokenizado

# + colab={"base_uri": "https://localhost:8080/", "height": 347, "referenced_widgets": ["59820c0f304f48559d68ba764f8456b5", "c2916237667a4c1ebd09b8d15c69ccbc", "7eff8d8ba1434cdc93347ddb8fef4ce2", "8f5a33740d864810b941abcd789f3277", "dac8252bb9104feaa6d00b10c5acc931", "2c32a55b2ba94b54942ef0619e4cdbde", "49a57aa81ead45ceb969e532e3f82530", "2e8924844e4a485f9f047d17d5c8a3f8", "c64f51bd577448708cc90981048ab220", "6f906f1e44614bbfac30f08bcf4f7c9c", "5b9bace469654ead99719c66e1604b90"]} id="2vDi6S3QD6X9" outputId="0ec4d297-b632-4720-d2cd-08d8b959db80"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_dataset = raw_datasets.map(do_tokenization, batched=True)
tokenized_dataset

# + [markdown] id="p3ffCklYJARl"
# ##### Dynamic padding

# + [markdown] id="2xolE9sHJ3Zx"
# No aplicamos el pagging en la función de tokenizado porque no es optimo
#
# El *dynamic padding* permite aplicar padding dependiendo de la oración más grande en el batch
#
# Aplicar esta técnica puede no ser optima para cierto hardware como las *TPUs* que esperan siempre un padding fijo
#

# + id="6aXEwTFsKijq"
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

# + colab={"base_uri": "https://localhost:8080/"} id="Jfa-l62GKxrv" outputId="d035566c-bc99-4769-d717-7725bafda625"
samples = tokenized_dataset["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]

# + colab={"base_uri": "https://localhost:8080/"} id="gVR9C0GeK8K7" outputId="8d03662e-2c49-4641-80d2-fec735fe6089"
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

# + [markdown] id="25ZfhB5SLDWR"
# Vemos como en este batch simulado de 8 elementos el padding queda en 67

# + [markdown] id="_uiB5LKova3t"
# #### Entrenamiento

# + id="Rqg1-W7obLT6"
from transformers import TrainingArguments

training_args = TrainingArguments("my-trainer")

# + colab={"base_uri": "https://localhost:8080/"} id="_2wVA33odbtN" outputId="9d5f260d-43d2-4167-d63a-7d4e1038f039"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="wrSSB09wdRjf" outputId="d178ecf3-7d33-49e2-802a-34e05e5166b2"
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# + id="ftLDqXqedf-9"
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# + [markdown] id="LkmX7sjQd6La"
# ##### Fine-tuning time!!!

# + colab={"base_uri": "https://localhost:8080/", "height": 173} id="Iv5-iJhTdoVZ" outputId="38ac0509-2dd3-4d01-a8d1-f6250a5cf257"
trainer.train()

# + [markdown] id="OeubquckeAkc"
# El modelo solo muestra el `loss` pero no nos dice que tan bien o mal esta prediciendo nuestro modelo ya que no indicamos una forma de hacerlo.

# + [markdown] id="vCH0IwY7jvj9"
# #### Probando el modelo manualmente

# + id="LWtMoU1CeluB"
other_model = AutoModelForSequenceClassification.from_pretrained("my-trainer/checkpoint-500", local_files_only=True)

# + colab={"base_uri": "https://localhost:8080/"} id="6yZ8Lzode5BT" outputId="3fce8171-53f6-4d46-8a56-c1848e4387a4"
raw_inputs = [
    "This bike is just like other but without breaks. I called it fixie gear",
    #"Fixie bikes are bikes that does not have breaks"
    "I like turtles"
]
inputs = tokenizer(raw_inputs[0], raw_inputs[1], return_tensors='pt')
outputs = other_model(**inputs)
outputs.logits

# + colab={"base_uri": "https://localhost:8080/"} id="SoG47DgVjMDd" outputId="1c1fffac-5fa8-4ee5-c59d-4e6f1637abd3"
import torch

logits = outputs.logits
softmax = torch.nn.Softmax(dim=-1)
probs = softmax(logits)
print(probs)

# + [markdown] id="YUAQvxHqvm3k"
# #### Evaluación

# + colab={"base_uri": "https://localhost:8080/"} id="_wh0JMO9gQVV" outputId="1f02a60d-0e07-432b-e104-bb59ecdbc5db"
tokenized_dataset["validation"]

# + colab={"base_uri": "https://localhost:8080/", "height": 34} id="DwKfE1k0foQ4" outputId="8dbebf39-163b-437a-f98b-af6da7217ef8"
predictions = trainer.predict(tokenized_dataset["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="fT-y8xFdfuVh" outputId="a9d7c49f-6136-463a-9367-35228b099794"
predictions.metrics

# + id="pZm2l6m-gFZH"
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

# + colab={"base_uri": "https://localhost:8080/"} id="bfTT9-t-gHqE" outputId="8bcf1a43-93c8-4e11-bbc1-2a967d7e17d9"
preds

# + [markdown] id="D9plFQrfj9pn"
# Para obtener el golden standard vamos a descargar las metricas asociadas al dataset. Utilizaremos el método `compute()`

# + colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["3981c33ba3f54fafa5ccf02777f1a3e4", "73b4dc6874a842cf968dcfacbf5ba079", "52aab4b85c904ecfb60fbb66fe68f870", "7807903c66a142a7a7aaa374bd21f53a", "8ff333a63d904619aa1438b09afe4433", "24fbd1c2ea9649afa79a561e31c9213e", "a686171b58d54c84bf95c8d8fec9d200", "426bafa7bbb84f32ab150f9ca3a98d14", "4fb49aedadf74da5ba40d4c6367cc175", "13f2c2aa43454eb9a65d6293a6960321", "f79d1241bbdf49748abac1c71a3a7c7f"]} id="YKBdj5jXj311" outputId="2b1d144a-1107-4efe-8d80-bae4041b9f52"
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)


# + [markdown] id="Ep9_nU0jkoZv"
# Con esto podemos crear una función que compute las métricas de evaluación para reportar el desempeño del modelo en el entrenamiento

# + id="TKADpv1tkz_Y"
def compute_metrics(eval_preds: np.array) -> dict:
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# + colab={"base_uri": "https://localhost:8080/"} id="kqDNDWokk9Nz" outputId="c4ee20eb-93de-4d4e-97a7-40fbfdc41e73"
training_args = TrainingArguments("my-metric-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # Esto es nuevo O:
    compute_metrics=compute_metrics,
)

# + colab={"base_uri": "https://localhost:8080/", "height": 283} id="jSx4FQoGlKIq" outputId="fe8e7585-86a6-43f5-816a-7f525d2748a3"
trainer.train()

# + [markdown] id="hb3ZBLjCCeAd"
# ## Limitaciones y Bias de modelos

# + [markdown] id="tczdRLOjmDwV"
# Recordemos que estamos utilizando modelos preentrenados con grande cantidades de datos, mucho con datos de internet.
#
# Esto implica que los modelos pueden estar cesgados

# + colab={"base_uri": "https://localhost:8080/"} id="3BxUJUMlfszg" outputId="cf6d8d91-9e5b-4f96-aac7-6a7fe234bc22"
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

# + [markdown] id="ufXA04SDmi9N"
# Es importante tener en cuenta que estos cesgos presentes en los modelos que pueden arrojar predicciones sexistas, homofobicas, racistas por mencionar algunas.
#
# Realizar fine-tuning no hará que estos cesgos intrinsecos desaparezcan.

# + [markdown] id="7vd4LDQPnHZ-"
# #### Una última cosa

# + [markdown] id="kTxM4E7hnK6g"
# ![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjg4b3Z1bXh4Y3ZwcXBtaXJ4eTE5aHhmaTJtY3RqYjljeHk4c2Z5OSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5IT69msgpaOcg/giphy.gif)

# + [markdown] id="A-1Z6lbhoX5q"
# # Referencias
#
# - El código e imagenes presentadas en esta práctica son una adaptación de las secciones 1, 2 y 3 del [curso NLP de Hugging Face](https://huggingface.co/learn/nlp-course/chapter1/1)
