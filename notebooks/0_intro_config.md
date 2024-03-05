---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true id="040c998d-09c3-48cc-99c0-d18bb12ca229" slideshow={"slide_type": "slide"} -->
# Laboratorio de Lingüística Computacional 🥼
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
### 0. Introducción al laboratorio y entornos de trabajo
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
* ¿Qué vamos a trabajar?
* Objetivo
* Git + Github
* Google colab
* ¿Cómo vamos a trabajar? Metodología de entrega de tareas y practicas
<!-- #endregion -->

<!-- #region editable=true id="8f152cc8-1daf-4f4e-a4ec-0c010fbc5155" slideshow={"slide_type": "slide"} -->
## Temario
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
### 1. Fundamentos
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Evolución del procesamiento del lenguaje natural
- Niveles de estudio del lenguaje natural  (fonología, morfología, sintaxis, semántica, pragmática)
- Aplicaciones

<center><img src="http://csunplugged.jp/csfg/csfieldguide/_images/AI-eliza-nlp-addiction.png"></center>
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
### 2. Pre-procesamiento del texto
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Propiedades estadísticas del lenguaje natural (ley de Zipf y otras leyes empíricas, entropía de los textos)
- Normalización y filtrado de textos
- Tokenización

<center><img src="https://www.freecodecamp.org/news/content/images/size/w2000/2021/10/IMG_0079.jpg" height=500 width=500></center>
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
### 3. Representaciones vectoriales de palabras (word embeddings)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Matriz documento-término
- Vectores de palabra estáticos
- Vectores de palabra contextualizados

<center><img src="https://kr.mathworks.com/help/examples/textanalytics/win64/VisualizeWordEmbeddingsUsingTextScatterPlotsExample_01.png"></center>
<!-- #endregion -->

<!-- #region editable=true id="ba2da4ef-1b93-4b68-825b-c46bb2f8b4f0" slideshow={"slide_type": "subslide"} -->
### 4. Aplicaciones generales y otros temas
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Grandes modelos de lenguaje (LLMs)
- Traducción automática
- Multilingüismo y consideraciones éticas

<center><img src="https://miro.medium.com/v2/resize:fit:1000/1*vxBU7p3eOhiavr6D5qSdKA.jpeg"></center>
<!-- #endregion -->

<!-- #region editable=true id="2eb7b951-6871-4762-ac66-5334bc81db54" slideshow={"slide_type": "slide"} -->
## Objetivo general
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Conocer herramientas altamente usadas en la industria y academia para ser unæ **destacadæ** practicante del *Natural Language Processing (NLP)*
- Que se sientan comodæs con estas herramientas para que puedan usarlas para sus propios proyectos o colaborar en otros
- Practicar lo que vean en clase :)

<center><img src="http://i0.kym-cdn.com/entries/icons/facebook/000/008/342/ihave.jpg"></center>
<!-- #endregion -->

<!-- #region editable=true id="c0095dd9-9ba9-4e64-8533-283a7ad6ccc4" slideshow={"slide_type": "slide"} -->
## ¿Qué vamos a trabajar?
<!-- #endregion -->

<!-- #region editable=true id="213c2791-f7f7-4eba-ab85-b088a5ae3bd3" slideshow={"slide_type": "fragment"} -->
- Este laboratorio que cubre los aspectos prácticos de la clase teórica
- Varias prácticas pequeñas y acumulativas
    - Procuraremos ir a la par con los temas de la clase principal
- Al final de cada práctica se dejará un ejercicio y opcionalmente un **EXTRA**
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Queremos que se sientan comodos con `Google colab`, `jupyter notebooks`, `git` y workflows con plataformas como Github
- Entregables a través de Github usando `git`, forks y pull requests (más al respecto a continuación)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
**NOTA: Es obligatorio pasar el laboratorio para aprobar la materia 😨 Aprobar es que las prácticas entregadas promedien mínimo 6.**
<!-- #endregion -->

<!-- #region editable=true id="0647ea84-d949-418a-a37a-6807d674a648" slideshow={"slide_type": "slide"} -->
## Sobre mi y contacto
<!-- #endregion -->

<!-- #region editable=true id="ff945f2f-7648-46f7-aa38-ccb7da61f306" slideshow={"slide_type": "fragment"} -->
- Me llamo Diego Barriga (asies, como el Señor Barriga)
- Trabajo como MLOps engineer en Mercado Libre
- Hice [mi tesis](https://github.com/umoqnier/otomi-morph-segmenter) sobre NLP y lenguas indígenas mexicanas
- Desarrollo software libre para Comunidad Elotl y LIDSoL
- Me late la cultura libre, andar en bici y uso arch BTW :p

<div style="display: flex; justify-content: space-between;">
    <a href="https://elotl.mx/" target="_blank"><img src="https://elotl.mx/wp-content/uploads/2020/07/logo_elotl_transparente-e1596147405682.png" style="width: 30%;"></a>
    <a href="https://lidsol.org/" target="_blank"><img src="https://www.lidsol.org/img/logo.png" style="width: 80%;"></a>
</div>
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
### Contacto

- dbarriga@ciencias.unam.mx
- Discord: @umoqnier
- Jueves de 4 a 6pm en el Laboratorio :p
<!-- #endregion -->

<!-- #region editable=true id="07b18364-3786-42a3-938e-3656db5b8d17" slideshow={"slide_type": "slide"} -->
## 0. Introducción al laboratorio y entorno de trabajo 💽
<!-- #endregion -->

<!-- #region editable=true id="4acac100-bc97-43f1-bd5f-3cfa30091658" slideshow={"slide_type": "slide"} -->
### Objetivos
<!-- #endregion -->

<!-- #region editable=true id="a72a8dc2-2093-4572-b234-7a8f4e4e2866" slideshow={"slide_type": "fragment"} -->
 - El alumno conocerá la herramienta `git` para el control de versiones y aprendera fundamentos de la misma usando la línea de comandos
 - Uso de plataforma Github para entrega de practicas
   - Forks
   - PRs 
 - Uso de plataforma Google Colab
<!-- #endregion -->

<!-- #region editable=true id="090edbec-84e0-4bff-81e7-b734755d03a2" slideshow={"slide_type": "slide"} -->
### ¿Qué es git y pa' que sirve?
<!-- #endregion -->

<!-- #region editable=true id="4b771e93-e717-4659-8787-a54c224f680e" slideshow={"slide_type": "fragment"} -->
Software libre cuyo proposito es versionar los archivos de un **directorio de trabajo** dado. Se le conoce como Sistema de control de versiones.

![Imagen de versiones de word](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.campingcoder.com%2Fpost%2F20180412-git-flow.png&f=1&nofb=1&ipt=a575ef3f95d4c42e34292c8e8f4a92e6c8d54399c506ad7269303b14d7524948&ipo=images)
<!-- #endregion -->

<!-- #region editable=true id="c5e8db82-2e99-43e2-aed7-3551c0f1f3d8" slideshow={"slide_type": "subslide"} -->
#### Sistemas de control de versiones
<!-- #endregion -->

<!-- #region editable=true id="5b5c16eb-f971-480e-b97d-750c0791c585" slideshow={"slide_type": "fragment"} -->
- Guardado de versiones de archivos
- Visualizar diferencias entre archivos
- Trabaja con texto principalmente
- Asociación de cambios con autoræs
- Regresar en el tiempo a versiones anteriores
<!-- #endregion -->

<!-- #region editable=true id="f810e75f-1075-4491-9021-99fbcb9be0de" slideshow={"slide_type": "subslide"} -->
#### Caracteristicas de git
<!-- #endregion -->

<!-- #region editable=true id="2b6a5a08-e3ae-4e0a-8e8b-5133cdcd5869" slideshow={"slide_type": "fragment"} -->
- Creado en el contexto del desarrollo del [kernel de linux](https://www.youtube.com/watch?v=5iFnzr73XXk)
- Enfoque distribuido
- Velocidad y simpleza
- Creación de multiples ramas
- Varias desarrolladoras trabajando en un mismo proyecto
<!-- #endregion -->

```python editable=true id="de0e9924-0890-4105-89a6-700d5a14bea4" outputId="cb39f3b5-e923-4356-f644-1f1277b6e9fb" slideshow={"slide_type": "subslide"}
%%HTML
<center><iframe width="960" height="615" src="https://www.youtube.com/embed/5iFnzr73XXk?controls=1"></iframe></center>
```

<!-- #region editable=true id="05e2cae4-cace-4533-828b-6358dff98a60" slideshow={"slide_type": "subslide"} -->
#### Areas de trabajo
<!-- #endregion -->

<!-- #region editable=true id="ccaaf4e7-ef47-4b7e-afbc-1d4ec00dabd9" slideshow={"slide_type": "fragment"} -->
Git trabaja con tres áreas principales:
1. Working directory
2. Staging area
3. Repositorio local (`.git`)
    - Aquí se guardan los commits (en tu computadora local)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
#### ¿Cómo mover archivos entre areas?
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- `git init` - Iniciar un repositorio en el directorio actual
- `vim archivo.c` - Modifica un archivo y git detecta cambios
- `git add` - Envia las modificaciones al *staging area*
- `git commit` - Guarda los cambios que estuvieran en el *staging area* en el directorio `.git`
- `git restore` - Se pueden deshacer cambios en el *staging area*
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
<center> <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fthachpham.com%2Fwp-content%2Fuploads%2F2015%2F04%2Fgit-staging-area.png&f=1&nofb=1&ipt=df8409aae278df92116d596f7d7768669566b858cebc4c0f51270c239b22a133&ipo=images">  </center>
<!-- #endregion -->

<!-- #region editable=true id="f26dbd9b-8009-44de-9f61-682a97083dda" slideshow={"slide_type": "subslide"} -->
#### Instalación de git (por si no lo tienen)
<!-- #endregion -->

<!-- #region editable=true id="01d373d7-9d4f-479c-852e-47ded7a38107" slideshow={"slide_type": "fragment"} -->
- [Guia de instalación](https://github.com/git-guides/install-git)
- [Descarga](https://git-scm.com/downloads)
<!-- #endregion -->

<!-- #region editable=true id="f5582a00-c0fd-4d1d-b09b-c0f572f7431e" slideshow={"slide_type": "slide"} -->
### Git crash course
<!-- #endregion -->

<!-- #region editable=true id="d2123ccf-3c70-405f-a5cc-cb7b1d541fe4" slideshow={"slide_type": "subslide"} -->
#### Configuraciones básicas
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- `git config user.name "Dieguito Maradona"`
- `git config user.email "maradona@gmail.com"`
    - Usa la bandera `--global` para configuraciones a nivel global
<!-- #endregion -->

<!-- #region editable=true id="4e2498dc-0985-429c-894c-76cd4227d31d" slideshow={"slide_type": "subslide"} -->
#### ¿Cómo obtener un repositorio?
<!-- #endregion -->

<!-- #region editable=true id="10171ef2-75a7-4507-98e8-45cd33a51f55" slideshow={"slide_type": "fragment"} -->
- Localmente - `git init`
- Remotamente - `git clone <url>`
<!-- #endregion -->

<!-- #region editable=true id="99007aec-f070-42fd-8797-51e0822203d0" slideshow={"slide_type": "subslide"} -->
#### `git add`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
Este comando agrega archivos al área de staging que es previa al área de cambios definitivos

- Agregar achivos individuales - `git add main.py`
- Agregar por extension - `git add *.cpp`
- Agregar todo en un directorio - `git add .`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
#### `git commit`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
Confirma los cambios que esten en el area de staging. Se crea un punto en la historia del proyecto

**Buenas prácticas**:

- El commit debe ser acompañado con un mensaje descriptivo de los cambios
- Es mejor hacer varios commits que hagan una cosa pequeña en lugar de un solo commit con muchos cambios
- Recomendable seguir algun estandar. Ya sea definido por el equipo o [algún otro](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
<center><img src="https://imgs.xkcd.com/comics/git_commit.png"></center>
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
#### `git status`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
Verificamos el estado actual del proyecto. ¿Qué cambios hubo y dónde?

- `.gitignore`: Archivo de texto que permite excluir archivos para que `git` los ignore
<!-- #endregion -->

<!-- #region editable=true raw_mimetype="" slideshow={"slide_type": "subslide"} -->
#### `git push` / `git pull`
<!-- #endregion -->

<!-- #region editable=true id="6281060f-4639-4caf-92e6-b082ec9bde7c" slideshow={"slide_type": "fragment"} -->
Para mandar y recibir cambios desde un repositorio remoto (o sea algún lugar fuera de nuetra PC gamer)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
#### `git log`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
Revisar el historial de commits del proyecto, sus mensajes y quien realizó los cambios
<!-- #endregion -->

<!-- #region editable=true id="28801fa7-9e13-4534-86e2-294499974c1f" slideshow={"slide_type": "subslide"} -->
#### Ramas

<center><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.perforce.com%2Fsites%2Fdefault%2Ffiles%2Fstyles%2Fsocial_preview_image%2Fpublic%2Fimage%2F2020-07%2Fimage-blog-git-branching-model.jpg%3Fitok%3DrJ4GurJ8&f=1&nofb=1&ipt=319db8a24f10e5417740fc07d1b0a52225c1ae0f1ee28fc384a4b001733510c3&ipo=images"></center>
<!-- #endregion -->

<!-- #region editable=true id="b65bf6dc-c133-40d5-8ada-efd0aead6855" slideshow={"slide_type": "subslide"} -->
- Parte fundamental del software `git` es su manejo de ramas (o *branches*)
- Considerada la *killer feature* de `git`
- Trabajar con ramas permite explorar nuevos desarrollos sin romper, ensuciar o arruinar nuestra rama principal
    - Considerenlo un universo paralelo (o multiverso)
    - Trabajar en diferentes versiones del software de forma simultanea
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
<center><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmemeguy.com%2Fphotos%2Fimages%2Fmeanwhile-in-a-parallel-universe-71332.gif&f=1&nofb=1&ipt=8c983e591b8dfbc6cf45daa0c858cc4a6fbd57fb9a3ada81158058c778c97c08&ipo=images"></center>
<!-- #endregion -->

<!-- #region editable=true id="4f753c10-c188-44e2-9a28-bc61deb281bf" slideshow={"slide_type": "subslide"} -->
#### Comandos útiles para manejo de ramas
<!-- #endregion -->

<!-- #region editable=true id="0cc76aab-fd68-49d6-9999-fee34c4a4c90" slideshow={"slide_type": "fragment"} -->
- `git checkout -b <nueva-rama>`
- `git branch`
- `git merge`
<!-- #endregion -->

<!-- #region editable=true id="398b9629-10da-4a5c-8bb2-dcec02592941" slideshow={"slide_type": "subslide"} -->
#### Remotos
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
- Recuerdan que existian los repositorios remotos?
- Podemos sincronizar nuestro trabajo local de una rama especifica con el servidor
    - `git push origin <nombre-rama>`
    - **NOTA:** `origin` es el nombre por defecto del respositorio remoto. Se puede llamar como queramos
- Tambien podemos agregar mas remotos
    - `git remote add <nombre-remoto> <url>`
<!-- #endregion -->

<!-- #region editable=true id="fe82b8d5-5d8b-4508-bf83-24edfe9cc604" slideshow={"slide_type": "subslide"} -->
#### Forks
<!-- #endregion -->

<!-- #region editable=true id="97ff5eb8-1694-468c-9f43-9883155731ca" slideshow={"slide_type": "fragment"} -->
- Un fork es una copia de un respositorio de GitHub pero que ahora es de tu autoría.
- Sin embargo, el respositorio original queda ligado con la copia que realizaste
- Por medio de forks es que podemos realizar contribuciones a respositorios ajenos 🤗
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
<center><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.freecodecamp.org%2Fnews%2Fcontent%2Fimages%2F2022%2F02%2FGitHub-Fork.gif&f=1&nofb=1&ipt=9436ce7f55c6189a5e9c21fc39bc6a82e3ceba07dc23fadfe8251c67699add4b&ipo=images"></center>
<!-- #endregion -->

<!-- #region editable=true id="0b50e9b1-2b59-4e53-af85-98b3f0e280c4" slideshow={"slide_type": "subslide"} -->
### Ejercicio: Crear un PR hacia el repositorio principal del laboratorio

<center><img src="https://cdn.ebaumsworld.com/mediaFiles/picture/718392/84890866.jpg" height=400 width=400></center>
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
### URL: https://github.com/umoqnier/cl-2024-2-lab

- El PR deberá crear una carpeta con su username de GitHub dentro de `practices/`
    - `practices/umoqnier/`
- Agrega un archivo llamado `README.md` a esta carpeta con información básica sobre tí. Ejemplo:
    - `practices/umoqnier/README.md`
    - Usar lenguaje de marcado [Markdown](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

```markdown
# Diego Alberto Barriga Martínez

- Número de cuenta: `XXXXXXXX`
- User de Github: @umoqnier
- Me gusta que me llamen: Dieguito

## Pasatiempos

- Andar en bici

## Proyectos en los que he participado y que me enorgullesen 🖤

- [Esquite](https://github.com/ElotlMX/Esquite/)
```
<!-- #endregion -->

<!-- #region editable=true id="2f3b3f40-4333-482e-a876-b73a8914b555" slideshow={"slide_type": "subslide"} -->
#### Conectando remotos nuevos
<!-- #endregion -->

<!-- #region editable=true id="ab9ad4e6-d677-4c7d-a250-408b25db6b97" slideshow={"slide_type": "fragment"} -->
- `git remote add <nombre-remoto> <url>`
- Comando util para agregar el remoto del lab ;)
<!-- #endregion -->

<!-- #region editable=true id="77e4476a-5d28-4fef-a597-afc705b72532" slideshow={"slide_type": "subslide"} -->
#### Sincronizar el repo local con un repo forkeado
<!-- #endregion -->

<!-- #region editable=true id="505d1c33-0ca4-4a1f-8a5e-20a59f926ce7" slideshow={"slide_type": "fragment"} -->
- `git pull <remoto> <rama-a-sync>`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "subslide"} -->
#### Git flow
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
<center><img height="900" width="900" src="https://wac-cdn.atlassian.com/dam/jcr:cc0b526e-adb7-4d45-874e-9bcea9898b4a/04%20Hotfix%20branches.svg?cdnVersion=1437"></center>
<!-- #endregion -->

<!-- #region editable=true id="3b62d3c9-0ca1-4a37-bb02-f3cc1dc1bb8b" slideshow={"slide_type": "slide"} -->
## Google colab
<!-- #endregion -->

<!-- #region editable=true id="a868cc2d-be66-4ea0-bec4-a337e5031e0a" slideshow={"slide_type": "fragment"} -->
- Plataforma para ejecutar código python desde un navegador
- Se requiere tener una cuenta de Google
- Recursos de computo compartidos
  - GPUs, TPUs 
- Los archivos se pueden guardar en tu Drive o puedes guardar copias en GitHub
- https://colab.research.google.com/
<!-- #endregion -->

<!-- #region editable=true id="a9505dc1-4aaf-4633-a337-451f9244f269" slideshow={"slide_type": "slide"} -->
### Flujo de trabajo
<!-- #endregion -->

<!-- #region editable=true id="e7f436cf-6293-4663-8f84-46b6e22bfec8" slideshow={"slide_type": "fragment"} -->
#### La primera vez
<!-- #endregion -->

<!-- #region editable=true id="334c9347-9387-4e96-8549-765c8d40e161" slideshow={"slide_type": "fragment"} -->
0. [ ] Crear cuenta de Github
1. [ ] Crear un fork del [repositorio de prácticas](https://github.com/umoqnier/cl-2024-2-lab)
2. [ ] Clonar **tu fork** en tu computadora
    - **OJO 👁️👄👁️**: No el repositorio princial, es tu fork el que debes clonar
<!-- #endregion -->

<!-- #region editable=true id="7f3e97ee-f90a-422a-81cf-792525bcb633" slideshow={"slide_type": "slide"} -->
### Entrega de práctica
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
0. Sincronizar rama `main` local con la del lab
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
1. Crear una **nueva rama**
    - Usa nombres descriptivos: `git checkout -b practica02/arboles-sintaxis`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
2. Crear una carpeta para la practica dentro de su carpeta personal
    - `mkdir practices/umoqnier/practica1`
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
3. Hacer su practica
    - Usaremos el plug-in *[jupytext](https://jupytext.readthedocs.io/en/latest/install.html)*
    - Lo que entregaran cada práctica es **un archivo `.py`** que será un export del notebook de su práctica
    - El archivo deberá estar dentro de la carpeta de la práctica en turno con todos sus archivos
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
4. Hacer commits de sus cambios y push **a su fork**
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
5. Crear su *Pull Request*
    - Despues de una revisión la práctica se acepta o en su defecto se notificará que se requieren cambios
    - Cuando se acepte la práctica sus cambios estarán en el repositorio principal
<!-- #endregion -->

<!-- #region editable=true id="cacd67f9-7552-43bc-bcf0-65d0e37dd170" slideshow={"slide_type": "slide"} -->
## Links útiles

- [Repositorio de prácticas del lab ⭐](https://github.com/umoqnier/cl-2024-2-lab)
- [Git branches](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
- [Write better commit messages](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/)
- [Guia sencilla a git](https://rogerdudler.github.io/git-guide/index.es.html)
- [¿Cómo forkear un repo?](https://docs.github.com/es/get-started/quickstart/fork-a-repo)
- [Hola mundo de Github](https://docs.github.com/es/get-started/quickstart/hello-world)
- [Sintaxis básica de Markdown](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Generación de llaves SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
<!-- #endregion -->
