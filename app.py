import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np

@st.cache_resource
#---------------------------------------------------------------------------------------
def carrega_modelo():
  #https://drive.google.com/file/d/1VPpw18cagLfP8j6smglsCrarV0182Jkh/view?usp=drive_link
  url = 'https://drive.google.com/uc?id=1VPpw18cagLfP8j6smglsCrarV0182Jkh'
  
  gdown.download(url,'modelo_quantizado16bits.tflite')
  interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
  interpreter.allocate_tensors()
  
  return interpreter
#---------------------------------------------------------------------------------------
def carrega_imagem():
  uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png','jpg','jpeg'])
  if uploaded_file is not None:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))

    st.image(image)
    st.success('Imagem carregada com sucesso')
    
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    
    return image    
#---------------------------------------------------------------------------------------
def main():
  st.set_page_config(
    page_title="Classifica Folhas de Videira",
    page_icon=" "
  )

st.write("# Classifica Folhas de Videira!")

  # Carrega modelo
interpreter = carrega_modelo()

  # Carrega imagem
image = carrega_imagem()

  # Classifica

#--------------------------------------------------------------------------------------

if __name__==__main__:
  main()
